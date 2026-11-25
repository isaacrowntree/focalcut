"""A still photo made a spatial scene -- the way iOS and macOS animate a photo on the lock screen or the wallpaper:
the people lift off the background, and as a virtual camera drifts in and across, near things move more than far
ones, so the photo opens into depth.

Two stages, so that the render needs no GPU:

1. layers(photo) -- once per photo, in an environment with torch (the reel's ML venv). SAM 3 finds the people, the
   subject is the most salient and nearest of them (and whoever stands with them), BiRefNet gives their edge
   (masks.refine: hair-level, kept to the people); Depth Anything V2 gives how near each pixel is; the background
   behind the subject is filled (Stable Diffusion inpainting where it is cached, else OpenCV's) so the camera can
   look past them. Written to <dir>/{photo,alpha,plate}.png and depth.png (16-bit, 1 = nearest).
2. render(dir, out, ...) -- numpy and OpenCV: every frame the background is remapped pixel by pixel by its own
   depth (a dolly: nearer pixels grow and shift more) and the subject, cut out, moves and grows a little more
   than what is behind it; the two composited, piped to ffmpeg.

`focalcut fx spatial PHOTO --out clip.mp4 [--size 1080x1920] [--seconds 6]` does both (the first needs torch).
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

PUSH = 0.04              # how far the camera moves in over the shot (enough that the frame's edge never shows)
SWAY = 0.022             # how far it orbits across (of the frame width): what is far slides at most about half
                         # of this against the subject -- the revealed strip stays under 2% of the frame
RISE = 0.006             # and up (a slight arc, as a hand-held phone moves)
POP = 1.0                # the subject's push against what is at its own depth (1: it holds still, the orbit's
                         # pivot -- the background does the moving, as on the lock screen)
FILL_STRENGTH = 0.6     # how far the fill departs from the surroundings carried in
LAYERS_VERSION = 8      # (8: the whole group facing us)
# (7: big-LaMa fill, the subject's colours re-estimated in its edge)
# (2: colour profiles converted, the subject the people facing us; 3, 4: the SD fill
#                          steered; 5: the real background mirrored in instead; 6: the fill only under the
#                          subject, the photo's own background kept round them)
FILL_PROMPT = "the background of the photo continued naturally, same light, no people in front, photograph"


def _ease(x):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0))


def _faces(rgb):
    """Frontal faces (x, y, w, h in pixels) -- YuNet, as the people step uses."""
    import cv2

    from ..reel.people import models
    h, w = rgb.shape[:2]
    s = min(1.0, 1600 / max(h, w))
    small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (int(w * s), int(h * s)))
    det = cv2.FaceDetectorYN.create(models()["det"], "", (small.shape[1], small.shape[0]), 0.7, 0.3, 5000)
    _, f = det.detect(small)
    return [] if f is None else [x[:4] / s for x in f]


def subject_alpha(rgb):
    """The subject's matte (0..1): of SAM 3's people, the most salient and nearest -- and whoever stands with them
    showing their face as large (two friends arm in arm), never someone merely touching them (a back turned in the
    crowd behind). Its edge is BiRefNet's, kept to a narrow band round their own masks (a matte that runs on into a
    neighbour would carry them along, half transparent: a ghost)."""
    import cv2

    from ..reel import subject_engine as E
    people = E.people(rgb)
    sal, dep = E.maps(rgb)
    if not len(people):                                   # no people: what stands out
        return E.matte(rgb), dep
    faces = _faces(rgb)

    def face_of(m):                                       # the biggest face whose centre is in this person
        hs = [f[3] for f in faces if m[min(int(f[1] + f[3] / 2), m.shape[0] - 1), min(int(f[0] + f[2] / 2),
                                                                                      m.shape[1] - 1)]]
        return max(hs, default=0.0)
    score = []
    for m in people:
        a = m.mean()
        score.append(float(sal[m].mean() * (0.4 + dep[m].mean()) * np.sqrt(a)) if a > 0 else 0.0)
    order = list(np.argsort(score)[::-1])
    top = order[0]
    keep, f0 = [top], face_of(people[top])
    grow = cv2.dilate(people[top].astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    near = max(41, int(0.03 * max(rgb.shape[:2])) | 1)  # (a step apart still counts as together)
    grow = cv2.dilate(people[top].astype(np.uint8), np.ones((near, near), np.uint8)) > 0
    added = True
    while added:                                          # beside the group (anyone already in it), facing us,
        added = False                                     # their face as big: with them -- a whole row of
        for k in order[1:]:                               # friends, not just whoever stands next to the first
            if k in keep:
                continue
            fk = face_of(people[k])
            if score[k] >= 0.3 * score[top] and (people[k] & grow).any() and f0 > 0 and fk >= 0.55 * f0:
                keep.append(k)
                grow |= cv2.dilate(people[k].astype(np.uint8), np.ones((near, near), np.uint8)) > 0
                added = True
    coarse = np.any(people[keep], axis=0).astype(np.uint8)
    others = np.any(people[[k for k in range(len(people)) if k not in keep]], axis=0) if len(people) > len(keep) \
        else np.zeros_like(coarse, bool)
    H, W = coarse.shape
    ys, xs = np.nonzero(coarse)
    pad = int(0.05 * max(H, W))
    x0, x1, y0, y1 = max(0, xs.min() - pad), min(W, xs.max() + pad), max(0, ys.min() - pad), min(H, ys.max() + pad)
    matte = np.zeros((H, W), np.float32)
    matte[y0:y1, x0:x1] = E.matte(rgb[y0:y1, x0:x1])
    r = max(3, int(0.006 * max(H, W)))                    # the edge band: a few pixels either side of their masks
    k_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    band = cv2.GaussianBlur(cv2.dilate(coarse, k_).astype(np.float32), (0, 0), r / 2)
    core = cv2.GaussianBlur(cv2.erode(coarse, k_).astype(np.float32), (0, 0), r / 2)
    a = np.maximum(matte * band, core)
    a[others & (coarse == 0)] = 0.0                       # (never anyone else's pixels)
    a = np.clip((a - 0.15) / 0.7, 0, 1)                   # a firm edge: no veil of half-there pixels
    return a * a * (3 - 2 * a), dep


def read_photo(photo):
    """The photo as sRGB / Rec. 709 RGB (uint8): an embedded colour profile (a Rec. 2020 or Display P3 export) is
    converted, not ignored -- read raw, a wide-gamut photo comes out flat and shifted."""
    import io

    from PIL import Image, ImageCms
    im = Image.open(photo)
    im.load()
    icc = im.info.get("icc_profile")
    im = im.convert("RGB")
    if icc:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        dst = ImageCms.createProfile("sRGB")
        intent = ImageCms.Intent.PERCEPTUAL if ImageCms.isIntentSupported(
            src, ImageCms.Intent.PERCEPTUAL, ImageCms.Direction.INPUT) else ImageCms.Intent.RELATIVE_COLORIMETRIC
        im = ImageCms.profileToProfile(im, src, dst, renderingIntent=intent, outputMode="RGB")
    return np.asarray(im)


LAMA = os.path.expanduser("~/.cache/focalcut/big-lama.pt")   # big-LaMa (Apache-2.0, advimman/lama) as TorchScript


def fill_lama(rgb, hole, side=1024):
    """The hole filled by big-LaMa: it continues the texture round it (sand, leaves, rock) and does not invent
    things, as a diffusion inpainter does in a tall hole. Run on a crop round the hole at most `side` pixels on its
    longer edge (the scale it works best at); only the hole's pixels change."""
    import cv2
    import torch
    H, W = hole.shape
    ys, xs = np.nonzero(hole)
    pad = int(0.08 * max(H, W))
    x0, x1, y0, y1 = max(0, xs.min() - pad), min(W, xs.max() + pad), max(0, ys.min() - pad), min(H, ys.max() + pad)
    crop, ch = rgb[y0:y1, x0:x1], hole[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    s = min(1.0, side / max(h, w))
    sw, sh = int(np.ceil(w * s / 8) * 8), int(np.ceil(h * s / 8) * 8)
    img = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    m = (cv2.resize(ch.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
    dev = "cpu" if os.environ.get("FOCALCUT_DISABLE_CUDNN") else "cuda"   # (the TorchScript model has cuDNN baked
    model = torch.jit.load(LAMA, map_location=dev).eval()                # in: where cuDNN is broken, the CPU)
    with torch.inference_mode():
        out = model(torch.from_numpy(img.transpose(2, 0, 1))[None].to(dev), torch.from_numpy(m)[None, None].to(dev))
    res = (out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    del model
    torch.cuda.empty_cache()
    painted = cv2.resize(res, (w, h), interpolation=cv2.INTER_CUBIC)
    outimg = rgb.copy()
    soft = cv2.GaussianBlur(ch.astype(np.float32), (0, 0), 1.5)[..., None]
    outimg[y0:y1, x0:x1] = (painted * soft + crop * (1 - soft)).astype(np.uint8)
    return outimg


def foreground(rgb, alpha):
    """The subject's own colours where its matte is soft (hair, edges): pymatting's foreground estimate, so no trace
    of what stood behind them travels with them as the background moves (a fringe, a ghost)."""
    try:
        from pymatting import estimate_foreground_ml
    except ImportError:
        return rgb
    f = estimate_foreground_ml(rgb.astype(np.float64) / 255.0, alpha.astype(np.float64))
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def fill_mirror(rgb, hole):
    """The hole filled row by row with the real background either side of it, mirrored inward (the left half from
    the left, the right from the right), then a soft blend where they meet in the middle. The camera only ever shows
    a narrow strip beside the subject's outline, and this keeps that strip the photo's own sand, leaves and rock --
    nothing invented. Rows with background on one side only take it from that side."""
    import cv2
    out = rgb.copy()
    H, W = hole.shape
    for y in range(H):
        row = hole[y]
        if not row.any():
            continue
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        for x0, x1 in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            n = x1 - x0
            xs = np.arange(x0, x1)
            left = np.clip(2 * x0 - 1 - xs, 0, W - 1)     # mirrored about the run's left edge
            right = np.clip(2 * x1 - 1 - xs, 0, W - 1)    # and its right
            if x0 == 0:
                left = right
            if x1 == W:
                right = left
            w = np.clip((xs - x0) / max(n - 1, 1), 0, 1)[:, None]
            w = np.clip((w - 0.35) / 0.3, 0, 1)           # (each side's own until near the middle)
            out[y, x0:x1] = (rgb[y, left] * (1 - w) + rgb[y, right] * w).astype(np.uint8)
    return cv2.GaussianBlur(out, (0, 0), 0.8) * hole[..., None] + rgb * ~hole[..., None]


def fill(rgb, hole, prompt=FILL_PROMPT):
    """The photo with `hole` (bool) painted over: Stable Diffusion inpainting on a crop around it (cached model),
    else OpenCV's Telea fill. Only the hole changes."""
    import cv2
    H, W = hole.shape
    ys, xs = np.nonzero(hole)
    if not len(xs):
        return rgb.copy()
    pad = int(0.06 * max(H, W))
    x0, x1, y0, y1 = max(0, xs.min() - pad), min(W, xs.max() + pad), max(0, ys.min() - pad), min(H, ys.max() + pad)
    crop, ch = rgb[y0:y1, x0:x1], hole[y0:y1, x0:x1]
    out = rgb.copy()
    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        from PIL import Image
        pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",
                                                              torch_dtype=torch.float16, safety_checker=None)
        pipe = pipe.to("cuda")
        h, w = crop.shape[:2]
        s = 768 / max(h, w)
        sw, sh = max(8, int(w * s) // 8 * 8), max(8, int(h * s) // 8 * 8)
        g = torch.Generator("cuda").manual_seed(7)
        base = cv2.inpaint(crop, ch.astype(np.uint8) * 255, 25, cv2.INPAINT_TELEA)   # the surroundings' colours
        res = pipe(prompt=prompt, strength=FILL_STRENGTH,     # carried in first; SD only textures them (left to
                   negative_prompt=("person, people, figure, face, body, dress, gown, cloth, "   # itself it invents
                                    "fabric, statue, object, text, watermark, blurry"),          # a scene)
                   image=Image.fromarray(base).resize((sw, sh)),
                   mask_image=Image.fromarray((ch * 255).astype(np.uint8)).resize((sw, sh)),
                   height=sh, width=sw, num_inference_steps=40, guidance_scale=7.0, generator=g).images[0]
        painted = cv2.resize(np.asarray(res), (w, h), interpolation=cv2.INTER_CUBIC)
        del pipe
        torch.cuda.empty_cache()
    except Exception as e:                                # (no diffusers or model: the plain fill)
        print(f"[spatial] SD inpainting unavailable ({e}); OpenCV fill", flush=True)
        painted = cv2.inpaint(crop, ch.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)
    soft = cv2.GaussianBlur(ch.astype(np.float32), (0, 0), 3)[..., None]
    out[y0:y1, x0:x1] = (painted * soft + crop * (1 - soft)).astype(np.uint8)
    return out


def scene_prompt(rgb, hole):
    """What the hole's surroundings are, in words, for the fill: the colours round it read as sky, foliage, sand,
    rock or water (the generic prompt invents things -- a figure, a dress -- in a tall hole)."""
    import cv2
    ring = cv2.dilate(hole.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool) & ~hole
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[ring]
    h, sat, v = hsv[:, 0].astype(float), hsv[:, 1].astype(float), hsv[:, 2].astype(float)
    words = []
    for name, m in (("sandy beach", (h >= 8) & (h <= 28) & (sat > 50) & (v > 120)),
                    ("green bushes and trees", (h > 28) & (h <= 85) & (sat > 50)),
                    ("blue sky", (h > 85) & (h <= 130) & (v > 150) & (sat > 60)),
                    ("sea water", (h > 85) & (h <= 110) & (v <= 150)),
                    ("sandstone rocks", (h >= 5) & (h <= 30) & (sat <= 50) & (v > 120))):
        if m.mean() > 0.08:
            words.append(name)
    return ", ".join(words or ["the background"]) + ", empty, natural daylight, photograph"


def layers(photo, folder):
    """Stage 1: the photo's layers into folder (made again only when the photo changes). Needs torch."""
    import cv2
    os.makedirs(folder, exist_ok=True)
    stamp = os.path.join(folder, "source.txt")
    sig = f"{os.path.abspath(photo)} {os.path.getmtime(photo)} {LAYERS_VERSION}"
    if os.path.exists(stamp) and open(stamp).read() == sig and os.path.exists(os.path.join(folder, "plate.png")):
        return folder
    if os.environ.get("FOCALCUT_DISABLE_CUDNN"):
        import torch
        torch.backends.cudnn.enabled = False
    rgb = read_photo(photo)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    alpha, dep = subject_alpha(rgb)
    r = max(4, int(0.004 * max(rgb.shape[:2])))          # the hole: the subject and a few pixels round them (so
    hole = cv2.dilate((alpha > 0.02).astype(np.uint8),   # no halo of their colour is left in the background)
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    how = os.environ.get("FOCALCUT_SPATIAL_FILL") or ("lama" if os.path.exists(LAMA) else "mirror")
    painted = (fill(rgb, hole, scene_prompt(rgb, hole)) if how == "sd"   # (Stable Diffusion invents in a tall
               else fill_lama(rgb, hole) if how == "lama" else fill_mirror(rgb, hole))   # hole; LaMa textures it)
    under = cv2.dilate((alpha > 0.02).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    under = cv2.GaussianBlur(under, (0, 0), 1.5)[..., None]   # the fill only where the subject was: everywhere
    plate = (painted * under + rgb * (1 - under)).astype(np.uint8)   # else the photo's own background, as shot
    dfill = cv2.inpaint((dep * 255).astype(np.uint8), hole.astype(np.uint8) * 255, 15,
                        cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    dep_bg = np.where(hole, dfill, dep)                   # (behind the subject: the background's depth carried in)
    cv2.imwrite(os.path.join(folder, "photo.png"), bgr)
    cv2.imwrite(os.path.join(folder, "alpha.png"), (alpha * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(folder, "fg.png"), cv2.cvtColor(foreground(rgb, alpha), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(folder, "plate.png"), cv2.cvtColor(plate, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(folder, "depth.png"), (np.clip(dep_bg, 0, 1) * 65535).astype(np.uint16))
    fd = float(np.median(dep[alpha > 0.5])) if (alpha > 0.5).any() else 0.7
    with open(os.path.join(folder, "subject.txt"), "w") as f:
        ys, xs = np.nonzero(alpha > 0.5)
        f.write(f"{fd:.4f} {xs.min() / rgb.shape[1]:.4f} {ys.min() / rgb.shape[0]:.4f} "
                f"{xs.max() / rgb.shape[1]:.4f} {ys.max() / rgb.shape[0]:.4f}\n")
    with open(stamp, "w") as f:
        f.write(sig)
    return folder


def frames(folder, W, H, n, push=PUSH, sway=SWAY, pop=POP, direction=1.0, fade=0.0, fps=50):
    """Stage 2: n frames (BGR, W x H) of the camera's move. The output window is the photo cropped to W:H round the
    subject, as large as fits; the camera pushes in by `push` and drifts across by `sway` (direction: +1 to the
    right, -1 to the left)."""
    import cv2
    img = cv2.imread(os.path.join(folder, "photo.png"))
    plate = cv2.imread(os.path.join(folder, "plate.png"))
    alpha = cv2.imread(os.path.join(folder, "alpha.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    dep = cv2.imread(os.path.join(folder, "depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    fd, sx0, sy0, sx1, sy1 = map(float, open(os.path.join(folder, "subject.txt")).read().split())
    fgp = os.path.join(folder, "fg.png")                  # the subject's own colours in its soft edge (layers:
    if os.path.exists(fgp):                               # pymatting) -- else what stood behind them ghosts along
        img = cv2.imread(fgp)
    ih, iw = img.shape[:2]
    a = W / H                                             # the window: W:H, as big as the photo allows (with room
    cw = min(iw, ih * a) / (1 + push + sway)              # to drift), centred on the subject -- and no wider than
    cw = min(cw, max((sx1 - sx0) * iw * 1.18, cw * 0.6))  # the group with a margin (no one else at the edges)
    ch_ = cw / a
    cx = np.clip((sx0 + sx1) / 2 * iw, cw / 2 * (1 + sway), iw - cw / 2 * (1 + sway))
    cy = (sy0 + sy1) / 2 * ih - 0.05 * ch_
    cy = min(cy, sy0 * ih - 0.10 * ch_ + ch_ / 2)         # (people taller than the frame: their heads kept in, with a
    cy = float(np.clip(cy, ch_ / 2, ih - ch_ / 2))       # little room above -- the legs go, never the faces)
    gx, gy = np.meshgrid(np.linspace(-0.5, 0.5, W, dtype=np.float32), np.linspace(-0.5, 0.5, H, dtype=np.float32))
    base_x = (cx + gx * cw).astype(np.float32)            # where each output pixel looks in the photo at rest
    base_y = (cy + gy * ch_).astype(np.float32)
    dsmall = cv2.resize(dep, (iw // 4, ih // 4), interpolation=cv2.INTER_AREA)
    d_at = cv2.remap(dsmall, base_x / 4, base_y / 4, cv2.INTER_LINEAR)   # the background's depth under each pixel
    span = max(float(np.percentile(dep, 98) - np.percentile(dep, 2)), 1e-3)
    rel = (d_at - fd) / span                              # nearer than the subject > 0, farther < 0
    for k in range(n):
        p = _ease(k / max(n - 1, 1))
        z_bg = 1 + push * p * (0.55 + 0.45 * np.clip(d_at, 0, 1))   # a dolly: what is near grows more
        shift = direction * sway * cw * (p - 0.5) * rel   # and slides more against the subject
        lift = RISE * ch_ * np.sin(np.pi * p) * rel       # (a slight arc up and back)
        mx = (cx + (base_x - cx) / z_bg - shift).astype(np.float32)
        my = (cy + (base_y - cy) / z_bg + lift).astype(np.float32)
        bg = cv2.remap(plate, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        zf = 1 + push * p * (0.55 + 0.45 * fd) * pop      # the subject: as the background at its own depth (pop 1)
        sx, sy = cx, (sy0 + sy1) / 2 * ih
        fx = (sx + (base_x - sx) / zf).astype(np.float32)
        fy = (sy + (base_y - sy) / zf).astype(np.float32)
        fg = cv2.remap(img, fx, fy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        al = cv2.remap(alpha, fx, fy, cv2.INTER_LINEAR, borderValue=0)[..., None]
        f = fg * al + bg * (1 - al)
        left = (n - 1 - k) / fps                          # (the last `fade` seconds down to black)
        if fade > 0 and left < fade:
            f = f * _ease(left / fade)
        yield f.astype(np.uint8)


def render(folder, out, W, H, seconds, fps=50, encoder=None, **kw):
    """Stage 2 to a video file (no audio)."""
    n = int(round(seconds * fps))
    enc = encoder or ["-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p"]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                          "-r", str(fps), "-i", "-",           # (the photo's RGB into BT.709 video exactly as it
                          "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",   # exported:
                          *enc, out], stdin=subprocess.PIPE)   # matrix is BT.601, which shifts the colours)
    for f in frames(folder, W, H, n, fps=fps, **kw):
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"spatial: ffmpeg failed writing {out}")
    return out
