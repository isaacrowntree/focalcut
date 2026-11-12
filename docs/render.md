# Rendering: speed and quality

What a reel's render does, where its time goes, and what was measured to get it from 22 minutes to 4 (1080p) and to
make 4K a matter of minutes. The numbers are from one machine (WSL2, 24-core Core Ultra 9 285K, 41 GB given to WSL,
RTX 4070 Ti 12 GB, Intel iGPU through a Windows ffmpeg) and one reel: Electric Nation's 90 s cut (85 shots
landscape, 82 portrait, 4,462 frames at 50 fps before an 8.4 s stinger, the MilkDrop overlay on, one SAM 3 effect
shot, nine transitions, 53 landscape and 78 portrait shots stabilised), from 4K 10-bit 4:2:2 HEVC camera files on a
Windows drive, every cache cold.

## Rendering in 4K

```json
{"resolution": "4k"}
```

in reel.json, at the top level, per format (`{"landscape": "4k", "portrait": "1080p"}`) or inside a cut
(`"cuts": {"short": {"resolution": "4k"}}`). `"1080p"` is the default; `"1440p"`, `"<n>p"` and `[width, height]`
work too. Then render as usual (`--steps render,xml,review`). Shots, effects, transitions, the overlay and the
stinger are all made at that size, the XML's sequence is that size and its V1 media are the 4K shot files; shots
are cached per size, so going back to 1080p re-renders nothing that was already there.

What 4K gives, honestly:

- **Landscape** is real 4K: an uncropped shot is the camera's own 3840 x 2160 pixels, and a reframed one (most
  crops are 0.85 to 1.0 of the frame) is scaled up by at most 18%.
- **Portrait** is not. A 9:16 strip of a 16:9 frame is 1,215 px wide, so a 2160 x 3840 reel is that strip scaled
  up 1.8x. Side by side at pixel level a Lanczos upscale, the same with an unsharp mask, and the 1080 x 1920
  render enlarged are the same picture; the sharpened one only has crisper noise. A sharpen or AI-upscale step is
  not warranted: there is no detail to recover from night footage at that size, and Instagram plays 1080 wide
  anyway. `{"landscape": "4k", "portrait": "1080p"}` is the honest pairing; 4K portrait is there if a platform
  wants the file.
- **Effects at 4K** keep the look of the 1080p reel rather than gaining detail: MilkDrop's presets draw lines,
  glows and feedback blur in pixels, so the layer keeps a 1080p canvas (overlay `"milkdrop_side"`: 1920) and is
  scaled up bicubically; the built-in overlay effects (confetti, sparks, ...) likewise; SAM 3 tracks at 960 px and
  BiRefNet mattes at 1024 px whatever the frame, so a 4K effect shot's masks are made at 1080p size and scaled as
  used. Transitions run on the full 4K frames.
- **File size**: constant quality 16 gives about 115 Mb/s at 4K50 on this footage (1.4 GB for the 98 s reel),
  38 Mb/s at 1080p (470 MB).

## Before and after

Wall-clock seconds per stage; stages marked *beside* overlap the shots, so the total is not the sum.

| stage | 1080p landscape, before | after | 1080p portrait, before | after | 4K landscape | 4K portrait |
|---|---|---|---|---|---|---|
| shots (decode, stabilise, crop, scale, encode) | 242 | «L_SHOTS» | 235 | «P_SHOTS» | «L4_SHOTS» | «P4_SHOTS» |
| effect shot (SAM 3 + BiRefNet), *beside* | 139 | «L_FX» | 60 | «P_FX» | «L4_FX» | «P4_FX» |
| MilkDrop layer | 49 (after the shots) | «L_MD» *beside* | 52 | «P_MD» *beside* | «L4_MD» *beside* | «P4_MD» *beside* |
| transitions | 21 | «L_TR» | 29 | «P_TR» | «L4_TR» | «P4_TR» |
| overlay composite and final encode | 1,024 | «L_OV» | 1,207 | «P_OV» | «L4_OV» | «P4_OV» |
| joins, stinger, mix, loudness, mux | 6 | «L_MUX» | 6 | «P_MUX» | «L4_MUX» | «P4_MUX» |
| **total** | **1,344 (22.4 min)** | **«L_TOT»** | **1,541 (25.7 min)** | **«P_TOT»** | **«L4_TOT»** | **«P4_TOT»** |

(The portrait "before" shared the machine with other work for part of its overlay pass; the same pass in the
project's own logs runs at 3.9 frames a second, about 1,140 s.)

A re-render after a small plan change (one shot's in-point moved 0.1 s): «NUDGE» s. One shot is rendered again,
the transitions, the MilkDrop layer and the stinger come from the cache, and the overlay pass runs again
(«NUDGE_OV» s), as it must for any change.

With `"stab_analysis": "half"` (finding 3) the 1080p landscape render takes «HALF_TOT» s.

## The review

1. **The overlay pass was one Python thread.** `overlay.frame` composites in numpy at full size: 3.9 to 4.4
   frames a second at 1080p, 17 to 20 minutes for the reel, with the GPU idle (8% average over the render) and 23
   of 24 cores idle. It was 76% of the render. *Fix:* `composite.py` does the same arithmetic on the GPU, arranged
   so a frame costs few passes: the footage stays in the 4:2:0 it is decoded to, nearly all of the show is
   `footage * a + b` per pixel and is applied to the planes directly, `a` and `b` are built at the MilkDrop
   layer's size and scaled up once, and only the few frames that fold or recolour the footage (RGB split,
   kaleidoscope, duotone, light bursts) go through RGB. Decode and encode run on threads of their own. The
   composite alone does 170 frames a second at 1080p and 70 at 4K; the pass is now bounded by the encoder (about
   105 frames a second at 1080p, 45 at 4K with the composite sharing the GPU). *Gain:* 1,024 s to «L_OV» s.
   `tests/test_composite.py` holds the GPU and numpy paths to the same picture (mean difference half a code value:
   numpy truncates).
2. **Frames were converted more often than needed, and one conversion was wrong.** Every overlay frame went
   YUV to RGB to YUV through swscale (a chroma resample each way, and highlights above 235 clipped), and the
   joined cut was written twice more (`shots.mp4`, `video.mp4`) only to be read back. *Fix:* the overlay reads
   the shots as one (ffmpeg's concat list) and the final mux joins them straight into the delivered file; the
   clear centre of the frame now passes through the overlay bit for bit. Found on the way: raw Y'CbCr piped to
   ffmpeg without colour tags is taken for BT.601 and converted to the BT.709 the output asks for (Y 126, Cb 100,
   Cr 150 came out 123, 102, 148), so raw YUV is now declared BT.709 on the way in (`hw.RAW_YUV`), with a test.
   The old RGB path was right (red 255 encodes to Y 63, Cb 102, Cr 240, as BT.709 says).
3. **Stabilisation is now the largest cost, and it is all CPU.** 53 of 85 landscape shots and 78 of 82 portrait
   shots are stabilised. vidstab's pass 1 costs about half a core-second per 4K frame (9.4 s wall on all cores
   for a 2 s stretch; 65 core-seconds with its OpenMP, 42 on one thread) and its pass 2 0.18 s per frame on one
   thread: about 0.8 core-seconds per stabilised frame, 2,100 core-seconds for the landscape reel and 3,400 for
   the portrait one, before anything is encoded. The old render ran both passes inside the seven shot slots.
   *Fix:* pass 1 runs on its own (no encoder session), one thread each, as many at once as the machine's memory
   allows (0.6 GB each: 16 here), at lower CPU priority than the effect shot and the encodes, and each shot goes
   to a render slot as soon as its analysis is done. Motion files are cached per source stretch
   (`~/.cache/focalcut/stab`), so only new shots pay again. *What is left:* this is the floor of the shots stage.
   Three ways under it, none taken by default because each changes the picture:
   - `"stab_analysis": "half"` measures the shake on a half-size picture and scales the motion file: 7x less CPU
     for pass 1 (3.9 s against 31.7 s for a 2 s stretch on two threads). The stabilised picture then sits 0.4 to
     6 pixels (of 4K) from the full-size analysis's, moving against it by 0.2 to 2.4 px from frame to frame: a
     different, not a visibly worse, result on the four stretches tried, but not the same one.
   - The Windows ffmpeg's newer libvidstab is 4 to 5x faster at both passes, but frames the same shot about 15%
     tighter (`compare/stab_linux_vs_windows.jpg`): not a drop-in.
   - Stabilising fewer shots. Measured on 16 stabilised landscape shots (the background's frame-to-frame motion
     by tracked features, then the size of its change per frame), vidstab reduced the jitter on 8, raised it on 5
     and left 3 the same (mean 2.08 px before, 2.16 after). On crowd and dance footage the dancers are much of
     what it tracks. That is the plan's call, not the render's; the two clips in `compare/` show a shot it helps
     and one it hurts.
4. **Effect shots.** One SAM 3 effect shot takes about two minutes (models loading and the video tracker 48 s,
   BiRefNet's matte per frame 31 s, the morphology around it 10 s, imports 17 s) and ran beside the plain shots
   already; it is now the other bound on the shots stage. At 4K it took 2.6x longer and 17 GB (masks at 33 MB a
   frame). *Fix:* masks are made at 1080p size and scaled up as each is used (`fx.base.Scaled`), and spotlight's
   arithmetic is gathered into single-channel weights (334 ms to 104 ms a 4K frame, the same picture to one code
   value on three pixels in a million): a 4K effect shot now costs what a 1080p one does when it has the machine
   to itself. *Left:* batching BiRefNet and keeping the models warm between renders would take perhaps 30 s off.
5. **Transitions** read both shots whole into memory as RGB (a 4 s shot at 4K is 5 GB), blended a few frames and
   re-encoded both. *Fix:* only the frames inside the blend are read; the rest streams through in the shot's own
   4:2:0, so its untouched frames are re-encoded from exactly what was decoded. (A trimmed read through ffmpeg
   repeats its first frame to fill the time before it unless told `-fps_mode passthrough`; the reads now say so
   and check their counts.)
6. **Parallelism and the encoder's limits.** The driver allows 12 NVENC sessions at once across the machine
   (measured: 16 started, 11 succeeded beside one already open). Defaults are now 6 Quick Sync + 3 CPU shots
   (was 5 + 2), an effect shot and the MilkDrop layer: 11. An encode that finds no session free waits and tries
   again instead of failing the reel (two renders at once). A 4K H.264 encode runs at about 75 frames a second on
   this card whatever the preset from p5 up; the delivered file is one session, so 60 s is the floor for a 90 s 4K
   reel's final encode. The MilkDrop layer (about 100 frames a second at 1080p) now renders beside the shots.
7. **Memory is the limit at 4K.** A stabilised 4K render holds 1.6 GB, an analysis 0.6 GB, a 4K effect shot over
   10 GB; nine renders, 24 analyses and the effect shot together passed 41 GB, swapped, and the first 4K run took
   twice as long as it should have. *Fix:* the analyses' width follows the machine's memory, and a new analysis
   or stabilised render waits while under 4 GB is free.
8. **Intermediates and generations.** Shots are H.264 at constant quality 16; on this footage that already runs
   into the level's bitrate ceiling (42 Mb/s at 1080p50, 130 Mb/s at 4K50; a lower `-cq` changes nothing without
   a `-maxrate`), at 45.7 dB PSNR against the scaled source. Each further generation costs 0.6 dB (a transition
   shot is a second, the overlay pass a second or third): not worth lossless intermediates, which would be 10x the
   size on a 9P drive. The MilkDrop layer, which is keyed and scaled after decoding, is the exception: it is now
   kept 4:4:4 at constant quality 10.
9. **Caching** works as intended: shots are keyed by everything that shapes them (now also the frame size and
   the encode's settings), transitions by the pair of files, the MilkDrop layer by the song, the frame count and
   the preset schedule. The measured re-render above re-made one shot.
10. **I/O is not a bottleneck.** The Windows drive reads at 280 to 420 MB/s over 9P; the reel reads about 2.5 GB.
    Proxies would not help the render (the crop needs the full-size picture); they help the analysis steps.

## Quality review

Looked at frame by frame in the old and new renders (stills in the scratch `compare/` folder of this work):

- **Flash and dip to white under the overlay** were a white oval inside a black border: the overlay darkened
  the frame's edges after the cut had gone to white. The show now follows the cut's own whiteness
  (`overlay.whites`), so the frame is white edge to edge.
- **Zoom transition**: five copies of the frame 2% apart showed as ghost outlines (fingers and a logo repeated).
  It now averages copies about 2.5 px apart at the frame's edge (30 at 1080p, 48 at 4K); the spin transition
  likewise.
- **Banding**: the overlay's fade to black and MilkDrop's fields are synthetic gradients written in 8 bits. The
  compositor now dithers before rounding, and the delivered encode uses spatial AQ (more of the same bitrate for
  dark, flat areas: +0.25 dB XPSNR at the same size) and a keyframe every second (it was every five, which seeks
  badly in players and editors). 10-bit delivery (HEVC) was considered and left: the footage's own noise dithers
  everything except those synthetic layers, and H.264 8-bit is what every platform and Premiere take.
- **The beat's zoom punch** resampled every frame bilinearly; it is bicubic now, and frames whose zoom is under a
  quarter of a pixel are left alone.
- **Blended frames were truncated** to 8 bits (half a code value darker than their neighbours); they are rounded.
- **Colour**: BT.709 tags, limited range, checked on every output; cut points land on the same frames as before
  (4,462 + 419 frames); loudness -14.0 LUFS.

Not changed, for the editor to decide:

- **Highlights above 100%.** The camera files carry luma up to 255 (super-whites) in limited-range BT.709. Plain
  shots keep them; anything that goes through RGB (effects, transitions' blended frames, the overlay) clips at
  235, and players clip there anyway. A grade would roll those highlights off into range instead of clipping
  them; that is a look, and the footage is delivered ungraded.
- **The stinger** is 24 fps, conformed to 50 by repeating frames (a 2-2-2-3 cadence); a 25 or 50 fps export of it
  would move evenly.
- **Stabilisation's benefit** (finding 3).

## What still dominates, and what would move it

At 1080p: stabilisation's CPU (two thirds of the shots stage) and the SAM 3 effect shot, about three minutes
between them, then 45 s of overlay. At 4K: the same three minutes, plus the encoder (shots, transitions, the
final file: about 75 frames a second per session) and memory.

- `"stab_analysis": "half"`, or stabilising only what needs it: the shots stage falls to the effect shot's two
  minutes; with no effect shot, to about one.
- More memory for WSL (`.wslconfig`): 4K runs 16 analyses at once here for lack of it, 1080p could run 24.
- A second render at the same time (the other format) is not a gain on this machine: both want every core for
  stabilisation and share 12 encoder sessions.
- An RTX 50 card decodes the camera's 4:2:2 HEVC itself, which would move decode, crop and scale onto one GPU
  (`scale_cuda`) for unstabilised shots; with most shots stabilised on the CPU it would change little here.
