# Rendering: speed and quality

What a reel's render does, where its time goes, and what was measured to take a 90 s reel from 22 minutes to 4 at
1080p and to make 4K a 6 to 7 minute job. The numbers are from one machine (WSL2, 24-core Core Ultra 9 285K,
41 GB given to WSL, RTX 4070 Ti 12 GB, Intel iGPU through a Windows ffmpeg) and one reel: Electric Nation's 90 s
cut (85 shots landscape, 82 portrait; 4,462 frames at 50 fps before an 8.4 s stinger; the MilkDrop overlay on; one
SAM 3 effect shot; nine transitions; 53 landscape and 78 portrait shots stabilised), from 4K 10-bit 4:2:2 HEVC
camera files on a Windows drive, every cache cold.

## Rendering in 4K

```json
{"resolution": "4k"}
```

in reel.json, at the top level, per format (`{"landscape": "4k", "portrait": "1080p"}`) or inside a cut
(`"cuts": {"short": {"resolution": "4k"}}`). `"1080p"` is the default; `"1440p"`, `"<n>p"` and `[width, height]`
work too. Then render as usual (`--steps render,xml,review`). Shots, effects, transitions, the overlay and the
stinger are all made at that size, the XML's sequence is that size and its V1 media are the 4K shot files. Shots
are cached per size, so going back to 1080p re-renders nothing that was already there.

What 4K gives:

- **Landscape** is real 4K: an uncropped shot is the camera's own 3840 x 2160 pixels, and a reframed one (most
  crops are 0.85 to 1.0 of the frame) is scaled up by at most 18%.
- **Portrait** is not. A 9:16 strip of a 16:9 frame is 1,215 px wide, so a 2160 x 3840 reel is that strip scaled
  up 1.8x. At pixel level a Lanczos upscale, the same with an unsharp mask, and the 1080 x 1920 render enlarged
  are the same picture; the sharpened one only has crisper noise. A sharpen or AI-upscale step is not warranted:
  there is no detail to recover from night footage at that size, and Instagram plays 1080 wide anyway.
  `{"landscape": "4k", "portrait": "1080p"}` is the honest pairing; 4K portrait is there if a platform wants the
  file.
- **Effects at 4K** keep the look of the 1080p reel rather than gaining detail. MilkDrop's presets draw their
  lines, glows and feedback blur in pixels, so the layer keeps a 1080p canvas (overlay `"milkdrop_side"`: 1920)
  and is scaled up bicubically; the built-in overlay effects (confetti, sparks, ...) likewise. SAM 3 tracks at
  960 px and BiRefNet mattes at 1024 px whatever the frame, so a 4K effect shot's masks are made at 1080p size
  and scaled as they are used. Transitions run on the full 4K frames.
- **File size**: constant quality 16 gives about 115 Mb/s at 4K50 on this footage (1.4 GB for the 98 s reel) and
  38 Mb/s at 1080p (470 MB).

## Before and after

Wall-clock seconds per stage. The stages marked *beside* run while the shots do, so a total is less than its
column's sum.

| stage | 1080p landscape: before | after | 1080p portrait: before | after | 4K landscape | 4K portrait |
|---|---|---|---|---|---|---|
| shots (decode, stabilise, crop, scale, encode) | 242 | 200 | 235 | 186 | 242 | 211 |
| effect shot (SAM 3 + BiRefNet), *beside* | 139 | 150 | 60 | 131 | 283 | 199 |
| MilkDrop layer | 49, after the shots | 77, *beside* | 52, after | 58, *beside* | 89, *beside* | 58, *beside* |
| transitions | 21 | 7 | 29 | 9 | 24 | 33 |
| overlay composite and final encode | 1,024 | 37 | 1,207 | 37 | 103 | 98 |
| stinger, mix, loudness, mux | 6 | 7 | 6 | 7 | 10 | 10 |
| **total** | **1,344 (22.4 min)** | **251 (4.2 min)** | **1,541 (25.7 min)** | **240 (4.0 min)** | **422 (7.0 min)** | **354 (5.9 min)** |

(The portrait "before" shared the machine with other work for part of its overlay pass; the same pass in the
project's own logs runs at 3.9 frames a second, about 1,140 s. The effect shot is slower "after" because it now
shares every core with the stabilisation analysis; on its own it takes about 120 s at either size.)

A re-render after a small plan change (one stabilised shot's in-point moved 0.1 s) takes 79 to 93 s: that one
shot is rendered again; the other shots, the transitions, the MilkDrop layer and the stinger come from the cache;
and the overlay pass runs again (37 s), as it must after any change.

With `"stab_analysis": "half"` (finding 3) the cold 1080p landscape render takes 207 s.

## The review

1. **The overlay pass was one Python thread.** `overlay.frame` composites in numpy at full size: 3.9 to 4.4
   frames a second at 1080p, 17 to 20 minutes for the reel, with the GPU idle (8% average over the render) and 23
   of 24 cores idle. It was 76% of the render, and would have been over an hour at 4K. *Fix:* `composite.py` does
   the same arithmetic on the GPU, arranged so a frame costs few passes: the footage stays in the 4:2:0 it is
   decoded to; nearly all of the show is `footage * a + b` per pixel and is applied to the planes directly; `a`
   and `b` are built at the MilkDrop layer's size and scaled up once; only the frames that fold or recolour the
   footage itself (RGB split, kaleidoscope, duotone, light bursts) go through RGB. Decode and encode run on
   threads of their own and the frames travel in recycled page-locked buffers. The composite alone does 170
   frames a second at 1080p and 70 at 4K; the whole pass runs at 120 at 1080p and 45 at 4K, where the frames that
   do go through RGB (a drop's bursts: 780 of the reel's 4,462, 43 ms each) take half of its time. *Gain:*
   1,024 s to 37 s. `tests/test_composite.py` holds the GPU and numpy paths to the same picture (mean difference
   half a code value: numpy truncates).
2. **Frames were converted and copied more often than needed.** Every overlay frame went YUV to RGB to YUV
   through swscale (a chroma resample each way; highlights above 235 and colours outside the RGB gamut clipped),
   and the joined cut was written to disk twice more (`shots.mp4`, `video.mp4`) only to be read back. *Fix:* the
   overlay reads the shots as one (ffmpeg's concat list) and the final mux joins them straight into the delivered
   file; where the show does nothing to a pixel (the clear centre of the frame, off the beat's zoom) it now
   passes through bit for bit. One trap on the way: raw Y'CbCr piped to ffmpeg without colour tags is taken for
   BT.601 and converted to the BT.709 the output asks for (Y 126, Cb 100, Cr 150 came out 123, 102, 148), so raw
   YUV is declared BT.709 as an input (`hw.RAW_YUV`), with a test. The old RGB path was right (red 255 encodes to
   Y 63, Cb 102, Cr 240, as BT.709 says).
3. **Stabilisation is now the largest cost, and it is all CPU.** 53 of 85 landscape shots and 78 of 82 portrait
   shots are stabilised. vidstab's pass 1 costs about half a core-second per 4K frame (a 2 s stretch: 9.4 s wall
   and 65 core-seconds with its OpenMP on every core, 42 core-seconds on one thread) and its pass 2 0.18 s per
   frame on one thread: about 0.8 core-seconds per stabilised frame, 2,100 for the landscape reel and 3,400 for
   the portrait one, before anything is encoded. The old render ran both passes inside its seven shot slots.
   *Fix:* pass 1 runs on its own (it needs no encoder session), one thread each, one per core at 1080p (24 here:
   2,400 frames in 52 s, against 72 s for 12 x 2 threads) and as many as the memory allows above that, at lower
   CPU priority than the effect shot and the encodes; each shot goes to a render slot as soon as its analysis is
   done. Motion files are cached per source stretch (`~/.cache/focalcut/stab`), so only new shots pay again.
   *What is left:* this is the floor of the shots stage. Three ways under it, none taken by default because each
   changes the picture:
   - `"stab_analysis": "half"` measures the shake on a half-size picture and scales the motion file: 7x less CPU
     for pass 1 (3.9 s against 31.7 s for a 2 s stretch on two threads). The stabilised picture then sits 0.4 to
     6 pixels (of 4K) from the full-size analysis's and moves against it by 0.2 to 2.4 px from frame to frame: a
     different result on the four stretches tried, not a visibly worse one, but not the same one.
   - The Windows ffmpeg's newer libvidstab is 3 to 5x faster at both passes, but frames the same shot about 15%
     tighter (it reads much larger motions and zooms in to hide them): not a drop-in.
   - Stabilising fewer shots. Measured on 16 stabilised landscape shots (the background's frame-to-frame motion
     by tracked features, then the size of its change per frame), vidstab reduced the jitter on 8, raised it on 5
     and left 3 the same (mean 2.08 px before, 2.16 after). On crowd and dance footage the dancers are much of
     what it tracks. Which shots to stabilise is the plan's call, not the render's.
4. **Effect shots.** One SAM 3 effect shot takes about two minutes (loading the models and the video tracker
   48 s, BiRefNet's matte per frame 31 s, the morphology around it 10 s, imports 17 s) and already ran beside
   the plain shots; it is now the other bound on the shots stage. At 4K it took 2.6x longer and 17 GB (masks at
   33 MB a frame). *Fix:* masks are made at 1080p size and scaled up as each is used (`fx.base.Scaled`), and
   spotlight's arithmetic is gathered into single-channel weights (334 ms to 104 ms a 4K frame, the same picture
   to one code value on three pixels in a million): a 4K effect shot now costs what a 1080p one does when it has
   the machine to itself. *Left:* batching BiRefNet and keeping the models warm between renders would take
   perhaps 30 s off.
5. **Transitions** read both shots whole into memory as RGB (a 4 s shot at 4K is 5 GB), blended a few frames and
   re-encoded both, one transition after another. *Fix:* only the frames inside the blend are read; the rest
   streams through in the shot's own 4:2:0, so its untouched frames are re-encoded from exactly what was decoded;
   a transition's two shots are written at once, and transitions that share no shot run side by side. 21 s to
   7 s. (A trimmed read through ffmpeg repeats its first frame to fill the time before it unless told
   `-fps_mode passthrough`; the reads say so and check their counts.)
6. **Parallelism and the encoder's limits.** The driver allows 12 NVENC sessions at once across the machine
   (measured: 16 started beside one already open, 11 succeeded). The defaults are now 6 Quick Sync + 3 CPU shots
   (was 5 + 2), an effect shot and the MilkDrop layer: 11. An encode that finds no session free waits and tries
   again instead of failing the reel (two renders at once). A 4K H.264 encode runs at about 75 frames a second
   on this card whatever the preset from p5 up. The MilkDrop layer (about 100 frames a second at 1080p) renders
   beside the shots: it needs the music and the plan's acts, not the pictures.
7. **Memory is the limit at 4K.** A stabilised 4K render holds 1.6 GB, an analysis 0.6 GB, a 4K effect shot over
   10 GB; nine renders, 24 analyses and the effect shot together passed 41 GB and swapped. *Fix:* above 1080p the
   analyses' width follows the machine's memory (16 here), and a new analysis or stabilised render waits while
   under 4 GB is free.
8. **Intermediates and generations.** Shots are H.264 at constant quality 16; on this footage that already runs
   into the level's bitrate ceiling (42 Mb/s at 1080p50, 130 Mb/s at 4K50; a lower `-cq` changes nothing without
   a `-maxrate`), at 45.7 dB PSNR against the scaled source. Each further generation costs 0.6 dB (a transition
   shot is a second, the overlay pass a second or third): not worth lossless intermediates, which would be ten
   times the size. The MilkDrop layer, which is keyed and scaled after decoding, is the exception: it is now kept
   4:4:4 at constant quality 10.
9. **Caching** works as intended: shots are keyed by everything that shapes them (now also the frame size and
   the encode's settings), transitions by their pair of files, the MilkDrop layer by the song, the frame count
   and the preset schedule. The re-render measured above re-made one shot.
10. **I/O is not a bottleneck.** The Windows drive reads at 280 to 420 MB/s over 9P; the reel reads about 2.5 GB
    of camera files. Proxies would not help the render (the crop needs the full-size picture); they help the
    analysis steps.

## Quality review

Looked at frame by frame in the old and new renders.

- **A flash or dip to white under the overlay** was a white oval inside a black border: the overlay darkened
  the frame's edges after the cut had gone to white. The show now follows the cut's own whiteness
  (`overlay.whites`), so the frame is white edge to edge.
- **Zoom transition**: five copies of the frame 2% apart showed as ghost outlines (fingers and a logo repeated).
  It now averages copies about 2.5 px apart at the frame's edge (30 at 1080p, 48 at 4K); the spin transition
  likewise.
- **Banding**: the overlay's fade to black and MilkDrop's fields are synthetic gradients written in 8 bits. The
  compositor dithers before rounding, and the delivered encode uses spatial AQ (more of the same bitrate for
  dark, flat areas: +0.25 dB XPSNR at the same size) and a keyframe every second (it was every five, which seeks
  badly in players and editors). 10-bit delivery (HEVC) was considered and left: the footage's own noise dithers
  everything except those synthetic layers, and 8-bit H.264 is what every platform and Premiere take.
- **The beat's zoom punch** resampled every frame bilinearly; it is bicubic now, and frames whose zoom is under a
  quarter of a pixel are left alone.
- **Blended frames were truncated** to 8 bits (half a code value darker than their neighbours); they are rounded.
- **The MilkDrop layer** lost its colour detail to 4:2:0 before it was keyed; it is kept 4:4:4.

Measured, new against old at 1080p portrait (the frame's clear centre, where the overlay leaves the footage
alone): PSNR 41.1 dB luma, 40.1 and 47.0 dB chroma over the whole reel (median 42.0 dB luma, 50 dB chroma on the
stabilised shots), about what two encodes of the same picture give; every cut on the same frame (4,462 + 419
frames, checked shot by shot against the shot files); BT.709 tags and limited range on every output; -14.0 LUFS.
17 frames differ by more (23 to 29 dB): blue stage light outside the RGB gamut (Cb above 200), which the old RGB
round trip clipped and the new path keeps as the camera recorded it. The 4K renders scaled down match the 1080p
ones to 43.5 dB (landscape) and 44.6 dB (portrait), with no frame under 30 dB.

Not changed, for the editor to decide:

- **Highlights above 100%.** The camera files carry luma up to 255 (super-whites) in limited-range BT.709. Plain
  shots keep them; the overlay pass holds its output to 16-235 as it always did, and players clip there anyway.
  A grade would roll those highlights off into range instead of clipping them; that is a look, and the footage
  is delivered ungraded.
- **The stinger** is 24 fps, conformed to 50 by repeating frames (an uneven 2-2-2-3 cadence); a 25 or 50 fps
  export of it would move evenly.
- **Stabilisation's benefit** (finding 3).

## What still dominates, and what would move it

At 1080p: stabilisation's CPU (the shots stage is bound by it) and the SAM 3 effect shot, about three minutes
side by side, then 37 s of overlay. At 4K the same, with the effect shot slowed by the 4K renders beside it, plus
the overlay pass (100 s) and memory.

- `"stab_analysis": "half"`, or stabilising only what needs it: the shots stage falls to the effect shot's two
  minutes; with no effect shot, to about one.
- More memory for WSL (`.wslconfig`): a 4K render runs 16 analyses at once here for lack of it.
- The overlay's RGB route (finding 1) could be trimmed further: worth 20 to 30 s of a 4K render, a few at 1080p.
- A second render at the same time (the other format) is not a gain on this machine: both want every core for
  stabilisation and share 12 encoder sessions.
- An RTX 50 card decodes the camera's 4:2:2 HEVC itself, which would move decode, crop and scale onto one GPU
  (`scale_cuda`) for unstabilised shots; with most shots stabilised on the CPU it would change little here.
