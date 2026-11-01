# Hardware: what decodes, encodes and renders where

focalcut renders with whatever the machine has. Plain ffmpeg on the CPU always works; everything below is optional
and found at run time (`focalcut/reel/hw.py`).

## The three chips

| Chip | Job in a render |
|---|---|
| CPU | crop, scale and stabilise every shot (filters); decodes the shots the GPUs can't |
| Intel iGPU (media engine) | decodes camera files the other paths can't, with Quick Sync (via a Windows ffmpeg on WSL) |
| NVIDIA GPU | encodes every shot (NVENC); SAM 3, BiRefNet and the other models; the MilkDrop overlay |

Plain shots render several at once (`render.DECODERS`, or `FOCALCUT_DECODERS="qsv=5,cpu=2"`); effect shots, which
run their own GPU models, go one at a time beside them.

## Sony XAVC HS (10-bit 4:2:2 HEVC) and hardware decode

Sony's 4K 10-bit 4:2:2 HEVC is the hard case:

- **NVIDIA NVDEC has no 4:2:2 HEVC before Blackwell (RTX 50).** On an RTX 4070 Ti, `-hwaccel cuda` fails with
  "Hardware is lacking required capabilities" and ffmpeg falls back to the CPU.
- **Intel's media engine has it** (Core Ultra 200S "Arrow Lake" here; Intel iGPUs since Ice Lake). Through D3D12 the
  Intel driver reports `D3D12_VIDEO_DECODE_PROFILE_HEVC_MAIN10_422` at 4K50 into Y210 as supported.
- **Mesa's D3D12 VA-API bridge in WSL doesn't map it** (checked against Mesa main, November 2026: the D3D12 decoder
  maps only HEVC Main and Main10; its 4:2:2 profiles are for encode). No package or setting changes that; it would
  take a Mesa patch.
- **ffmpeg's generic D3D11/D3D12 decode doesn't either:** it falls back to the CPU without an error. Only Intel's
  own `hevc_qsv` decoder works.

So on WSL, focalcut decodes those files with a **Windows ffmpeg build using Quick Sync**, frames copied back to
memory for the usual CPU filters, and encodes with NVENC:

| 10 s of 4K50 XAVC HS → 1080p | CPU time |
|---|---|
| CPU decode + scale | 41 s |
| Quick Sync decode, copy back, CPU scale (what focalcut does: the crop needs full resolution) | 6.4–7.5 s |
| Quick Sync decode + GPU scale (`vpp_qsv`) | 1.8 s |

Decoded frames are pixel-identical to the CPU decode. **`-async_depth 1` is required:** at the default depth
`hevc_qsv` labels each frame three frames late, so `-ss` lands three frames early.

With 3 Intel + 3 CPU shots at once on a 24-thread Core Ultra 9 285K, the plain shots of a 140-shot landscape reel
took about 4.5 minutes (one at a time on the CPU with libx264: about 30), with the CPU at about 85% and the iGPU's
decoder at about 40%; hence the default of 5 + 2.

### Setting it up (WSL)

1. Download a Windows ffmpeg build (for example BtbN's `win64-gpl` zip) and unpack it under `C:\tools`, so that
   `C:\tools\ffmpeg-*\bin\ffmpeg.exe` exists, or point `FOCALCUT_WIN_FFMPEG` at it. `FOCALCUT_WIN_FFMPEG=""`
   turns it off.
2. Only files on a Windows drive (`/mnt/<letter>/`) go through it; stabilised shots stay on the Linux side (their
   vidstab motion file lives there).
3. Encoding uses NVENC if the GPU has it (constant quality 16, SPS/PPS in every keyframe so shots from either ffmpeg
   join by stream copy). `FOCALCUT_ENCODER=x264` forces libx264. The encoder is part of each shot's cache key.

### VA-API in WSL (for the formats it does support)

WSL has no `/dev/dri` until the `vgem` module is loaded, and Mesa's VA-API driver also needs `GALLIUM_DRIVER=d3d12`
(without it: "resource allocation failed"):

```bash
sudo modprobe vgem                                         # persist: echo vgem | sudo tee /etc/modules-load.d/vgem.conf
sudo chmod 666 /dev/dri/card0 /dev/dri/renderD128          # persist: a udev rule, SUBSYSTEM=="drm", MODE="0666"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib LIBVA_DRIVER_NAME=d3d12 GALLIUM_DRIVER=d3d12
MESA_D3D12_DEFAULT_ADAPTER_NAME=Intel vainfo --display drm --device /dev/dri/card0
```

That gives H.264, HEVC Main/Main10, VP9 and AV1 decode and encode on either GPU, but not 4:2:2.

## Other notes

- **Scoring decodes every 4K frame on the CPU** (about 1.6x real time); it skips non-reference frames and
  deblocking, about 20% faster and fine for analysis. reel.json `proxies` can point it at 1080p H.264 copies that
  NVDEC decodes (see the recipe).
- **Two scoring workers on a 12 GB GPU.** More overflows VRAM; on WSL that pages to system RAM and everything slows
  to a crawl.
- **Cap SAM 3 in crowds:** the masks module tracks at most 6 confident people at 960 px.
- **Headless WebGL on WSL** (the MilkDrop overlay): Chromium needs `GALLIUM_DRIVER=d3d12`,
  `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`, `LD_LIBRARY_PATH=/usr/lib/wsl/lib` and `--use-angle=gl`, or it falls
  back to software.
- Set `FOCALCUT_DISABLE_CUDNN=1` if your torch build has broken cuDNN engines.
- **Another Mac or PC** could take shots too (ffmpeg with VideoToolbox, over SSH and a shared folder), but it only
  speeds up plain shots; measure where a render's time goes first.
