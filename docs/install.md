# Install

focalcut needs Python 3.10 or later, with `ffmpeg` and `ffprobe` on `PATH`.

## The GPU-free pipeline

```bash
pip install "git+https://github.com/isaacrowntree/focalcut"
focalcut --help
```

This installs numpy and OpenTimelineIO and nothing else. It covers contact sheets, the vision-judge library
pipeline and timeline export.

## Recap reels and effects

```bash
pip install "focalcut[reel] @ git+https://github.com/isaacrowntree/focalcut"
```

The `reel` extra pulls torch, transformers and the models' dependencies (SAM 3, BiRefNet, Depth Anything V2, SigLIP 2).
The subject pass and the SAM 3 effects want a CUDA GPU; two scoring workers fit a 12 GB card. Set
`FOCALCUT_DISABLE_CUDNN=1` if your torch build has broken cuDNN engines. [Hardware](hardware.md) covers what
decodes, encodes and renders on which chip.

## The SAM 3 engines

`focalcut cut` uses SAM 3 (`facebook/sam3`), a **gated** model. Accept its licence on Hugging Face and run
`hf auth login` first.

```bash
pip install "focalcut[sam] @ git+https://github.com/isaacrowntree/focalcut"
```

SAM 3's weights are under Meta's own SAM Licence, separate from focalcut's MIT licence.

## The Claude Code skill

`skills/focalcut` is a Claude Code skill that lists each editing concept with the code behind it. It isn't part
of the pip package. To install it from a clone:

```bash
mkdir -p ~/.claude/skills && ln -s "$PWD/skills/focalcut" ~/.claude/skills/focalcut
```

## From source

```bash
git clone https://github.com/isaacrowntree/focalcut && cd focalcut
python -m venv .venv && .venv/bin/pip install -e ".[reel,dev]"
.venv/bin/python -m pytest -q && .venv/bin/ruff check .
```
