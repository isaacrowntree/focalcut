# Security Policy

## Reporting a vulnerability

Please report security issues **privately** through GitHub Security Advisories:
[Report a vulnerability](https://github.com/isaacrowntree/focalcut/security/advisories/new).

Don't open a public issue for anything security-sensitive. You can expect a first reply within a week.

## Supported versions

Only the latest commit on `main` gets security fixes. There is no support commitment before a `1.0` release.

## Threat model

focalcut runs on your own machine, over footage you shot.

- **Media is decoded by ffmpeg.** Every video, image and audio file goes through `ffmpeg` / `ffprobe` from your
  `PATH`, so keep that up to date. Treat footage from elsewhere as untrusted input to ffmpeg.
- **The review server is localhost only.** `focalcut reel serve` answers only requests whose `Host` header names
  this machine (`localhost`, `127.0.0.1`, `[::1]`). A page on another site that rebinds its DNS to 127.0.0.1 still
  sends its own host name and is refused. It serves only files inside each project's own folder, and writes only
  that project's `picks.json`. Don't expose its port to a network.
- **Model weights are downloaded from Hugging Face.** The `reel` and `sam` extras fetch SAM 3, BiRefNet, Depth
  Anything V2 and SigLIP 2 on first use. BiRefNet loads with `trust_remote_code=True`, which runs Python from its
  model repository (`ZhengPeng7/BiRefNet`). Pin a revision or review that code if your threat model needs it.
- **reel.json and picks.json are trusted.** They name files and ffmpeg filters (`rotate`) that are passed to
  ffmpeg. Don't run a reel from a project folder you didn't make.
