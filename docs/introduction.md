# focalcut

**Cut dance and event footage by who the camera is following: per-subject clips, beat-synced recap reels and
editor timelines.**

![The Electric Nation 2026 landscape reel, shot by shot along the song: 151 shots in six acts (teaser, arrival,
shows, night, climax, finale), most of them cut from the editor's picks, above the music's energy per
beat.](img/reel-timeline.jpg)

focalcut does two jobs.

- **Split a library by subject.** Point it at a folder of dance footage, or any video where the camera moves from
  one person to the next. It finds where the focus changes from one dancer or couple to another and exports
  per-subject clips plus a timeline for Premiere, DaVinci Resolve, Final Cut or Avid. This path needs no GPU: it
  uses ffmpeg contact sheets read by a vision model. See [Cutting a library by subject](library.md).
- **Cut recap reels.** From a curated folder it builds a landscape and a portrait reel on the music's beats, told in
  acts. The reels have velocity edits, motivated transitions, portrait reframing that follows a dancer and favours
  faces, shake checks, effects and an optional MilkDrop light show. Each comes with a Premiere XML and a review page
  where the editor picks, bans and reframes shots. The editor's picks always win. See [Recap reels](recipes/recap-reel.md)
  and [The editor's picks](picks.md).

Every editing rule the planner follows is its own piece of code, listed in [Editing concepts](concepts.md), so a
teaser, a highlight or a promo can reuse them without the full reel.

- Source: [github.com/isaacrowntree/focalcut](https://github.com/isaacrowntree/focalcut) (MIT)
- Write-up: [focalcut on zackdesign.biz](https://zackdesign.biz/focalcut/)
