"""pytubefix download: ptdl.py <id> <outdir> <slug>  -> <slug>.mp4 (<=1080p avc video + m4a audio, muxed by ffmpeg)"""
import os, subprocess, sys
from pytubefix import YouTube
vid, outdir, slug = sys.argv[1:4]
client = sys.argv[4] if len(sys.argv) > 4 else None
yt = YouTube("https://www.youtube.com/watch?v=" + vid, **({"client": client} if client else {}))
vs = [s for s in yt.streams.filter(adaptive=True, only_video=True, file_extension="mp4") if s.resolution and int(s.resolution[:-1]) <= 1080]
vs.sort(key=lambda s: (int(s.resolution[:-1]), "avc1" in (s.video_codec or ""), s.fps or 0), reverse=True)
au = yt.streams.filter(only_audio=True, file_extension="mp4").order_by("abr").desc().first()
v = vs[0]
print(yt.title, "|", v.resolution, v.video_codec, v.fps, "|", au.abr, flush=True)
vp = v.download(output_path=outdir, filename=slug + ".v.mp4")
ap = au.download(output_path=outdir, filename=slug + ".a.m4a")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vp, "-i", ap, "-c", "copy", "-movflags", "+faststart", os.path.join(outdir, slug + ".mp4")], check=True)
os.remove(vp); os.remove(ap)
