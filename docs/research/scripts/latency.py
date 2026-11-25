import json, sys, subprocess, tempfile, os, numpy as np, librosa
D="/mnt/j/Videos/References/edit-research"
res={}
for slug in sys.argv[1:]:
    A=json.load(open(f"out/{slug}.a.json")); beats=np.array(A['music']['beats'])
    v=f"{D}/{slug}.mp4" if slug!="malabar-2025-reference" else "malabar-2025-reference.mp4"
    with tempfile.TemporaryDirectory() as td:
        w=os.path.join(td,"a.wav"); subprocess.run(["ffmpeg","-v","error","-y","-i",v,"-vn","-ac","1","-ar","22050",w],check=True)
        y,sr=librosa.load(w,sr=22050)
    hop=64
    env=librosa.onset.onset_strength(y=y,sr=sr,hop_length=hop,n_fft=1024)
    pk=librosa.onset.onset_detect(onset_envelope=env,sr=sr,hop_length=hop,units="frames",backtrack=False)
    st=pk[env[pk]>=np.percentile(env[pk],60)]
    rms=librosa.feature.rms(y=y,frame_length=256,hop_length=hop)[0]
    bk=librosa.onset.onset_backtrack(st,rms)           # start of the energy rise
    tp,tb=librosa.frames_to_time(st,sr=sr,hop_length=hop),librosa.frames_to_time(bk,sr=sr,hop_length=hop)
    def med(t):
        d=np.array([x-beats[np.argmin(np.abs(beats-x))] for x in t]); d=d[np.abs(d)<0.12]; return float(np.median(d)), len(d)
    a,b=med(tp),med(tb)
    res[slug]=dict(onset_peak_minus_beat_s=round(a[0],3), attack_start_minus_beat_s=round(b[0],3), n=a[1])
    print(slug, res[slug], flush=True)
json.dump(res,open("out/latency.json","w"),indent=1)
