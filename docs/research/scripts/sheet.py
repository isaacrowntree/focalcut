import json, sys, subprocess, numpy as np, cv2
slug, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
D="/mnt/j/Videos/References/edit-research"; v=f"{D}/{slug}.mp4" if not slug.startswith("malabar") else "malabar-2025-reference.mp4"
f=[x for x in json.load(open("out/references.json"))["films"] if x["slug"]==slug][0]
W,H,N=200,112,5; rows=[]
for i,s in enumerate(f["shots"][a:b],a):
    row=[]
    for q in np.linspace(0.08,0.92,N):
        t=s["t0"]+q*s["length"]
        raw=subprocess.run(["ffmpeg","-v","error","-ss",f"{t:.3f}","-i",v,"-frames:v","1","-vf",f"scale={W}:{H}","-f","rawvideo","-pix_fmt","bgr24","-"],capture_output=True).stdout
        row.append(np.frombuffer(raw,np.uint8).reshape(H,W,3).copy() if len(raw)==W*H*3 else np.zeros((H,W,3),np.uint8))
    im=np.hstack(row)
    cv2.putText(im,f"{i} {s['t0']:.1f}s len {s['length']:.1f} {s['scale']} {s['move']}{'' if s['hard_in'] else ' SOFT'}",(4,14),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1,cv2.LINE_AA)
    rows.append(im)
cv2.imwrite(f"out/sheet_{slug}_{a}.jpg",np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,70])
