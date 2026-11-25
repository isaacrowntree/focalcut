import json, numpy as np
R=json.load(open('out/references.json')); F=R['films']
g=lambda d,*k: (g(d[k[0]],*k[1:]) if len(k)>1 else d.get(k[0])) if d is not None else None
f2=lambda x,n=2: '–' if x is None else f"{x:.{n}f}"
pc=lambda x: '–' if x is None else f"{x*100:.0f}%"
print("T1 shot length\n| film | kind | dur | shots | cuts/min | median s | IQR s | p10–p90 s | max s | median/mean | tempo | median beats | IQR beats | ≥2 bars | ≥4 bars | dissolves |")
for f in F:
    L=f['shot_length_s']; B=f['grid']['length_beats']
    print(f"| {f['slug']} | {f['kind']} | {f['music_end']:.0f} | {L['n']} | {f['cuts_per_minute']} | {L['median']:.2f} | {L['q25']:.2f}–{L['q75']:.2f} | {L['p10']:.2f}–{L['p90']:.2f} | {L['max']:.1f} | {L['median']/L['mean']:.2f} | {f['music']['tempo']:.0f} | {B['median']:.1f} | {B['q25']:.1f}–{B['q75']:.1f} | {pc(f['grid']['shots_2_bars_or_more'])} | {pc(f['grid']['shots_4_bars_or_more'])} | {f['cuts']['soft_dissolves']} |")
print("\nT2 sync\n| film | grid conf | locked | best share/null95/chance | lead vs tracked beat (s / frames) | lead vs attack (frames) | on beat | chance | on downbeat | best phase | chance | 4-bar start | best | chance | whole-bar lengths | strong onset/chance |")
for f in F:
    s=f['sync']; l=s['beat_lock']
    print(f"| {f['slug']} | {f['music']['grid_confidence']} | {'yes' if l['locked'] else 'no'} | {pc(l['on_beat_at_lead'])}/{pc(l['null95_of_best'])}/{pc(l['chance'])} | {l['cuts_lead_beat_s']} / {l['cuts_lead_beat_frames']} | {l['cuts_lead_attack_frames']} | {pc(s['on_beat'])} | {pc(s['chance_beat'])} | {pc(s['on_downbeat'])} | {pc(max(x or 0 for x in s['downbeat_by_phase']))} | {pc(s['chance_downbeat'])} | {pc(s['on_phrase4'])} | {pc(max(x or 0 for x in s['phrase4_by_phase']))} | {pc(s['chance_phrase4'])} | {pc(f['grid']['whole_bars_share'])} | {pc(s['on_strong_onset'])}/{pc(s['chance_strong_onset'])} |")
print("\nT3 pace\n| film | median shot by fifth (s) | corr cuts~RMS | corr cuts~onset | fastest at / energy rank | slowest at / rank | loudest at |")
for f in F:
    p=f['pace']
    print(f"| {f['slug']} | {' · '.join(f2(x,1) for x in p['median_shot_by_fifth'])} | {p['corr_cuts_rms']} | {p['corr_cuts_onset']} | {p['fastest_at']} / {p['fastest_energy_rank']} | {p['slowest_at']} / {p['slowest_energy_rank']} | {p['loudest_at']} |")
print("\nT4 scale\n| film | screen time A/W/M/C | median s A / W / M / C | cut changes scale | first/last | aerial positions |")
for f in F:
    s=f['scale']; st=s['screen_time']; ls=s['length_by_scale']
    print(f"| {f['slug']} | {'/'.join(pc(st[n]) for n in ['aerial','wide','medium','close'])} | {' / '.join(f2(ls[n]['median'],1)+'('+str(ls[n]['n'])+')' for n in ['aerial','wide','medium','close'])} | {pc(s['cut_changes_scale'])} | {s['first_shot']}/{s['last_shot']} | {s['aerial_positions'] if len(s['aerial_positions'])<=14 else str(len(s['aerial_positions']))+' shots'} |")
print("\nT5 moves\n| film | held-like share | aerial held-like med(n) | aerial moving med(n) | ground held-like | ground moving | moving shots n | in already moving | in from rest | out still moving | out at rest |")
for f in F:
    m=f['moves']; e=m['cut_edges']; d=lambda k: f2(m[k]['median'],1)+'('+str(m[k]['n'])+')'
    print(f"| {f['slug']} | {pc(m['share_held_like'])} | {d('aerial_held_like')} | {d('aerial_moving')} | {d('ground_held_like')} | {d('ground_moving')} | {e['n']} | {pc(e['in_already_moving'])} | {pc(e['in_from_rest'])} | {pc(e['out_still_moving'])} | {pc(e['out_at_rest'])} |")
print("\nT6 move kinds pooled by group (median s, n)")
for kind in ['drone','event','doc','own']:
    for sc in ['aerial','ground']:
        d={}
        for f in F:
            if f['kind']!=kind: continue
            for s in f['shots']:
                if s.get('card') or s.get('title') or s['dark'] or s['t1']>f['music_end']+0.5 or not s['move']: continue
                if (s['scale']=='aerial')==(sc=='aerial'): d.setdefault(s['move'],[]).append(s['length'])
        print(kind,sc,' | '.join(f"{k} {np.median(v):.1f}s [{np.percentile(v,25):.1f}–{np.percentile(v,75):.1f}] n={len(v)}" for k,v in sorted(d.items(),key=lambda kv:-len(kv[1]))))
print("\nT7 scale sequences")
for f in F: print(f['slug'], f['scale']['sequence'][:140])
