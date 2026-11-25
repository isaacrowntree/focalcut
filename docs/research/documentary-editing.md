# Cutting a slow, documentary-style event film: research and measured references

Research for a timing/planning engine for contemplative films (a church beach baptism: drone + tripod + handheld,
one song, 3–4 minutes). Two parts: what editors say (web, sourced), and what 17 finished films actually do
(measured with focalcut's own analysers). It ends with numbered rules a planner could implement.

Measurements: [`references.json`](references.json). Films: `J:\Videos\References\edit-research\` (+ `index.json`).
Measurement scripts (not part of focalcut): `/tmp/claude-1000/-home-isaac/63c15aab-8602-49c9-a804-16bfaac61f86/scratchpad/research/`
(`measure_a.py`, `measure_b.py`, `summarise.py`, `latency.py`, `tables.py`, `sheet.py`) — a scratch folder that a
reboot may clear; copy them somewhere durable if the measurements are to be repeated.

## How far to trust this

- **Written sources.** Pages were read through a fetch tool that returns extracts, not raw HTML. Quotes below are as
  that tool returned them: spot-check before quoting anyone. Items marked *(snippet)* come from search-result text
  only. Source types: **A** academic, **B** book via a secondary page, **T** trade publication, **V** single-author
  or vendor blog (weakest), **F** forum of working editors.
- **Thin areas.** Baptism-specific editing advice: one practitioner blog. Frame counts for cutting ahead of the
  beat: two vendor blogs (our measurements below are better evidence). No source gives a share of runtime for
  drone footage, a documentary average shot length, or a list of "never cut to" defects.
- **Measurements.** 17 films, chosen by search, not sampled. Three of the four beach-baptism films have under
  1,100 views: they are on-topic, not proven craft. Cuts are ffmpeg scene-score cuts plus a SigLIP check for
  dissolves; white-flash transitions and same-scene dissolves are sometimes missed, and light leaks or strobing can
  add false cuts (the sub-second medians in the wedding and Passion films are partly real stutter-cutting, partly
  this). Tempo is whatever librosa tracked: a ballad may be tracked at double time and a 6/8 song is read as 4/4,
  so "beats" and "bars" are less certain than seconds. Shot scale is zero-shot SigLIP and was checked by eye on
  three films: **aerial vs ground is reliable; wide / medium / close is not** (it calls most ground shots "medium").
  Camera-move labels on handheld ground shots are noisy ("spin" there is hand rotation); on drone shots they are
  usable. The four short documentaries include interview and narration passages; their numbers are whole-film,
  not montage-only.

---

## Part 1 — what editors say

### 1.1 Cut timing against music

| Claim | Numbers | Source |
|---|---|---|
| Murch's Rule of Six ranks what a cut must serve: emotion, story, rhythm, eye-trace, 2D plane, 3D space. The order matters more than the percentages; "don't ever give up emotion before story". | 51 / 23 / 10 / 7 / 5 / 4 % | B — [PremiumBeat](https://www.premiumbeat.com/blog/when-and-where-to-make-the-cut-inspired-by-walter-murchs-in-the-blink-of-an-eye/), [No Film School](https://nofilmschool.com/2016/11/6-rules-good-cutting-according-oscar-winning-editor-walter-murch) |
| A blink marks a completed thought; cut where the viewer would blink. Pape's paraphrase: Murch cuts one frame before a blink. "Come late, leave early." | 1 frame (paraphrase) | B/T — [PremiumBeat](https://www.premiumbeat.com/blog/cutting-on-the-blink-editing-tips-from-walter-murch), [No Film School](https://nofilmschool.com/2018/02/7-simple-editing-tips-will-help-you-edit-pro) |
| Do not cut on the beat throughout: "doing it on every cut will start to feel redundant". Keep beat hits for moments that matter, and place them "just a few frames before or after the change in the song". | no frame count | T — [PremiumBeat, Noam Kroll](https://www.premiumbeat.com/blog/how-to-edit-a-film-to-music-without-it-becoming-a-music-video) |
| Anticipation: cuts "land 2-4 frames before the beat rather than exactly on it"; "nudge by one to three frames and compare". | 1–4 frames early | V — [Wideframe](https://try.wideframe.com/blog/how-to-use-ai-to-match-cuts-to-music-beats/), [Kudoflix](https://kudoflix.com/blog/2026/10/27/cut-to-the-beat/) |
| Downbeats are "the strongest rhythmic anchors": keep them for scene changes and reveals. Phrase boundaries fall "every 4 or 8 measures": use them for section changes. | 4 or 8 bars | V — Wideframe, Kudoflix (above) |
| Holding is emphasis: "holding a single shot for 4 or 8 beats draws the viewer's attention". | 4–8 beats | V — Wideframe |
| Cut rate by song section: verse "one cut every bar or two"; a build goes "one cut per bar to one per two beats to one per beat". | 1–2 bars → 1 beat | V — Kudoflix |
| Make the action land on the beat, not the cut: "cut so the ACTION in the SHOT 'hits' on the beat"; all-beat cutting is "metronomic and overly predictable". | — | F — [Creative COW](https://creativecow.net/forums/thread/music-videos-edit-to-the-beat-or-lyrics-2/) |
| Even for music videos, beat cuts are "more of a guideline and an opportunity but not a hardline rule". | — | T — [Videomaker](https://www.videomaker.com/article/c3/17048-rocking-out-with-a-music-video/) |
| Rhythm is "time, energy, and movement shaped by timing, pacing, and trajectory phrasing" to shape "cycles of tension and release" (Pearlman, *Cutting Rhythms*). | — | B *(snippet; review read)* — [Senses of Cinema](https://www.sensesofcinema.com/2026/book-reviews/feeling-the-screen-storys-pulse-karen-peralmans-cutting-rhythms-creative-film-editing/) |
| Pace over a film is scalloped, not a ramp: the longest shots sit at the beginning, the end and act boundaries; mid-act shots are "as much as 1.1 seconds shorter". | 1.1 s | A — [Thompson on Cutting et al.](https://www.davidbordwell.net/blog/2011/06/09/cognitive-scientists-1-screenplay-gurus-0/print/) |
| At the climax shots shorten and motion rises, then shots lengthen and motion falls. | — | A *(snippet)* — [Cutting 2016](https://link.springer.com/article/10.1186/s41235-016-0029-0) |
| Shot lengths in well-made films fluctuate at several timescales (1/f), "tens of seconds" and longer. | — | A — [Cornell on Cutting 2010](https://news.cornell.edu/node/271563) |
| Shot-length distributions are skewed: median ≈ 0.62 × mean across 1,520 features. | 0.62 | A — [Barry Salt, Cinemetrics](https://cinemetrics.uchicago.edu/article/616e7ecc-7915-4768-b84d-7dec79aa77c2) |

### 1.2 Drone footage

| Claim | Numbers | Source |
|---|---|---|
| Finished drone clip length. "If a shot doesn't change or build, it's too long." | 3–5 s; 3–6 s; 5–10 s for slow films | V — [drone.courses](https://drone.courses/blog/drone-video-editing-beginners/), [Uplink](https://www.uplinkimaging.com/pages/edit-drone-footage-premiere-pro), [PickDrones](https://pickdrones.com/how-to-edit-drone-footage/), [Uplink shots](https://www.uplinkimaging.com/pages/cinematic-drone-shots) |
| A reveal "lands its impact in the first 4 seconds"; 12 s "dilutes the effect". | 4 s; not 12 s | V — PickDrones |
| "You're not going to have a place for 20-second drone shots" unless "speeding up the middle of it". | < 20 s | T — [DJI wedding tips](https://store.dji.com/event/content-mavic-3-wedding-shot-tips) |
| Orbit: never more than 1.5 turns in the edit; fly a circle in 8–12 s, a 180° pan in 15–25 s. | 1.5 turns | V — [Drone U](https://www.thedroneu.com/blog/types-of-shots-in-drone-cinematography/) |
| Placement by move: fly-over for "opening shots, scene transitions"; push-in for "introducing a location"; pull-back for "ending a video … revealing scale"; top-down rise as an opener. | — | V — Drone U, Uplink shots |
| Open and/or close on an aerial; "the drone helps transition the story". Aerials establish "in a matter of seconds". | — | T — [Adorama](https://www.adorama.com/alc/how-to-use-drones-better-storytelling-video), [PremiumBeat](https://www.premiumbeat.com/blog/drone-footage-pro-and-con) |
| Use sparingly: "a couple times per video"; "less is more"; "don't fall into the trap of creating a drone promo". No source gives a share of runtime. | ~2 uses | V/T — [TunePocket](https://www.tunepocket.com/wedding-videography-tips/), [PremiumBeat](https://www.premiumbeat.com/blog/4-easy-ways-to-make-your-drone-footage-more-cinematic/), DJI |
| Speed ramp: fast, then "drop to 30-40% right as you reach the subject", eased both ways; speed up long tracking shots and slow "when landing on something of interest". | 30–40 % | V/T — drone.courses, [Streaming Media](https://www.streamingmedia.com/Producer/Articles/Editorial/Featured-Articles/Five-Creative-Techniques-for-Editing-Drone-Video-116692.aspx) |
| Trim the start and end of every clip "so you only have the best part". | — | V/T — drone.courses, [Videomaker](https://www.videomaker.com/article/c6/17123-how-to-get-cinematic-drone-shots/) |
| Cutting into a move — sources disagree: "Cut on movement" (Uplink) vs cut before a move starts or after it ends, "never in the middle" (PickDrones). Our measurements settle it for finished films: see 2.5. | — | V |
| Hyperlapse: DJI's finished hyperlapse runs up to 30 s. No source says how long to hold one in an edit. | ≤ 30 s | T — [Adorama](https://www.adorama.com/alc/drone-hyperlapse-videos) |

### 1.3 Documentary and event sequencing without dialogue

| Claim | Numbers | Source |
|---|---|---|
| Five-shot sequence: hands, face, medium, over-the-shoulder, a creative angle. Sequences usually run wide → medium → close: "the wide for context, the medium for the action, the detail for the emotion". | 5 shots | T/V — [Poynter](https://poynter.org/?p=183861), [Adorama](https://www.adorama.com/alc/create-better-videos-shooting-in-sequences), [PlayPause](https://playpause.io/blogs/capturing-b-roll-that-elevates-your-edit) |
| Wide shots need longer than close-ups: across 31,000 shots, "close-ups were briefer" and "longer-scaled shots were longer in duration". | no ratio read | A — [Cornell on Cutting](https://news.cornell.edu/node/285439) |
| 30-degree rule: a cut to an angle less than 30° away is jarring; change size and angle at each cut; cutting on action hides larger changes. | 30° | T — [No Film School](https://www.nofilmschool.com/30-degree-rule-dps-editors) |
| Cut "during a moment of action". | — | V — [Bride & Groom Video](https://bridengroom.video/blog/wedding-film-editing-techniques/) |
| Reaction shots hold "a beat or two, just long enough to read the expression"; the shot before "does half the work". | 1–2 beats | V — [Morphic glossary](https://morphic.com/resources/shots/reaction-shot) |
| Shoot holds of 10–15 s; the edit uses about 5 s. Editors want a 10 s handle each side. | 10–15 s raw, ~5 s used | T — [Baruch course](https://blogs.baruch.cuny.edu/summermultimedia/?p=257), [Fstoppers](https://fstoppers.com/education/follow-these-six-shooting-tips-make-editing-breeze-194011), [No Film School](https://nofilmschool.com/b-roll-editor-tips) |
| "One deliberate movement per shot": lock off or move, not both. An entrance or exit is a natural in- or out-point. | — | V — [Ken Jones](https://www.kenjonesnyc.com/b-roll-planning/) |
| J-cut: the next scene's sound starts under the current picture (two wedding blogs define it backwards). | — | [Wikipedia](https://en.wikipedia.org/wiki/J_cut) |
| Highlight films are ordered by emotion and song, not time: "save the order for the full-length film". | — | T — [PremiumBeat](https://www.premiumbeat.com/blog/5-tips-editing-cinematic-wedding-highlight-videos) |

### 1.4 Baptism, church and wedding films

| Claim | Numbers | Source |
|---|---|---|
| The one baptism editor found: music drives the emotional arc; each baptism is its own segment, ordered by emotional flow; a whole family's baptisms are placed on a key song section; the song should grow "from a meek beginning to a full and rich melody"; everything is shot for slow motion. | 60 and 240 fps | V — [Noble Limits](http://www.noblelimits.com/journal/baptism-videos) |
| Wedding highlight length. | 3–5 min (one source: 5–8) | V — [Novacut](https://novacut.ai/guides/how-to-edit-wedding-video), [Atavia](https://ataviaweddings.com/blog/wedding-highlight-reel-best-practices-for-2026) |
| Arc: rising action to the kiss as climax; at the climax "time slows down" and "the music should peak here too"; then "exhale"; end on fade to black. | — | T/V — [Videomaker](https://www.videomaker.com/article/c3/15368-cinematic-is-in-editing-wedding-videos/), [Roughcut](https://roughcut.com/blog/how-to-structure-wedding-highlight-film) |
| Hold emotional shots; hold the first kiss before cutting away; tempo that suits emotional pacing. | 6–10 s; ≥ 3 s; 90–110 BPM | V, single source — Atavia |
| Slow motion on only two or three key moments. | ~50 % speed | V — Novacut |
| Keep natural sound (laughter, applause, water) quietly under the music. | no levels given | V — [Imagen](https://imagen-ai.com/valuable-tips/wedding-video-editing/), Atavia |

### 1.5 Music video against documentary

| Claim | Numbers | Source |
|---|---|---|
| 548 official music videos: average shot 4.76 s, average bar 2.30 s. Shot length matches the beat in two thirds of videos, the bar in a fifth, four bars almost never — and "their boundaries might not always co-occur". | 4.76 s | A — [arXiv 2108.00970](https://ar5iv.labs.arxiv.org/html/2108.00970) |
| Feature films now average about 4 s a shot. | ~4 s | A — [Cornell](https://news.cornell.edu/node/271563) |
| Documentaries "tend to have the least number of cuts". No average found. | — | T — [BFI](https://www.bfi.org.uk/sight-and-sound/videos/movie-musical-evolution-dance-film) |
| A montage compresses time with separate images; a sequence plays an event in continuity. | — | T — [StudioBinder](https://www.studiobinder.com/blog/what-is-a-montage-definition/) |

---

## Part 2 — what 17 finished films do

### 2.1 The films

| Slug | Film | Channel | Views | Length |
|---|---|---|---|---|
| drone-dolomites-chasing-light | [DOLOMITES: Chasing Light](https://www.youtube.com/watch?v=lmpDQq7bPjk) | Denis Barbas | 401 k | 4:46 |
| drone-istanbul-timelab | [Istanbul. Love of the continents](https://www.youtube.com/watch?v=eV6lTEY95yY) | Timelab Pro | 5.27 M | 3:22 |
| drone-saudi-from-above | [Saudi Arabia from Above](https://www.youtube.com/watch?v=G5KCKKia61o) | Stef Hoffer | 481 k | 4:25 |
| drone-autumn-stillness-dolomites | [Autumn Stillness in the Dolomites](https://www.youtube.com/watch?v=qDIt3h6CKgo) | darylswalker | 168 k | 6:23 |
| drone-iceland-drone-trip | [Iceland 4K, Drone Trip](https://www.youtube.com/watch?v=4OfD10P6OXc) | Gaëtan Piolot | 257 k | 5:30 |
| baptism-sunlightcc-beach-2019 | [Beach Baptism Spring 2019](https://www.youtube.com/watch?v=Gmog2cm1630) | sunlightcc | 158 k | 3:30 |
| baptism-renew-oc-beach-2021 | [June Beach Baptism, Renew Church OC](https://www.youtube.com/watch?v=0UIXcQ2TwoI) | Greg Wan | 295 | 2:25 |
| baptism-living-faith-beach-2025 | [Beach Baptism 2025](https://www.youtube.com/watch?v=HLEoGVP2KWM) | Living Faith Christian Church | 929 | 2:58 |
| baptism-crossview-highlight-2022 | [Baptism Highlight, June 2022](https://www.youtube.com/watch?v=RxOzTAt4edU) | Crossview Church | 1.1 k | 3:05 |
| wedding-adrianna-carson-fx3 | [An Emotional, Fun & Cinematic Wedding Film](https://www.youtube.com/watch?v=-ARVwU58l7A) | Isaac (a YouTube channel) | 402 k | 6:36 |
| wedding-tobifilms-lake-como | [The most emotional wedding film at rainy Lake Como](https://www.youtube.com/watch?v=G9VfCbSWzNA) | TOBIFILMS | 221 k | 6:24 |
| recap-passion-2026 | [Passion 2026 Recap](https://www.youtube.com/watch?v=B0OyHF9nD34) | passion | 1.82 M | 4:58 |
| doc-patagonia-too-old-not-to | [Too Old Not To](https://www.youtube.com/watch?v=ljp5H2pjAFc) | Patagonia Australia | 720 k | 6:35 |
| doc-catching-breath-cold-water | [Catching Breath](https://www.youtube.com/watch?v=Lq1Ym738dlk) | Fierce Beastie Productions | 62 k | 4:57 |
| doc-cast-out-with-love-staff-pick | [Cast Out With Love (Vimeo Staff Pick)](https://www.youtube.com/watch?v=VVUHIlZUZH8) | Greg Dennis | 1.4 k | 4:17 |
| doc-surfdome-battery-rocks-swimmers | [Cold water swimmers at Battery Rocks](https://www.youtube.com/watch?v=xbYXH8q54gI) | Surfdome | 6.3 k | 6:51 |
| malabar-2025-reference | The project's own 2025 reference film (the one the current cut copies) | — | — | 4:25 |

yt-dlp was refused by YouTube (HTTP 403 after the first chunk, both installed builds); all 16 were fetched with
`pytubefix` at 1080p H.264.

### 2.2 Shot length

Seconds are solid; beats depend on the tracked tempo. "≥ 2 bars" is the share of shots at least 8 tracked beats long.

| Film | Cuts/min | Median s | IQR s | p10–p90 s | Longest s | Tempo | Median beats | ≥ 2 bars | ≥ 4 bars | Dissolves |
|---|---|---|---|---|---|---|---|---|---|---|
| drone-dolomites | 3.8 | 12.6 | 11.0–19.9 | 10.7–23.8 | 32.9 | 129 | 27 | 100 % | 100 % | 0 |
| drone-istanbul | 12.2 | 3.4 | 1.8–4.3 | 1.4–8.7 | 28.0 | 140 | 8.0 | 54 % | 18 % | 4 |
| drone-saudi | 10.8 | 4.5 | 3.8–7.6 | 3.7–7.8 | 13.6 | 125 | 9.4 | 64 % | 18 % | 4 |
| drone-autumn-stillness | 7.4 | 7.9 | 5.2–9.6 | 4.5–11.6 | 24.1 | 123 | 16 | 96 % | 51 % | 0 |
| drone-iceland | 16.0 | 3.1 | 2.3–4.8 | 2.1–6.0 | 9.2 | 140 | 7.1 | 45 % | 5 % | 5 |
| **baptism-sunlightcc** | 10.1 | **5.7** | 3.7–7.1 | 2.4–9.2 | 19.0 | 102 | 9.8 | 61 % | 12 % | 2 |
| baptism-renew-oc | 21.6 | 2.5 | 1.5–2.9 | 1.4–3.4 | 13.0 | 126 | 5.3 | 9 % | 4 % | 3 |
| baptism-living-faith | 21.8 | 1.9 | 1.1–3.3 | 0.4–4.4 | 16.1 | 144 | 4.6 | 24 % | 5 % | 7 |
| baptism-crossview | 26.7 | 1.9 | 1.3–2.5 | 1.0–4.0 | 11.0 | 110 | 3.6 | 5 % | 1 % | 1 |
| wedding-adrianna-carson | 42.3 | 0.8 | 0.4–1.8 | 0.2–3.4 | 12.4 | 105 | 1.4 | 7 % | 1 % | 9 |
| wedding-tobifilms | 33.6 | 1.6 | 1.0–2.3 | 0.4–3.4 | 7.2 | 129 | 3.4 | 8 % | 0 % | 0 |
| recap-passion | 59.8 | 0.7 | 0.3–1.2 | 0.2–1.9 | 7.7 | 144 | 1.7 | 4 % | 0 % | 11 |
| doc-patagonia | 25.5 | 1.9 | 1.2–3.0 | 0.6–4.4 | 9.8 | 129 | 4.1 | 17 % | 3 % | 8 |
| doc-catching-breath | 23.1 | 1.6 | 0.4–3.5 | 0.1–6.4 | 15.8 | 100 | 2.7 | 18 % | 4 % | 7 |
| doc-cast-out-with-love | 23.8 | 2.2 | 1.6–3.1 | 1.2–4.1 | 5.8 | 129 | 4.8 | 17 % | 0 % | 1 |
| doc-battery-rocks | 11.6 | 4.4 | 2.9–6.3 | 1.7–8.8 | 20.1 | 118 | 8.6 | 59 % | 14 % | 14 |
| **malabar-2025-reference** | 22.2 | **2.4** | 1.8–3.3 | 1.4–4.5 | 7.1 | 122 | 4.9 | 16 % | 0 % | 0 |

Reading it:

- There are two speeds of "slow". The genuinely contemplative films — sunlightcc, Saudi, Battery Rocks, Autumn
  Stillness — sit at a **median of 4.4–7.9 s and 7–12 cuts a minute**, with more than half their shots two bars or
  longer. The other baptism films and three of the four documentaries sit at **1.6–2.5 s and 21–27 cuts a minute**; the two
  weddings and the Passion recap are faster still (0.7–1.6 s, 34–60 cuts a minute).
- **The 2025 reference is in the faster group** (2.4 s, 22 cuts/min, no shot of four bars, longest 7.1 s). A cut
  that copies it shot for shot inherits a highlight-reel pace, not a contemplative one.
- The slow films have almost no short shots: their 10th percentile is 1.7–4.5 s. The fast ones reach 0.1–0.6 s.
- Hard cuts dominate everywhere (0–14 dissolves per film; Battery Rocks is the only dissolve-led one).

### 2.3 Cuts against the beat

"Lock" is the test that matters: slide every hard cut by one constant offset and find the offset that puts most
cuts within ±0.045 s of a beat; compare with the same search over randomly jittered cuts (the null's 95th
percentile). "Lead" is that offset, first against focalcut's tracked beat, then against the audible attack — the
tracker places beats 0.04 s after the attack in every film measured (`latency.py`). Positive lead = the cut comes
first. After that, cuts are scored at their own lead, within ±0.07 s.

| Film | Locked? | Cuts on a beat at best lead (null 95 %) | Lead vs tracked beat | Lead vs attack | On a beat (chance) | On a downbeat (chance) | Shots a whole number of bars (chance 12.5 %) |
|---|---|---|---|---|---|---|---|
| drone-istanbul | **yes** | 69 % (30 %) | 0.10 s | 0.06 s = 1.5 frames | 81 % (33 %) | 33 % (8 %) | 54 % |
| baptism-sunlightcc | **yes** | 59 % (27 %) | 0.09 s | 0.05 s = 1.4 frames | 66 % (24 %) | **59 % (6 %)** | 55 % |
| baptism-renew-oc | **yes** | 52 % (30 %) | 0.11 s | 0.07 s = 1.6 frames | 66 % (29 %) | 18 % (7 %) ¹ | 4 % ¹ |
| drone-iceland | **yes** | 51 % (30 %) | 0.10 s | 0.06 s (3.6 frames at 60 fps) | 64 % (32 %) | 5 % (8 %) ² | 2 % |
| drone-saudi | **yes** | 44 % (30 %) | 0.00 s | −0.04 s = 1 frame late | 56 % (29 %) | **56 % (7 %)** | 58 % |
| drone-autumn-stillness | yes | 38 % (29 %) | 0.04 s | 0.00 s | 51 % (29 %) | 21 % (7 %) | 13 % |
| malabar-2025-reference | weak | 34 % (26 %) | −0.02 s | −0.06 s = 2 frames late | 38 % (28 %) | 8 % (7 %) | 15 % |
| doc-patagonia | weak | 28 % (25 %) | 0.12 s | 0.08 s | 34 % (30 %) | 8 % (8 %) | 12 % |
| doc-catching-breath | weak | 26 % (22 %) | 0.08 s | 0.04 s | 30 % (23 %) | 5 % (6 %) | 9 % |
| baptism-crossview | marginal | 25 % (23 %) | 0.06 s | 0.02 s | 32 % (26 %) | 21 % (6 %) | 14 % |
| wedding-adrianna-carson | marginal | 20 % (20 %) | — | — | 31 % (24 %) | 11 % (6 %) | 7 % |
| wedding-tobifilms | marginal | 24 % (24 %) | — | — | 34 % (30 %) | 8 % (8 %) | 11 % |
| drone-dolomites, baptism-living-faith, recap-passion, doc-cast-out, doc-battery-rocks | **no** | 24–33 % (26–35 %) | — | — | 28–35 % (27–34 %) | at chance | 6–12 % |

¹ Twelve of renew-oc's 45 shots are exactly six tracked beats long: the song is almost certainly in 6/8 or 3/4, so
its cuts are on bar lines too, which the 4/4 downbeat column cannot see. ² Iceland's best bar phase scores 25 %:
the tracker's downbeat is probably one beat out.

Reading it:

- **The films that feel composed cut on the bar, not on the beat.** In sunlightcc and Saudi more than half of all
  cuts sit on a downbeat (chance 6–7 %); of their on-beat cuts, 91 % and 100 % are downbeats. Their shots are whole
  bars long: sunlightcc has 6 shots of two bars, 6 of three, 4 of one, 1 of four; Saudi 14 of two bars, 8 of four,
  3 of three. Four-bar phrase starts take 20–25 % of their cuts (chance 2 %).
- **They cut ahead of the beat.** Four independently made films (Istanbul, sunlightcc, renew-oc, Iceland) all lead
  focalcut's tracked beat by **0.09–0.11 s**, which is 0.05–0.07 s — one and a half to two frames — before the
  audible attack. That matches the "1–3 frames early" folklore with real numbers. Two others sit on the attack or a
  frame behind it. None of the locked films lags by more than a frame. (An export or upload audio offset could
  mimic this in one film; four films agreeing makes it unlikely.)
- **Not every cut is on the grid even then:** 19–49 % of cuts in the locked films are off any beat. Those are cuts
  made for the action.
- **Documentaries, weddings and the fast recap do not cut to the grid at all** — they are at chance on beats and
  downbeats. They cut on action, speech and sound. Catching Breath's cuts sit on sound onsets (50 % against 28 %
  chance) without sitting on beats.
- **The 2025 reference is nearly unlocked** (34 % against a 26 % null), at chance on downbeats, and its cuts land
  about two frames *after* the attack. This is the measurable form of "missing better beats".

### 2.4 Pace over the film, against the music's energy

Cuts per 10 s window against the window's loudness (Spearman), and the median shot length by fifth of the film.

| Film | Median shot by fifth (s) | Cuts ~ loudness | First shot / last shot (s) | Longest shots at (share of film) |
|---|---|---|---|---|
| drone-dolomites | 21 · 12 · 14 · 13 · 11 | 0.15 | 22.0 / 21.0 | 0.24, 0.53, 0.04 |
| drone-istanbul | 8.9 · 1.8 · 6.8 · 3.4 · 6.9 | 0.56 | 4.3 / 24.9 | 0.51, 0.90, 0.12 |
| drone-saudi | 3.8 · 7.6 · 4.8 · 6.0 · 4.8 | 0.19 | 4.5 / 13.6 | 0.92, 0.74, 0.14 |
| drone-autumn-stillness | 6.1 · 4.8 · 8.2 · 6.8 · 8.5 | 0.12 | 11.0 / 8.1 | 0.34, 0.28, 0.45 |
| drone-iceland | 2.3 · 4.5 · 3.0 · 5.6 · 2.5 | 0.06 | 4.5 / 6.3 | 0.77, 0.67, 0.80 |
| baptism-sunlightcc | 4.0 · 7.0 · 6.4 · 7.0 · 7.0 | −0.24 | 2.3 / 19.0 | 0.92, 0.49, 0.43 |
| baptism-renew-oc | 1.9 · 2.0 · 2.8 · 2.8 · 2.6 | 0.03 | 5.0 / 13.0 | 0.90, 0.48, 0.71 |
| baptism-living-faith | 3.5 · 3.9 · 1.6 · 2.4 · 0.8 | 0.69 | 14.5 / 12.3 | 0.62, 0.06, 0.97 |
| baptism-crossview | 1.4 · 2.1 · 2.7 · 1.8 · 1.6 | −0.30 | 4.0 / 4.3 | 0.89, 0.42, 0.59 |
| wedding-adrianna-carson | 0.3 · 2.2 · 1.8 · 0.4 · 0.8 | 0.41 | 0.7 / 3.5 | 0.32, 0.66, 0.24 |
| wedding-tobifilms | 1.5 · 1.7 · 2.1 · 1.7 · 0.9 | 0.51 | — / 1.3 | 0.64, 0.30, 0.49 |
| recap-passion | 0.5 · 0.9 · 0.8 · 1.0 · 0.5 | 0.63 | — / 3.3 | 0.96, 0.24, 0.63 |
| doc-patagonia | 1.8 · 1.8 · 2.1 · 1.9 · 2.2 | 0.02 | — / 2.4 | 0.43, 0.39, 0.71 |
| doc-catching-breath | 4.9 · 3.9 · 3.4 · 0.4 · 0.4 | 0.59 | 4.9 / 3.5 | 0.36, 0.21, 0.40 |
| doc-cast-out-with-love | 2.2 · 1.8 · 2.9 · 2.8 · 1.7 | 0.39 | 3.8 / 1.7 | 0.78, 0.76, 0.35 |
| doc-battery-rocks | 4.8 · 4.4 · 4.8 · 5.0 · 2.8 | 0.01 | 6.5 / 7.2 | 0.62, 0.86, 0.75 |
| malabar-2025-reference | 3.2 · 1.9 · 3.0 · 2.5 · 1.9 | −0.20 | 4.5 / 2.4 | 0.45, 0.60, 0.43 |

Reading it:

- **Contemplative films do not chase the music's loudness with cut rate.** sunlightcc, Saudi, Autumn Stillness,
  Battery Rocks, renew-oc and Patagonia are flat (−0.24 to 0.19). Energy shows in *what* is on screen, not in how
  often it changes. The recaps that accelerate into the loud sections (living-faith, Passion, tobifilms, Catching
  Breath: 0.5–0.7) are the ones that read as highlight reels. This is the clearest difference from a
  beat-every-cut music video.
- **The mid-section is the slow one.** In sunlightcc, renew-oc, crossview and Battery Rocks the middle fifths hold
  longer shots than the first: the baptisms themselves play long, the arrival montage plays short. That is the
  reverse of a build-to-drop edit, and it agrees with the wedding advice that time slows at the climax.
- **The last shot is long.** In 7 of 17 films one of the three longest shots sits in the last tenth or so (0.89–0.97 of the way through: Istanbul 24.9 s
  pull-back, Saudi 13.6 s pull-back, sunlightcc 19 s group photo into the fade, renew-oc 13 s, living-faith 12.3 s,
  Passion 7.7 s pull-back, crossview 11 s). The reference's last shot is 2.4 s.
- **Openers are longer than the median** in 11 of the 14 films with a clean first shot (1.3–7.6× the median); sunlightcc
  is one of the three exceptions only because its 5.6 s title shot is not counted.
- Long shots cluster: the lag-1 correlation of log shot length is positive in 13 of 17 films (0.15–0.78). Pace
  changes by section, not shot by shot.

### 2.5 Camera moves, and where cuts fall in a move

Pooled over the five drone films, aerial shots only (move labels from `moves.label`):

| Move | Median s | IQR s | Shots |
|---|---|---|---|
| drift (slow travel) | 5.8 | 3.9–9.6 | 62 |
| held | 6.0 | 3.8–8.3 | 36 |
| push (push-in / descend) | 4.5 | 3.0–7.7 | 48 |
| spin (orbit, top-down turn) | 3.4 | 2.1–4.5 | 19 |
| pull (pull-back, climb, reveal) | 3.4 | 2.5–5.6 | 14 |
| pan | 3.2 | 2.8–4.1 | 4 |

Aerial shots inside ground-led films are shorter: 3.3 s in the reference (5 shots), 2.1–3.8 s for the common moves in the four
documentaries (40 shots), 1.3–2.1 s in the event films (10 shots). The closing pull-backs are the exception (13.6
and 24.9 s).

Where the cut falls, for moving shots of 2 s or more (camera speed in the first and last 0.4 s against the shot's
median speed):

| | Drone films | Event films | Documentaries | 2025 reference |
|---|---|---|---|---|
| Already moving at the in-cut (≥ 60 % of cruising speed) | 85–98 % | 68–100 % | 84–96 % | 84 % |
| Starts from rest inside the shot (< 30 %) | 0–8 % | 0–8 % | 0–3 % | 0 % |
| Still moving at the out-cut | 94–100 % | 81–100 % | 80–92 % | 90 % |
| Comes to rest before the out-cut | 0–3 % | 0–7 % | 0–4 % | 5 % |

**Finished films cut into a move that is already under way and out before it stops** — in every one of the 17.
That settles the disagreement in 1.2: "cut on movement" is what editors do; the ramp-up and ramp-down are trimmed
away. (One limit: a move that changes direction mid-shot is not separated here.)

### 2.6 Aerials in ground-led films, and scale

| Film | Aerial share of screen time | Aerial shots at (share of film) | Opens / closes on |
|---|---|---|---|
| malabar-2025-reference | 7 % (5 shots) | 0.02, 0.07, 0.38, 0.51, 0.87 | aerial / ground |
| wedding-tobifilms | 3 % (5) | 0.01, 0.21, 0.26, 0.32, 0.90 | ground / ground |
| baptism-crossview | 3 % (3) | 0.07, 0.12, 0.17 | ground / ground |
| doc-battery-rocks | 4 % (3) | 0.15, 0.52, 0.82 | ground / ground |
| doc-cast-out-with-love | 3 % (2) | 0.43, 0.68 | ground / ground |
| doc-catching-breath | 2 % (2) | 0.63, 0.89 | ground / ground |
| doc-patagonia (a surf film) | 30 % (33) | throughout | ground / ground |
| sunlightcc, renew-oc, living-faith | 0 % | — | — |

- Outside the surf film, aerials take **2–7 % of the running time in 2–5 shots**: near the opening (first 10–20 %),
  once or twice mid-film as a breath between sections, and once near the end (0.82–0.90). This is the "couple of
  times per video" of the written advice, measured. Three of four baptism films use none.
- All five drone films open and close on an aerial, as expected.
- Wide against close: where the classifier found both, wides run longer in living-faith (13.4 s against 1.6 s) and
  Patagonia (aerial 2.9 s, medium 1.9 s, close 0.9 s), and about the same elsewhere. With the classifier this
  weak, treat the direction as supported by the academic source (1.3), not by these numbers.

### 2.7 How the immersion is built (read by eye from contact sheets)

Three films, five frames per shot: sunlightcc (all 35 shots), renew-oc (shots 14–33), the 2025 reference (shots
30–51). Small sample, but the three agree.

- **One baptism is one continuous shot.** Hands on → lean back → under → up → face. It is never cut into pieces
  mid-dunk. sunlightcc holds each for 4.7–11.7 s (most 6–8 s, i.e. two to three bars); the reference for 4.5–7.1 s
  when it shows the whole thing; renew-oc's hero dunk is 8.5 s, three times its median.
- **The dunk comes early in the shot.** The person is under water at roughly 25–40 % of the shot's length: about
  1.5–2.5 s of lead-in (hands on, the last words), then 3–5 s of rise, wiped face, smile, the hug starting. The
  shot ends on the reaction, not on the splash.
- **Short baptism shots show only the rise.** The reference's 2.2–2.3 s shots are the coming-up and the smile
  without the lead-in: a quicker beat between two full ones.
- **A cutaway every two to seven baptisms, 2–3 s long:** the crowd watching or filming (sunlightcc 2.3–2.9 s,
  renew-oc 2.8 s, reference 2.7 s), or in the reference a wide / top-down aerial of the same moment (1.9–3.3 s)
  showing the three figures alone in the bay. sunlightcc also uses the crowd cutaway to compress one baptism:
  family shot → crowd → same family, later.
- **The lead-in to the water is a short sequence, not one shot:** renew-oc goes praying hands (close, 1.4–2.9 s
  each) → walking into the surf from behind (three shots, 1.5–2.9 s) → the three in the water, wider each time
  (three shots, 2.8–3.0 s) → dunk in the wide (2.5 + 1.5 s) → the same dunk close, held 8.5 s.
- **The opening establishes by alternating scale:** sunlightcc runs wide crowd → pastor (medium) → wide → medium →
  wide for its first 35 s, 2.3–4.7 s a shot, then one 3 s close-up of a woman praying as the bridge into the
  water.
- **Endings:** sunlightcc ends on the group photo held 19 s into a fade; renew-oc on a 13 s shot.

### 2.8 Documentaries against event recaps and drone films

| | Drone films (5) | Contemplative event (sunlightcc) | Other event recaps (6) | Short documentaries (4) | 2025 reference |
|---|---|---|---|---|---|
| Median shot | 3.1–12.6 s | 5.7 s | 0.7–2.5 s | 1.6–4.4 s (three at 1.6–2.2) | 2.4 s |
| Cuts a minute | 4–16 | 10 | 22–60 | 12–26 | 22 |
| Shots under 1 s (10th percentile) | none (1.4–10.7 s) | none (2.4 s) | common (0.2–1.4 s) | mixed (0.1–1.7 s) | none (1.4 s) |
| Longest shot | 9–33 s | 19 s | 7–16 s | 6–20 s | 7 s |
| Cuts locked to the beat | 4 of 5 | yes, on downbeats | 1 of 6 (renew-oc) | none | weak |
| Cut rate follows loudness | no (4 of 5) | no | yes in 4 of 6 | mixed (2 of 4) | no |
| Cuts into moves already under way | 85–98 % | 68 % | 83–100 % | 84–96 % | 84 % |

- The documentaries' montage is **faster than expected and ignores the beat grid**. Three of four have a median of
  1.6–2.2 s, the same as the event recaps; only Battery Rocks (4.4 s, dissolve-led, the most observational) is
  slow. Their cuts are at chance on beats and downbeats. What they do instead: cut on action and on sound, cluster
  short shots into bursts between long held passages (Catching Breath: 4.9 s shots in the first fifth, 0.4 s in
  the last two; lag-1 correlation 0.78), and let interviews carry the long shots (8–20 s).
- So "documentary" does not mean "slow cutting". It means **continuity of an action inside a shot, long holds on
  people, and cut points chosen by what happens** — with the music as a bed. The beat-locked, bar-long style
  belongs to the music-led pieces: the drone films and the one contemplative baptism film.
- For a baptism film set to one song with no narration, the nearest models are therefore sunlightcc and the drone
  films for *timing* (bar-long shots, cuts just ahead of the downbeat, flat pace, long last shot) and the
  documentaries for *selection* (cut on the action, hold the person, sequence wide → medium → close).

---

## Part 3 — rules for an automatic planner

Each rule: condition → parameters → evidence. "M" = measured here (section), "W" = written source (section).
Numbers marked ★ come from this small sample and should be exposed as settings, not hard-coded.

**Grid and timing**

1. **Cut on bar lines, not beats, in contemplative mode.** Condition: mode is contemplative / reference pace is
   slow. Shot lengths are whole bars — 2 bars the default, 3 or 4 for holds, 1 for a cutaway; no shot under 1 bar.
   At 100–125 BPM that is 3.8–4.8 s for two bars. ★ Target mix from sunlightcc and Saudi: about 20 % one bar, 40 %
   two, 25 % three, 15 % four. — M 2.3 (56–59 % of cuts on downbeats against 6–7 % chance; 55–58 % of shots whole
   bars); W 1.1 (verse = "one cut every bar or two").
2. **Lead the beat.** Every grid cut goes **0.08–0.10 s before focalcut's tracked beat** (= 0.05–0.07 s, 1.5–2
   frames, before the attack; the tracker itself sits 0.04 s late). Never after the tracked beat. — M 2.3 (four
   films at 0.09–0.11 s); W 1.1 (1–4 frames early, vendor blogs; Kroll "a few frames").
3. **Section changes go on phrase starts.** Condition: the content changes scene (arrival → water → celebration)
   or an aerial punctuation begins. Place it on a 4- or 8-bar boundary. ★ About a fifth to a quarter of cuts. —
   M 2.3 (20–25 % of cuts on four-bar starts in the two bar-locked films, chance 2 %); W 1.1.
4. **Let the action override the grid.** Condition: a clip's natural cut point (the action completing, a look, an
   exit) is more than about half a beat from the grid. Keep the action's point; do not force the snap. Expect 20–45 %
   of cuts off the grid even in a bar-led film. Put the *action* (the splash, the arms going up) on a strong beat
   where the footage allows, rather than the cut. — M 2.3 (19–49 % off-beat in locked films; documentaries at
   chance); W 1.1 (Murch: emotion before rhythm; Creative COW).
5. **Check the metre.** Condition: many natural lengths come out at 6 or 3 tracked beats. Treat the song as 6/8 or
   3/4 and use 6- or 3-beat bars. — M 2.3 note 1.

**Pace**

6. **Contemplative pace target.** Median shot **4.5–6 s**, interquartile 3.7–7 s, 10th percentile ≥ 2.3 s, 7–12
   cuts a minute, more than half the shots ≥ 2 bars, no stutters. (The 2025 reference is 2.4 s and 22 cuts a
   minute: a highlight pace.) — M 2.2.
7. **Do not tie cut rate to loudness.** Condition: contemplative mode. Keep shot length roughly constant through
   choruses; spend the music's energy on content (the immersions on the chorus, the widest aerial on the biggest
   bar). ★ Allow at most about 25 % shorter shots in the loudest section. — M 2.4 (correlation −0.24 to 0.19 in the
   slow films against 0.5–0.7 in highlight recaps); W 1.4 (place the key baptisms on a key song section).
8. **The middle is the slow part.** Arrival / setting montage: 1–2 bars a shot. Baptisms: 2–4 bars. Celebration
   and exit: 1–2 bars again. — M 2.4, 2.7; W 1.4 ("time slows down" at the climax); W 1.1 (longest shots at act
   boundaries).
9. **Pace by section, not by alternation.** Group shots of like length (runs of 3–6), change length at phrase
   boundaries. — M 2.4 (positive lag-1 correlation in 13 of 17 films); W 1.1 (1/f).
10. **Open long, end longer.** First shot 1.5–2× the median (≥ 2 bars) with the title over it. Last shot **≥ 2× the
    median, 8–20 s**, the longest or second-longest of the film, a pull-back or a held group / horizon, into a
    fade. — M 2.4 (7 of 17 films end on a top-three-longest shot; closing pull-backs 7.7–24.9 s); W 1.2 (pull-back
    for endings), W 1.4 (fade to black).

**Drone shots**

11. **Length by move** (bars rounded per rule 1): held or slow drift 4–8 s; push-in 3–6 s; orbit / top-down spin
    2.5–4.5 s; pull-back 2.5–5.5 s (the closing one: rule 10). In a ground-led film an aerial is a punctuation of
    **2–4 s**. Nothing over about 10 s unless it is the last shot or is sped up. — M 2.5; W 1.2 (3–6 s typical, 5–10
    s for slow films, reveal lands in 4 s, no 20 s shots, orbit ≤ 1.5 turns).
12. **Cut mid-move.** In-point where the camera is already at ≥ 60 % of the move's cruising speed; out-point before
    it slows below that. Drop the first and last 0.5–1 s of any move and anything with a start, stop or change of
    direction inside the used span. One move per shot. — M 2.5 (84–100 % in every film); W 1.2 (trim both ends),
    W 1.3 ("one deliberate movement per shot").
13. **Where aerials go, and how many.** Ground-led film: 3–5 aerial shots, **3–7 % of running time**: one in the
    first 10 % (the opener or second shot), one or two mid-film on a phrase start as a breath between sections or
    as the wide of an immersion, one at 0.82–0.90 or as the last shot. — M 2.6; W 1.2.
14. **Speed.** A long held or slow-drift clip may be sped up to fit its slot (time-lapse) rather than cut short;
    a ramp slows to 30–40 % as the subject arrives, eased both ways. Thin: two written sources, not measured here.
    — W 1.2.

**Sequence and selection**

15. **One immersion = one unbroken shot of 2–3 bars (5–8 s).** In-point 1.5–2.5 s before the person goes under
    (hands on); the dunk at 25–40 % of the shot; out-point 3–5 s after, on the face or the start of the hug, never
    on the splash. A shorter variant (1 bar, 2–2.5 s) shows only the rise and the smile; use it between full ones.
    — M 2.7 (three films, by eye); W 1.3 (cut on action; arrive late, leave early).
16. **Cutaway rhythm.** After every 2–4 immersions insert one cutaway of 1 bar (2–3 s): the crowd watching or
    cheering, or an aerial / wide of the same moment. A cutaway may also bridge two parts of one baptism to
    compress it. Never two cutaways in a row. — M 2.7; W 1.3 (reaction "a beat or two").
17. **Lead into the water as a sequence.** Praying / listening close-ups (1 bar each) → walking in (1–3 shots, 1
    bar each) → the group in the water, wider each time → the first immersion, held longest. Establish the setting
    before that by alternating wide and medium, 2–5 s a shot, for the first 30–40 s. — M 2.7; W 1.3 (wide → medium
    → close; five-shot sequence).
18. **Change something at every cut.** Consecutive shots differ in scale or in angle by more than 30°; two shots of
    the same baptism from the same position are not cut together without a cutaway between. — W 1.3 (30-degree
    rule). Measured scale changes (14–52 % of cuts) are not reliable enough to set a number.
19. **Wides hold longer than close-ups.** Give a wide or an aerial about one bar more than a close-up of the same
    importance. ★ — W 1.3 (Cutting, 31,000 shots); M 2.6 (direction only).
20. **Order by song and emotion inside chronology.** Keep the day's order across sections (arrival → water →
    celebration); inside the baptism section order by strength — the strongest immersions and any family on the
    song's fullest section, the single best one last. — W 1.3, 1.4; M 2.4.
21. **What not to cut to** (inference; no written source lists these, and they are not measured): a frame with no
    subject, the camera still finding focus or reframing, the first and last half-second of a handheld take, two
    moves in one shot, a shot that only repeats the one before. Require at least 2–3 s of steady material around
    the chosen moment before a clip may fill a 2-bar slot. — W 1.3 (10 s handles; "fills time" → leave out); M 2.5.

**Sound**

22. **Keep the day under the music.** Water, the cheer and the splash stay in quietly under the song; the cheer
    may start a moment before its picture (a J-cut) over the end of the immersion shot. Levels: no source gives
    numbers. — W 1.3, 1.4.

### The short version

A contemplative cut is not a slower music video. It holds shots for whole bars, cuts a frame or two ahead of the
downbeat, keeps the same pace when the song gets loud, plays each key moment as one unbroken shot that ends on the
face, breathes with a crowd or aerial cutaway every few moments, and ends on its longest shot. The 2025 reference
does few of these: it is a 2.4-second highlight cut that barely touches the grid.
