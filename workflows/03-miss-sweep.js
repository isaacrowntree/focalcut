export const meta = {
  name: 'focalcut-miss-sweep',
  description: 'Independently re-read single-segment clips to catch couple changes the conservative coarse pass may have missed',
  phases: [{ title: 'Miss sweep', detail: 'skeptical second read per single-segment clip' }],
}

const SCHEMA = {
  type: 'object',
  properties: {
    n_couples: { type: 'integer' },
    cuts: { type: 'array', items: { type: 'number' }, description: 'interior cut points in seconds; empty if genuinely one couple' },
    disagrees_with_first_pass: { type: 'boolean' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    notes: { type: 'string' },
  },
  required: ['n_couples', 'cuts', 'disagrees_with_first_pass', 'confidence', 'notes'],
}

phase('Miss sweep')
const jobs = typeof args === 'string' ? JSON.parse(args) : args

const results = await parallel(jobs.map((j) => () =>
  agent(
`You are doing a SKEPTICAL second read of a dance contact sheet to catch a couple change the first pass may have MISSED.

Read the image with the Read tool: ${j.sheet}

Contact sheet for clip ${j.name} (duration ${j.duration}s): thumbnails every 8 seconds, each labelled MM:SS top-left. Social dance (Zouk) under heavy magenta lighting and lens flare.

A first pass judged this clip as ${j.prior} focal couple(s). First-pass note: "${j.prior_notes}"

Your job: independently scrutinise whether the camera actually frames 2+ DIFFERENT couples across the clip. Walk the thumbnails in order and track the focal couple's identity (clothing, hair, body type).
- Report a cut ONLY for a genuinely different couple (different people the camera moves to).
- Do NOT call a cut for partner rotation, spins/dips, flare, or the same couple seen from a different angle.
- If it really is one couple (or, for a DJ/non-couple clip, zero couples) throughout, return cuts: [].
- Set disagrees_with_first_pass=true if you found a couple change the first pass missed (or vice versa).

Return n_couples, cuts (seconds, where each new couple begins), disagrees_with_first_pass, confidence, and notes describing each couple.`,
    { label: j.name, phase: 'Miss sweep', schema: SCHEMA }
  ).then((r) => (r ? { clip: j.name, duration: j.duration, prior: j.prior, ...r } : { clip: j.name, error: true }))
))

return results.filter(Boolean)
