export const meta = {
  name: 'focalcut-library-sheets',
  description: 'Read 50 Zouk dance contact sheets in parallel; each agent returns the focal-couple cut points for one clip',
  phases: [{ title: 'Judge sheets', detail: 'one vision agent per contact sheet' }],
}

const SCHEMA = {
  type: 'object',
  properties: {
    n_couples: { type: 'integer', description: 'number of distinct focal couples the camera frames across the clip' },
    cuts: { type: 'array', items: { type: 'number' }, description: 'interior cut points in seconds where a new couple segment begins' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    notes: { type: 'string', description: 'brief description of each couple by clothing/hair so a human can verify' },
  },
  required: ['n_couples', 'cuts', 'confidence', 'notes'],
}

phase('Judge sheets')
const items = typeof args === 'string' ? JSON.parse(args) : args

const results = await parallel(items.map((it) => () =>
  agent(
`You are judging a video contact sheet to find where the camera's FOCAL COUPLE changes.

Read the image at this path (use the Read tool): ${it.sheet}

It is a contact sheet for clip ${it.name} (duration ${it.duration}s). Thumbnails are sampled every 8 seconds and each is labelled MM:SS in the top-left corner.

This is social dance (Zouk) footage under heavy magenta/purple stage lighting with frequent lens flare. The camera follows ONE couple at a time and occasionally pans to a different couple.

Task: identify the timestamps (in SECONDS) where a NEW couple's segment begins — the interior cut points.
- Judge identity by clothing, hair, body type, and recurrence across thumbnails.
- COAST through flare and lighting changes — a washed-out or bright frame is NOT a couple change.
- This is social dancing, so a person may rotate partners. Only call a cut when the camera is clearly framing a genuinely DIFFERENT couple, not a subtle partner swap in the same framing. Be conservative: prefer fewer, confident cuts.
- If it is one couple throughout, return cuts: [] (one segment).
- For a cut, use roughly the midpoint between the last old-couple thumbnail time and the first new-couple thumbnail time (convert MM:SS to seconds).

Return n_couples, cuts (seconds), confidence (high/medium/low by how legible the sheet is), and notes (describe each couple by clothing/hair).`,
    { label: it.name, phase: 'Judge sheets', schema: SCHEMA }
  ).then((r) => (r ? { clip: it.name, duration: it.duration, ...r } : { clip: it.name, duration: it.duration, error: 'no result' }))
))

return results.filter(Boolean)
