export const meta = {
  name: 'focalcut-ultra-boundary',
  description: 'Pin each confirmed cut to ~0.5s precision from an ultra-fine 0.5s-spaced contact sheet',
  phases: [{ title: 'Pin boundary', detail: 'one agent per cut, 0.5s precision' }],
}

const SCHEMA = {
  type: 'object',
  properties: {
    precise_time: { type: 'number', description: 'precise boundary in seconds (to ~0.5s) where the new couple is first clearly framed' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    notes: { type: 'string' },
  },
  required: ['precise_time', 'confidence', 'notes'],
}

phase('Pin boundary')
const jobs = typeof args === 'string' ? JSON.parse(args) : args

const results = await parallel(jobs.map((j) => () =>
  agent(
`Pin a known cut between two camera focal couples to ~0.5 second precision.

Read the image with the Read tool: ${j.sheet}

It is an ULTRA-FINE contact sheet: thumbnails every 0.5 seconds across the window ${JSON.stringify(j.window)}s of clip ${j.clip}, each labelled MM:SS in the top-left. (Two thumbnails per second; the labels repeat the same MM:SS for the two halves of a second.) A verified couple change happens near ${j.cut}s.

Walk the thumbnails left to right. Find the EXACT moment the camera's focal couple becomes the NEW couple — the last frame still showing the old couple, and the first frame clearly showing the new one. Return precise_time in seconds (you may use halves, e.g. 14.5), placed at that transition.

If the transition is a fast pan, place it at the first frame where the new couple is clearly the framed subject. Return precise_time, confidence, and a short note on what flips between the two adjacent frames.`,
    { label: `${j.clip}@${j.cut}`, phase: 'Pin boundary', schema: SCHEMA }
  ).then((r) => (r ? { clip: j.clip, old_cut: j.cut, ...r } : { clip: j.clip, old_cut: j.cut, error: true }))
))

return results.filter(Boolean)
