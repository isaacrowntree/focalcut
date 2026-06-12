export const meta = {
  name: 'focalcut-refine-cuts',
  description: 'For each detected cut, read a fine ±8s contact sheet to verify it is a real couple change and pin the precise boundary',
  phases: [{ title: 'Refine cuts', detail: 'one agent per cut window' }],
}

const SCHEMA = {
  type: 'object',
  properties: {
    is_real_cut: { type: 'boolean', description: 'true if a genuinely different couple is framed within the window; false if same couple through flare/spin/partner-rotation' },
    refined_time: { type: 'number', description: 'precise boundary in seconds where the new couple is first clearly framed; if not real, return the approx cut' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    notes: { type: 'string', description: 'what is seen at the boundary' },
  },
  required: ['is_real_cut', 'refined_time', 'confidence', 'notes'],
}

phase('Refine cuts')
const jobs = typeof args === 'string' ? JSON.parse(args) : args

const results = await parallel(jobs.map((j) => () =>
  agent(
`You are precisely locating (or rejecting) a cut between two camera FOCAL COUPLES in a dance video.

Read the image with the Read tool: ${j.sheet}

It is a FINE contact sheet: thumbnails every 2 seconds across the window ${JSON.stringify(j.window)}s of clip ${j.clip}, each labelled MM:SS in the top-left. The coarse pass flagged a possible couple change near ${j.approx_cut}s.

Coarse-pass notes about the couples in this clip:
${j.notes}

Decide three things:
1. is_real_cut: Is there a genuine change to a DIFFERENT couple within this window? Return false if it is actually the SAME couple seen through lens flare, a spin/dip, or a partner rotation within the same framing (these are NOT cuts).
2. refined_time: if real, the precise boundary in SECONDS — the moment the NEW couple is first clearly framed (read the MM:SS labels and convert; you may choose a value between two adjacent thumbnails). If NOT real, set refined_time to ${j.approx_cut}.
3. confidence and notes (briefly describe what you see at the boundary and which two couples are involved).`,
    { label: `${j.clip}@${j.approx_cut}s`, phase: 'Refine cuts', schema: SCHEMA }
  ).then((r) => (r ? { clip: j.clip, approx_cut: j.approx_cut, ...r } : { clip: j.clip, approx_cut: j.approx_cut, error: true }))
))

return results.filter(Boolean)
