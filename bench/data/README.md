# Benchmark data

Drop one ground-truth JSON per source clip here. Hand-label the seconds at which the
**focal dancer/couple changes** (watch the clip, note each handoff):

```json
{
  "video": "C1776_1_1.mp4",
  "duration": 168.5,
  "cuts": [12.4, 31.0, 58.7, 95.2]
}
```

Then compare any predictor against it:

```bash
# focalcut prediction
python -m focalcut.cli cut <video> -o /tmp/pred.xml --dump-json /tmp/focalcut_pred.json
python -m bench.eval --gt bench/data/C1776_1_1.json --pred /tmp/focalcut_pred.json

# frame-diff baseline (the strawman to beat)
python -m bench.baselines.framediff <video> > /tmp/framediff_pred.json
python -m bench.eval --gt bench/data/C1776_1_1.json --pred /tmp/framediff_pred.json
```

The headline number is `manual_corrections_per_min` (false splits + false merges per
minute). Driving it toward zero is the "we don't want to recut" goal. focalcut should
win on `false_splits_per_min` specifically, because object permanence stops it cutting
when a dancer turns away or another couple crosses frame.
