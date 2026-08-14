"""Segment merging.

`_merge_short` is what stops the judge's boundaries from producing unusable shards. It
runs on the output of every engine, so its edge cases (a short segment at the head, a
run of consecutive shorts) are the ones that reach an editor as one-frame clips.
"""
from __future__ import annotations

import itertools

from focalcut.segment import Segment, _merge_short


def seg(start, end, ids=None):
    return Segment(start, end, ids or [])


def bounds(segs):
    return [(s.start, s.end) for s in segs]


class TestMergeShort:
    def test_leaves_long_segments_untouched(self):
        segs = [seg(0, 10), seg(10, 20), seg(20, 30)]
        assert bounds(_merge_short(segs, 4.0)) == [(0, 10), (10, 20), (20, 30)]

    def test_folds_a_short_segment_into_its_predecessor(self):
        segs = [seg(0, 10), seg(10, 11), seg(11, 20)]
        assert bounds(_merge_short(segs, 4.0)) == [(0, 11), (11, 20)]

    def test_folds_a_short_leading_segment_forward(self):
        # the head has no predecessor to fold into, so it must absorb its successor
        segs = [seg(0, 1), seg(1, 10), seg(10, 20)]
        assert bounds(_merge_short(segs, 4.0)) == [(0, 10), (10, 20)]

    def test_collapses_a_run_of_consecutive_shorts(self):
        segs = [seg(0, 10), seg(10, 11), seg(11, 12), seg(12, 13), seg(13, 20)]
        assert bounds(_merge_short(segs, 4.0)) == [(0, 13), (13, 20)]

    def test_output_stays_contiguous_and_covers_the_input_span(self):
        segs = [seg(0, 1), seg(1, 9), seg(9, 9.5), seg(9.5, 30)]
        out = _merge_short(segs, 4.0)
        assert out[0].start == 0
        assert out[-1].end == 30
        for a, b in itertools.pairwise(out):
            assert a.end == b.start

    def test_merging_unions_the_object_ids(self):
        segs = [seg(0, 10, [1, 2]), seg(10, 11, [2, 3]), seg(11, 20, [4])]
        assert _merge_short(segs, 4.0)[0].obj_ids == [1, 2, 3]

    def test_empty_input_is_returned_as_is(self):
        assert _merge_short([], 4.0) == []

    def test_a_single_short_segment_survives(self):
        # nothing to merge with; dropping it would lose the clip entirely
        assert bounds(_merge_short([seg(0, 2)], 4.0)) == [(0, 2)]


class TestSegmentDuration:
    def test_duration_is_end_minus_start(self):
        assert seg(2.5, 11.0).duration == 8.5
