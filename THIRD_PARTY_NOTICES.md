# Third-Party Notices

## ByteTrack

MergenVision Phase 2 uses the ByteTrack multi-object tracking algorithm in the
`backend/native/tracking/` C++ library and in the `mvfacetracker` GStreamer
plugin.

Upstream project:
- Paper: https://arxiv.org/abs/2110.06864
- Repository: https://github.com/FoundationVision/ByteTrack
- Pinned reference commit: `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`
- License: https://github.com/FoundationVision/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/LICENSE

ByteTrack is licensed under the MIT License:

```
MIT License

Copyright (c) 2021 ByteTrack

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Adaptation Notes

The MergenVision implementation is an independent adaptation of the ByteTrack
state-machine and association ideas; it is not a direct copy of the upstream
demo application code. The following production-contract adaptations were made:

- Replaced OpenCV `cv::Rect` with a continuous `xyxy` `RectF` type and removed
  the `+1` pixel convention used by the upstream C++ example.
- Replaced the fixed `dt=1` assumption with a PTS-aware Kalman filter where
  `dt = delta_pts / nominal_frame_period_ns`.
- Removed the process-global static track ID generator in favor of per-source
  deterministic tracklet ID assignment.
- Removed hardcoded thresholds; all thresholds are declared in `TrackerConfig`.
- Removed `exit()` / `system("pause")` style error handling; errors are reported
  through structured `UpdateResult`, GStreamer bus messages, and sanitized logs.
- Removed single-source demo assumptions by introducing `MultiSourceTracker`.
- Added explicit sampling-gap and timestamp-gap handling; excessive gaps
  terminate active tracklets.
- Added optional face-embedding appearance gating in addition to IoU/motion
  association.
- No upstream C++ demo application code is copied into this repository.

See `docs/implementation/REFERENCE_DECISION_LOG.md` for the reference-first
engineering decision record.
