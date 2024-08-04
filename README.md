# cyberguard

DPU accelerated mask detection

## Backend

Reading image from `/dev/video0` and pass it through detector.

## Detector

Using `opencv` and `vart` API to perform DPU accelerated inference, the detector will apply an IoU-based
filter on inferred tensors and select the candidate with the highest confidence as the final result.

