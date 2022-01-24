# Comparison of DNN-based object detectors

## Structure

- `converters` contains scripts to convert groundtruth of the traffic
  videos (text format) to the format of PASCAL VOC.

- `readers` contains scripts to read detections and groundtruth
  represented in text format.

- `utilities` contains scripts to estimate detection/tracking quality
  and to perform visual inspection:

  - `average_precision.py` to calculate average precision (AP)
    and draw precision-recall curve.
  - `true_positive_rate.py` to compute true positive rate (TPR).
  - `false_detection_rate.py` to calculate false detection rate (FDR).
  - `false_positives_per_frame.py` to compute number of false
    positives per frame/image.
  - `play_bboxes.py` to show groundtruth and detections simultaneously.
  - `play_tracks.py` to show constructed tracks.
  - auxiliary scripts required for AP, TPR and FDR computation.

- `auxiliary/ssd-detector` contains scripts to install and to execute SSD.

- `vehicle-detector` is a video-based vehicle detection system.

  - `detector` is a package containing implementation of detection methods.
  - `tracker` is a package containing implementation of tracking methods.
  - `video-detector` is a package containing implementation of video-based
    detection algorithms (provide detection and tracking).
	`video_analyzer.py` is a starting point.
  - `tests` is a set of learning tests.

## References

1. Liu W., Anguelov D., Erhan D., Szegedy C., Reed S., Fu Ch.-Y.,
   Berg A.C. SSD: Single Shot MultiBox Detector. 2016.
   [https://arxiv.org/abs/1512.02325].
1. Sources of SSD [https://github.com/weiliu89/caffe/tree/ssd].
