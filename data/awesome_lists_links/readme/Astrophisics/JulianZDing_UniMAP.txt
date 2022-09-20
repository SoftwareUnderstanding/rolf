# `UniMAP`: Unicorn Multi-window Anomaly Detection Pipeline

A data analysis pipeline that leverages the Temporal Outlier Factor (TOF) method to find anomalies in LVC data.

## Setup

Clone this repository and navigate to the root directory. Install the required packages using

```python -m pip install -r requirements.txt```

## Basic Usage

Make a script or Jupyter notebook in the root directory and import the pipeline function:

```python
from tof_data_analysis import tof_detections_in_open_data
```

The pipeline requires a target detector (ex. `H1`) and a start and stop GPS time describing a time interval to analyze:

```python
anomalies = tof_detections_in_open_data(<detector>, <start time>, <stop time>)
```

The pipeline has three outputs:

1. The function `tof_detections_in_open_data` returns an array of GPS times corresponding to TOF detections.
2. Specifying `plot_summary=True` outputs a long [q-transform](https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries.html#gwpy.timeseries.TimeSeries.q_transform) of the entire data interval and visualizes TOF detections in the time series.
3. Specifying `plot_detection_windows=True` outputs q-transforms of the data windows that triggered TOF detections. Alternatively, specifying `plot_all_windows=True` outputs q-transforms of all the data windows regardless of whether an anomaly was detected or not.

A location to save the output plots can be specified using `save_dir`.

Example summary plot:

![h1_summary_plot](examples/multi_window_H1_1242242596.47_1242242676.47.jpg)

Example window plots:

<img src="examples/H1_1242242356.47_1242242386.47.png" alt="window_plot_1" width="45%"/> 
<img src="examples/H1_1242242260.47_1242242264.47.png" alt="window_plot_2" width="45%"/>

Please consult the docstrings for complete documentation, including additional function wrappers for offline/non-public data.

## Acknowledgements

UniMAP paper: (coming soon)

Original TOF paper: [arXiv:2004.11468](https://arxiv.org/abs/2004.11468)

The code in this repository was implemented by [Julian Ding](julianzding@gmail.com).
