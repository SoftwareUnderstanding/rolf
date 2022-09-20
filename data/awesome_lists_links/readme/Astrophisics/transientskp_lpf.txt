# Live Pulse Finder

This repository contains a method for detecting transient astronomical events in realtime.

## Installation
#### Requirements:
- A python3 (>3.4) installation with working pip3 and python3-venv.

#### 1. Clone the repository.
```
git clone git@github.com:transientskp/lpf.git
```

#### 2. Run Installation
Installation script is provided in `activate.sh` which you can run as follows:
```
cd lpf/
. activate.sh
```
Otherwise, check the `requirements.txt` file.

---
*In the following, you have to setup a configuration file. Consider copying one of the ones provided in `examples/` and editing it to your needs. We use `examples/aartfaac12.yml` in the guidelines.*

## Transient Simulation
To train the neural network for inference, we first build a dataset.
1. Create a parameter configuration file accustomed to your telescope. See the `examples` folder for inspiration. 
2. Run the `lpf/simulation/scripts/transients.py` script with as argument the path to your configuration file. E.g., 
```
python lpf/simulation/scripts/transients.py examples/aartfaac12.yml
```
3. Once the simulation is finished, some example PNGs will be given in the output folder that you provided in the configuration file. Make sure they look satisfactory.

## Noise Extraction (Optional)
This extracts background noise for the dynamic spectra. If skipped, you'll use Gaussian noise.
1. Specify correct parameters in the noise extraction section of your configuration file. 
2. Run the noise extractor: 
```
python lpf/simulation/scripts/extract_noise.py examples/aartfaac12.yml
```

## Neural Network Training
1. Edit the neural network section of your configuration file to your needs.
2. Run
```
python lpf/_nn/scripts/train.py examples/aartfaac12.yml
```
3. Wait until it's converged.

## Run LPF
1. Edit your configuration file to your needs. 
2. Run 
```
python lpf/main.py examples/aartfaac12.yml
```
3. The parameters of analyzed transients will be output to a `.csv` in the specified output folder. This can be opened for analysis. The `.npy` file in the output folder constains all the dynamic spectra.

## Analyse Results
1. The `.csv` file with the inferred parameters is in the output folder you specified. You can use `pandas` to inspect it and filter it for interesting bursts. An example `.ipynb` file is given in `lpf/analysis/result_analysis.ipynb`.
2. Also: in the provided output folder a catalog video is saved to show the source-detection pipeline and an example of the estimated background and variability maps are saved.
