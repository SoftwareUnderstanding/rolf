# VOLKS2
### VLBI Observation for single pulse Localization Keen Searcher, 2nd release

“VOLKS”为人民大众，意为发动群众，群策群力，在各种VLBI观测中寻找暂现源。

## Introduction
This is the second release of the VOLKS pipeline. Significant updates have been made since its first release. 

VOLK2 is designed for single pulse search and localization in regular VLBI observations as well as single pulse detections from known sources in dedicated observations. In VOLKS2, the search and localization are two independent steps. The search step takes the idea of geodetic VLBI post processing, which fully utilizes the cross spectrum fringe phase information to maximize the signal power. Compared with auto spectrum based method, it is able to extract single pulses from highly RFI contaminated data. The localization uses the geodetic VLBI solving methods, which derives the single pulse location by solving a set of linear equations given the relation between the residual delay and the offset to a priori position. 

## Highlights

- Whole FoV transient search in regular VLBI observations as commensal task;
- Single pulse detection from known sources in dedicated observation;
- No requirement for RFI excision;
- High accuracy single pulse localization with VLBI method.

## Main functions

- Flexible configurations: arbitrary selection of IFs, polarizations, baselines and DM values; easy adjustment of control parameters;
- Full FoV (Field of View) single pulse search;
- Geodetic VLBI solving for single pulse localization;
- DM search;
- GPU support (PyTorch and CuPy);
- Full parellelization (with `mpi4py`), optimized for multi nodes GPU clusters.

The whole pipeline has been tested with EVN observation EL060.

## Acknowledgement
The development of this pipeline is supported by the National Science Fundation of China (No. 11903067).  

**Note**: If you make use of VOLKS2 pipeline in your publication, we require that you quote the pipeline web address https://github.com/liulei/volks2 and cite the following papers:

- `Liu, L., Tong, F., Zheng, W., Zhang, J. & Tong, L. 2018, AJ, 155, 98`, which describes the cross spectrum based single pulse search method.
- `Liu, L., Zheng, W., Yan, Z. & Zhang, J. 2018, Research in Astronomy and Astrophysics, 18, 069`, which compares the cross spectrum based method and the auto based spectrum method for single pulse serach in VLBI observation.
- `Liu, L., Jiang, W., Zheng, W., et al. 2019, AJ, 157, 138`, which describes the geodetic solving based single pulse localzation method.
- `Liu, L., Xu, Z., Yan, Z., et al. 2021, AJ, 162, 159, arXiv:2108.05602`, which introduces the VOLKS2 pipeline and the EL060 observaion.

Please do not hesitate to contact me (E-mail: liulei@shao.ac.cn, WeChat: thirtyliu) if you have any question.

## Platform requirement

- Linux, MacOS, WSL.
- gcc/clang, gfortran, Python3, NumPy, ctypes, matplotlib (required only for plotting), PyTorch or CuPy (for GPU).

## Run

I have tried my best to make the pipeline easy to understand and use. However, due to the complexity of VLBI data processing, it still requires some efforts to have it run and get the final result. Since the Python code is self-explanatory, it is stronly suggested that users read the code and figure out how it works. I will give short explanation (**Description**, **Input** and **Output**) for each step.

### Correlation

VOLKS2 conducts SP search and localization with DiFX correlation result. Some settings in the correlation process are suggested:

- Clock: clock rate should be adjusted such that fringe rate (clock rate multiplied with sky frequency) within 10 mHz; no special requirement for clock offset, as small as possible. 

- FFT size: to cover the whole FoV, the minimum FFT size is specified in equation (12) of Liu et al. (2018). E.g., for 32 meter telescope, 3000 km baseline, L band, the recommended FFT size is 1024.

### Prepare configuration: `utils.py`
**Description**

- All programs in the pipeline will first call `utils.gen_cfg()` to obtain the configuration. In this way modification of source code in any other programs is avoided. One need to set the specific task in `gen_cfg()` and then prepare the corresponding configuration class. The detailed explanation of each term is explained in `gen_cfg_el060()` as an example.
 
**Input**:

- Once `gen_cfg()` is invoked, `load_config()` will read and parse the `.input` and `.calc` files of the corresponding scan, and generate the configuraiton class.

- When the `Config` class is initialized, it will check if `libio.so` exists. If this is the case, the C version `rec2buf()` in this lib will be used to speedup data loading.
- 
### Initial calibration: `gen_cal_initial.py`
**Description**:

- This program will conduct fringe fitting for each baseline, polarization and IF of the calibration scan, and derive the delay and initial phase for each IF and baseline. **One needs to specify the scan no of the calibration source and time range at the beginning of `main_cal()`.**

**Input**:

- `scan_no` and time range (`t1` and `t2`) of calibration scan. 
- Visibility file (`.swin`) of calibration scan.

**Output**:

- `cal_initial.npy`: initial phase and delay for each baseline, polarization and IF of calibration source.

### Fringe fitting: `pfit.py`
**Description**:

- This is the main routine of the VOLKS2 pipeline. It can be divided into 6 steps. The detailed explanation is given in Sec. 2.4.2 of the paper. 

- To conduct SP search, the integraiton time (accumulation period, AP) is usually very small, e.g. 1.024 ms. Then several these APs are combined together according to pre-set window sizes, e.g., 2 ms, 4 ms, 8 ms and 16 ms. This is determined by the `nsum` parameter. One may find the snum setting in the `utils.py` file: `cfg.nsum = [2, 4, 8, 16]`. 

- In the search mode, the DM of the SP is unknown, therefore one has to setup a DM list, e.g., from 50 to 1000 with an interval of 50: `cfg.dms  =   np.arange(50., 1000., 50.)`. For single pulse detection with known source, DM search is not required. Set one DM value: `cfg.dms = [26.833]` (for PSR J0332+5434).

- GPU acceleration. VOLKS2 provides GPU support with Torch and CuPy frameworks. `main()` function in `pfit.py` gives the demo of backend selection. The default is `numpy`, which selects CPU. `torch` and `cupy` select GPU. According to my test, the performance of these two GPU backends are similar. Once GPU are selected. `open_XXX()` will be called for GPU initilization. The calculation procs will be mapped to GPU devices. E.g., proc 1 for dev 0, proc 2 for dev 1, etc. To reduce the kernel launch and data upload overhead of GPU devices,  the data size should be as large as possible. E.g., `t_seg` is set to 4.0 s to 8.0 s for GPU backends. 

- For parallelization, data is divided into small pieces, e.g., 0.5 second per seg for numpy backend (CPU), and then sent to the calculation process. The fitting process is invoked with `mpi` command:

    `mpirun -np nproc -host hostfile ./pfit.py`
    
  At least 2 procs are required. Depends on the size of the vis data, the fringe fitting for each scan might take tens of minutes. The fitting result of each seg will be saved in the disk in the name `NoXXXX/segXXXX.npy` (see **output** for an explanation of the file structure), so as to prevent from being lost even if the execution of `pfit.py` is interupted unexpectly. Next time `pfit.py` is invoked, those already have been processed will be skipped automatically. 
  
- Once fringe fitting process is finished, rerun `./pfit.py`:

    `./pfit.py`
    
The program will realize only one proc is invoked and therefore call `combine_seg()`. This will combine all seg files in the scan_no to a single `NoXXXX/fitdump.npy` file.

**Input**:

- `cal_initial.npy`: initial calibration file
- Visibility file (`.swin`) of target scan.

**Output**:

- `NoXXXX/segXXXX.npy`: fringe fitting result of one baseline in the give time range. Note that every such kind of these seg files contains fitting result of **only one** baseline. 
- `NoXXXX/fitdump.npy`: this file contains fitting result of one scan. It is organized as a series of dicts of different level:
 
    `d = np.load('NoXXXX/fitdump.npy', allow_pickle=True).item()`
    `arr = d[bl_id][dm][nsum]`

  `arr` is a array of `dtype_fit`. Note that in most of cases only `mag` in the dtype arrray is meaningful. 

### Single baseline windows filtering: `winmatch.py`

**Description**:

- Conduct window filtering from multiple `nsum`. For each dm, first pick up SP according to threshold from fitdump file of previous step. Then count the number of windows (nsums) in which they are detected. 
- Single pulses are output if they are detected in at least `ne_min` windows. `ne_min` is set in configuration class. 

**Input**:

- `NoXXXX/fitdump.npy`
- Specify filtering parameters, including threshold to pick up single pulses (`cfg.sigma_winmatch`), minimum number of detected windows (`cfg.ne_min_winmatch`).

**Output**:

- `NoXXXX/dmxxx.xxx/blxxx.nsum`, records time, width of each single pulse candidate after multiple window size (nsum) filtering.

### Multiple baselines cross matching: `crossmatch.py`

**Description**:

- Cross matching SPs detected on multiple baselines for each dm.
- A single pulse is output if it is detected on at least `nbl_min` baselines.

**Input**:

- `NoXXXX/dmxxx.xxx/blxxx.nsum` files in previous step.
- Specify `cfg.nbl_min_crossmatch`.

**Output**:

- `NoXXXX_dmxxx.xxx_sss.ssssss.sp`, generated for every individual SP. Each line corresponds to detection info from one baseline. Columns are:
    - col 0: baseline id
    - col 1: start AP id
    - col 2: number of AP (nsum)
    - col 3: SP detection time (since scan start)
    - col 4: SP duration, in s
    - col 5: baseline residual delay, in ns
    - col 6: SP power, in units of sigma
    - col 7: baseline name, comments only, will not be read

Note: Due to the slightly different implementation of fitting algorithm, if GPU (with PyTorch or CuPy) is used, mbd will be zero. If available, this quantity can be used for solving. However `sp_fit.py` will conduct fine fitting and yield higher accuracy, and is therefore prefered and used as the default quantity for solving. 

### Fine fit for individual SP: `sp_fit.py`

**Description**:

- Read `.sp` file, conduct fine fit for the corresponding baselines,  derive delay and SNR.

- The program will call `utils.read_sp()`, which converts txt file to dict. Fitting result are inserted to the dict and saved.

**Input**:

- `.sp` file

**Output**:

- `.sp.npy` file, the program will convert the `.sp` (txt format) file to dict, insert fitting result (initial calibrated and dedispersed vis data, tau, SNR, tau_sig) and save the dict as `.npy` format.

### Calculation partials: `sp_calc.py`

**Description**:

- The pipeline derives SP location by solving a set of linear equations given the relation (partial) between the residual delay and the offset to a priori position. The partial calculation is conduct with `calc9.1`. 

- `calc9.1` is written in FORTRAN. I have add variable `PUTDSTRP`, `PARTIAL` to `cdrvr.f`, `calcmodl2.f` and `cpart.i`, so as to obtain partial derivatives of delay with Ra and Dec. 

- A wrapper is developed to provide IO in C interface. The whole program is packaged to a lib: `libcalc_cwrapper.so`.
 
- `sp_calc.py` interacts with the wrapper via `ctypes` package and obtain the partial derivatives. 

- For setup：
    - Complie library in your own platform:
    
      `cd calc9.1`
      
      `make`
      
      This will generate `libcalc_cwrapper.so`
      
    - Run command 
    
      `source environment` 
      
      This will tell `calc9.1` where to find `JPLEPH` and `Horizons.lis`. Please keep other settings in `environment` unchanged.

**Input**:

- `.sp.npy`: the program will recognize stations of each baseline, and calculate partial at given time.

**Output**:

- `.sp.npy`: add `pd` key in the dict and save.

### Fine calibration: `gen_cal_fine.py`

**Description**:

- VOLKS2 pipeline improves localization accuracy with fine calibration. See Sec. 3.1.2 and Fig. 7 for the remarkable result.  This program conducts fringe fitting with nearby fine calibration source and saves the fitting result (per baseline residual delay of calibration source).

**Input**:

- Fine calibration scan that observes nearby strong radio source with accurate position.

**Output**:

- `cal_fine_NoXXXX.npy`: fringe fitting result (residual delay) per baseline of fine calibration source.


### Localization: `sp_solve.py`

**Description**:

- This program reads the partial derivatives and delay of each baseline of the targets, as well as fine calibration result, solves linear equations about offset to the a priori position (Eq. 2). 

- To guarantee solving accuracy, the program could conduct selection in the data. According to the test, a SNR of above 7, at least 3 baselines (although 2 baselines are already solvable) are prefered.

**Input**:

- `.sp.npy`: `pd`, `tau`, `snr` keys of each baseline are required.
- `cal_fine_NoXXXX.npy`: fine calibration result. One need to specify the nearby calibration source when solving.

### Display: `showspec_all.py`

**Description**:

- This program generate spectrum plot for all baselines and stations at given scan and time range.
