# Code for "Minimising missed and false alarms: a vehicle spacing based approach to conflict detection"

## Abstract
Safety is the cornerstone of L2+ autonomous driving and one of the fundamental tasks is forward collision warning that detects potential rear-end collisions. Potential collisions are also known as conflicts, which have long been indicated using Time-to-Collision with a critical threshold to distinguish safe and unsafe situations. Such indication, however, focuses on a single scenario and cannot cope with dynamic traffic environments. For example, TTC-based crash warning frequently misses potential collisions in congested traffic, and issues false alarms during lane-changing or parking. Aiming to minimise missed and false alarms in conflict detection, this study proposes a more reliable approach based on vehicle spacing patterns. To test this approach, we use both synthetic and real-world conflict data. Our experiments show that the proposed approach outperforms single-threshold TTC unless conflicts happened in the exact way that TTC is defined, which is rarely true. When conflicts are heterogeneous and when the information of conflict situation is incompletely known, as is the case with real-world conflicts, our approach can achieve less missed and false detection. This study offers a new perspective for conflict detection, and also a general framework allowing for further elaboration to minimise missed and false alarms. Less missed alarms will contribute to fewer accidents, meanwhile, fewer false alarms will promote people's trust in collision avoidance systems. We thus expect this study to contribute to safer and more trustworthy autonomous driving.

## Package requirements
`jupyter notebook`, `numpy`, `pandas`, `tqdm`, `glob`, `matplotlib`, `scipy`

## In order to repeat the experiments:

- __Step 0 Download used data__
    - Step 0.1 Apply for the dataset [CitySim](https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset), download and put the subset of `FreewayB` in the folder `./localdata/rawdata`.
    - Step 0.2 We have processed and saved the 100Car NDS data in the folder `./localdata/inputdata/`. The readers are still encouraged to explore the raw data with the code in the [repository](https://github.com/Yiru-Jiao/Reconstruct100CarNDSData) if interested.

- __Step 1 Preprocess data__
    - Step 1.1 Run `./Pre-processing/FreewayB_preprocessing.py` to preprocess the CitySim FreewayB data.
    - Step 1.2 Run `./Pre-processing/HundredCar_preprocessing.py` to preprocess the 100Car NDS data.

- __Step 2 Run the experiments__
    - Step 2.1 Run `./ConflictDetection/Sampling.py` to determine conflicts and sample data for spacing inferences.
    - Step 2.2 Run `./ConflictDetection/Computing.py` to compute pma and pfa at each time moment.

- __Step 3 Produce and visualise results__
    - Step 3.1 Use `./ResultsVisualisation/IEEE IV.ipynb` to give results and visualise them for method validation.

## Citation

