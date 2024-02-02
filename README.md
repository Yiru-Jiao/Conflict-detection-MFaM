# Code for "Minimising missed and false alarms: a vehicle spacing based approach to conflict detection"

This work is being submitted to the 35th IEEE Intelligent Vehicles Symposium. The approach this paper is proposing is still under development and currently not mature enough for general applications. Hopefully I'll make it a success and soon open-source a more complete approach :)

## Abstract
Safety is the cornerstone of L2+ autonomous driving and one of the fundamental tasks is the detection of potential collisions between vehicles. Potential collisions are also called conflicts, which have long been detected using surrogate safety measures such as Time-to-Collision with a critical threshold to distinguish safe and unsafe situations. Such indication, however, focuses on a single scenario and cannot cope with dynamic traffic environments. For example, TTC-based crash warning frequently misses potential collisions in congested traffic, and issues false alarms during lane-changing or parking. Aiming to minimise missed and false alarms in conflict detection, this study proposes a data-driven approach based on vehicle spacing patterns. To test this approach, we use synthetic data by defining different types of conflicts in various conditions. Experiments show that our approach outperforms threshold-based Time-to-Collision unless real conflicts happen in the same way that TTC is defined. The proposed approach can achieve less missed and false detection when conflicts are heterogeneous and when the information of conflict situations is incompletely known. This approach offers a new perspective for conflict detection, and also a general framework allowing for further elaboration to minimise missed and false alarms. Less missed alarms will contribute to fewer accidents, meanwhile, fewer false alarms will promote people's trust in collision avoidance systems. We thus expect this study to contribute to safer autonomous driving.

## Package requirements
`jupyter notebook`, `numpy`, `pandas`, `tqdm`, `matplotlib`, `scipy`, `joblib`

## In order to repeat the experiments:

__Step 0. Preparation__

Create a conda environment for repeating the experiments. Install the required packages as listed above.

Clone this repository, then either 1) create/define a folder for data saving and copy the subfolders in "Data path example"; or 2) use the folder "Data path example" directly.

__Step 1. Download and save data__

Download the trajectory data of Lyft from <https://github.com/RomainLITUD/Car-Following-Dataset-HV-vs-AV> and save them in the folder "Data path example/InputData/Lyft/"; download processed data of Waymo from <https://data.mendeley.com/datasets/wfn2c3437n/2> and save it (`all_seg_paired_cf_trj_final_with_large_vehicle.csv`) in the folder "Data path example/InputData/Waymo/".

__Step 2. Preprocessing__ 


