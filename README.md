# Minimising missed and false alarms: a vehicle spacing based approach to conflict detection
This study is presented at the 2024 IEEE Intelligent Vehicles Symposium (IV) and accessible at <https://doi.org/10.1109/IV55156.2024.10588396>.

## Highlights
- Conflict detection involves a trade‐off between missed and false alarms.
- Probabilities of missed and false alarms are estimated from spacing distributions.
- Critical spacing is optimised to minimise missed and false alarms.
- Validation on synthetic and real-world conflicts confirms superior performance.
- Collision warning can be adaptive in varying traffic contexts and driver preferences.

## In order to repeat the experiments:
### Dependencies
`jupyter notebook`, `numpy`, `pandas`, `pytables`, `tqdm`, `glob`, `matplotlib`, `scipy`

### Data
- **Raw data**  
  Apply for the dataset [CitySim](https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset), download and put the subset of `FreewayB` in the folder `./localdata/rawdata`.
- **Test data**  
  We have processed and saved the 100Car NDS data in the folder `./localdata/inputdata/`. The readers are still encouraged to explore the raw data with the code in the [repository](https://github.com/Yiru-Jiao/Reconstruct100CarNDSData) if interested.
- **Results**  
  Resulting data, i.e., a zipped file of the `./localdata/` folder, can be downloaded from <https://doi.org/10.4121/252a79e7-d9ff-4181-a9e4-842ea7845a77>.

### Step-by-step instructions
- __Step 1 Preprocess data__
    - Step 1.1 Run `./Pre-processing/FreewayB_preprocessing.py` to preprocess the CitySim FreewayB data.
    - Step 1.2 Run `./Pre-processing/HundredCar_preprocessing.py` to preprocess the 100Car NDS data.

- __Step 2 Run the experiments__
    - Step 2.1 Run `./ConflictDetection/Sampling.py` to determine conflicts and sample data for spacing inferences.
    - Step 2.2 Run `./ConflictDetection/Computing.py` to compute pma and pfa at each time moment.

- __Step 3 Produce and visualise results__
    - Step 3.1 Use `./ResultsVisualisation/IEEE IV.ipynb` to give results and visualise them for method validation.

## Citation
````latex
@inproceedings{Jiao2024,
  title = {Minimising Missed and False Alarms: A Vehicle Spacing based Approach to Conflict Detection},
  doi = {10.1109/iv55156.2024.10588396},
  booktitle = {2024 IEEE Intelligent Vehicles Symposium (IV)},
  pages={1982-1987},
  author = {Jiao,  Yiru and Calvert,  Simeon C. and van Lint,  Hans},
  year = {2024},
  month = jun,
  address = {Jeju Island, Republic of Korea}
}
````
