# Resulting dataset readme

This dataset deposited at https://doi.org/10.4121/252a79e7-d9ff-4181-a9e4-842ea7845a77 contains the processed and analysed outputs. The dataset is organised into several subfolders containing preprocessed files, sampling outputs, spacing parameters, and additional supporting files for conflict detection experiments.

## Data Organization
- `./localdata/rawdata/`
  Empty folder, intended to store raw data that are sourced from the CitySim dataset.  
- `./localdata/inputdata/`  
  Processed input files. 
  - `./localdata/inputdata/100Car/` contains the preprocessed 100Car NDS data.
  - `./localdata/inputdata/FreewayB/` is empty but intended to store the preprocessed FreewayB data from the CitySim dataset.
- `./localdata/outputdata/`  
  Processed data in HDF5 format. 
- `./localdata/samples/`  
  Samples in HDF5 format. These data are used for spacing inference and conflict detection.
- `./localdata/spacing/`  
  Output CSV files of spacing inference results
- `./localdata/results_100Car.csv` and `./localdata/results_FreewayB.csv`  
  Summary results of the conflict detection experiments for the 100Car and FreewayB datasets, respectively.

## Data Variables and Column Headings

- **frame_id**: Unique frame identifier computed from the timestamp.
- **track_id / trip_id**: Identifier for each vehicle or trip.
- **s**: Vehicle spacing (m), the Euclidean distance between the ego vehicle and surrounding vehicle.
- **v**: Relative speed (m/s) calculated as the difference between speeds of vehicles.
- **speed**: Speed of the ego vehicle.
- **event/conflict**: Boolean flag indicating if a conflict is observed (different naming conventions are used in various stages).
- **round_v**: A quantized form of vehicle speed used for grouping data in inference routines.
- **ttc**: Time-to-collision, calculated as spacing divided by relative speed.
- **precedingId**: Identifier for the vehicle in front (used in filtering for lane consistency).
  
In the spacing parameter CSVs, additional symbols include:
- **alpha**: Weighting factor applied in the MFaM method for balancing missed and false alarms.
- **smax**, **cum_smax**, **c**: Variables used to derive probabilities that estimate the likelihood of missed and false alarms based on spacing distributions.

## Usage Notes
- Data files are stored in standard formats (HDF5 or CSV) and can be loaded using Python packages such as pandas (e.g., `pd.read_hdf` or `pd.read_csv`).
- This dataset is organised to support conflict detection experiments, particularly focusing on the inference of vehicle spacing and the evaluation of missed and false alarms.
- For replication details, experiment code is open-sourced at https://github.com/Yiru-Jiao/Conflict-detection-MFaM