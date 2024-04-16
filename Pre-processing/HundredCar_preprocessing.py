'''
This file is used to preprocess the HundredCar dataset.
'''

import numpy as np
import pandas as pd
from tqdm import tqdm

path_output = './Conflict-detection-MFaM/localdata/outputdata/'
path_processed = '../Process_100Car/ProcessedData/'
path_cleaned = '../Process_100Car/CleanedData/'

manualSeed = 131


class cf_extractor():
    def __init__(self, data, meta):
        super().__init__()
        data_ego, data_sur = data
        self.trip_ids = data_sur['trip_id'].unique()
        self.data_ego = self.initialze_data(data_ego)
        self.data_sur = self.initialze_data(data_sur)
        self.meta = meta
        self.cfdata = self.organise_data()

    
    # Initial formation of data
    def initialze_data(self, data):
        data = data.sort_values(['trip_id','time']).set_index('trip_id')
        data = data.rename(columns={'x_ekf':'x','y_ekf':'y','psi_ekf':'psi','v_ekf':'speed'})
        data['hx'] = np.cos(data['psi'])
        data['hy'] = np.sin(data['psi'])
        data['vx'] = data['speed']*data['hx']
        data['vy'] = data['speed']*data['hy']
        return data


    # Organize data
    def organise_data(self,):
        cfdata = []

        for trip_id in tqdm(self.trip_ids):
            df_ego = self.data_ego.loc[trip_id].copy()
            df_sur = self.data_sur.loc[trip_id].copy()
            merged = df_ego.merge(df_sur, on='time', suffixes=('_ego', '_sur'))
            forward = merged[merged['forward'].astype(bool)].groupby('time')['range'].idxmin()
            forward = merged.loc[forward][['time','target_id']]
            rearward = merged[~merged['forward'].astype(bool)].groupby('time')['range'].idxmin()
            rearward = merged.loc[rearward][['time','target_id']]

            df_sur = df_sur.reset_index()
            merged = df_ego.merge(df_sur, on='time', suffixes=('_i', '_j'), how='inner')
            merged = merged.set_index(['time','target_id'])
            # For forward conflicts
            if ('lead' in self.meta.loc[trip_id]['target']) and (len(forward['target_id'].unique())==1):
                df = merged.loc[pd.MultiIndex.from_frame(forward)].reset_index()
                df['frame_id'] = (df['time']*100).astype(int)
                df['s'] = np.sqrt((df['x_j']-df['x_i'])**2 + (df['y_j']-df['y_i'])**2)
                df['v'] = df['speed_i'] - df['speed_j']
                df['speed'] = df['speed_i']
            # For rearward conflicts
            elif ('follow' in self.meta.loc[trip_id]['target']) and (len(rearward['target_id'].unique())==1):
                df = merged.loc[pd.MultiIndex.from_frame(rearward)].reset_index()
                df['frame_id'] = (df['time']*100).astype(int)
                df['s'] = np.sqrt((df['x_i']-df['x_j'])**2 + (df['y_i']-df['y_j'])**2)
                df['v'] = df['speed_j'] - df['speed_i']
                df['speed'] = df['speed_j']

            if len(df)>0:
                cfdata.append(df[['trip_id','frame_id','s','v','speed','event']].rename(columns={'trip_id':'track_id','event':'conflict'}))
        
        return pd.concat(cfdata).reset_index(drop=True)


# Load data
print('Loading data...')
data_ego = pd.read_hdf(path_processed + 'HundredCar_NearCrash_Ego.h5', key='data')
data_sur = pd.read_hdf(path_processed + 'HundredCar_NearCrash_Surrounding.h5', key='data')
data_sur = data_sur.drop(columns=['x','y'])
meta = pd.read_csv(path_cleaned + 'HundredCar_metadata_NearCrashEvent.csv').set_index('webfileid')
meta = meta.loc[(data_ego['trip_id'].unique())]

counted_target = ['lead vehicle','following vehicle']
meta = meta[meta['target'].isin(counted_target)] # remain car-following scenarios only

data_ego = data_ego[data_ego['trip_id'].isin(meta.index)]
data_sur = data_sur[data_sur['trip_id'].isin(meta.index)]
print(f'There are {data_ego['trip_id'].nunique()} trips for car-following extraction\n')

# Extract car-following conflict data and save
cfe = cf_extractor([data_ego, data_sur], meta)
cfe.cfdata.to_hdf(path_output + 'HundredCar_CFData.h5', key='data', mode='w')
print(cfe.cfdata.head())
