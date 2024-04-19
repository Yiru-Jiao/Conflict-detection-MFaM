'''
This file is used to preprocess the HundredCar dataset.
'''

import numpy as np
import pandas as pd
from tqdm import tqdm

path_output = './localdata/outputdata/'
path_processed = '../../Process_100Car/ProcessedData/'
path_cleaned = '../../Process_100Car/CleanedData/'


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
        return data


    # Organize data
    def organise_data(self,):
        cfdata = []

        for trip_id in tqdm(self.trip_ids):
            df_ego = self.data_ego.loc[trip_id].reset_index()
            df_sur = self.data_sur.loc[trip_id]
            merged = df_ego.merge(df_sur, on='time', suffixes=('_ego', '_sur'))
            forward = merged[merged['forward'].astype(bool)].groupby('time')['range'].idxmin()
            forward = merged.loc[forward][['time','target_id']]
            rearward = merged[~merged['forward'].astype(bool)].groupby('time')['range'].idxmin()
            rearward = merged.loc[rearward][['time','target_id']]

            merged = merged.set_index('target_id')
            # For forward conflicts
            unique_forward = forward['target_id'].unique()
            unique_rearward = rearward['target_id'].unique()
            if ('lead' in self.meta.loc[trip_id]['target']) and (len(unique_forward)==1):
                df = merged.loc[unique_forward[0]].reset_index()
                if len(df[df['event'].astype(bool)])>10: # focus on the exact moment of conflict
                    tenth_range = df[df['event'].astype(bool)]['range'].sort_values().iloc[10]
                    df.loc[df['range']>tenth_range, 'event'] = 0
                df['frame_id'] = (df['time']*100).astype(int)
                df['s'] = np.sqrt((df['x_ego']-df['x_sur'])**2 + (df['y_ego']-df['y_sur'])**2)
                df['v'] = df['speed_ego'] - df['speed_sur']
                df['speed'] = df['speed_ego']
            # For rearward conflicts
            elif ('follow' in self.meta.loc[trip_id]['target']) and (len(unique_rearward)==1):
                df = merged.loc[unique_rearward[0]].reset_index()
                if len(df[df['event'].astype(bool)])>10:
                    tenth_range = df[df['event'].astype(bool)]['range'].sort_values().iloc[10]
                    df.loc[df['range']>tenth_range, 'event'] = 0
                df['frame_id'] = (df['time']*100).astype(int)
                df['s'] = np.sqrt((df['x_ego']-df['x_sur'])**2 + (df['y_ego']-df['y_sur'])**2)
                df['v'] = df['speed_sur'] - df['speed_ego']
                df['speed'] = df['speed_sur']
            else:
                continue

            if len(df)>0:
                cfdata.append(df[['trip_id','frame_id','s','v','speed','event']].rename(columns={'trip_id':'track_id','event':'conflict'}))
        
        return pd.concat(cfdata).reset_index(drop=True)


# Load data
print('Loading data...')
cfdata = []
for conflict_type in ['Crash', 'NearCrash']:
    data_ego = pd.read_hdf(path_processed + 'HundredCar_'+conflict_type+'_Ego.h5', key='data')
    data_sur = pd.read_hdf(path_processed + 'HundredCar_'+conflict_type+'_Surrounding.h5', key='data')
    data_sur = data_sur.drop(columns=['x','y'])
    meta = pd.read_csv(path_cleaned + 'HundredCar_metadata_'+conflict_type+'Event.csv').set_index('webfileid')
    meta = meta.loc[(data_ego['trip_id'].unique())]

    counted_target = ['lead vehicle','following vehicle']
    meta = meta[meta['target'].isin(counted_target)] # remain car-following scenarios only

    data_ego = data_ego[data_ego['trip_id'].isin(meta.index)]
    data_sur = data_sur[data_sur['trip_id'].isin(meta.index)]
    print(f'There are {data_ego['trip_id'].nunique()} trips for car-following extraction\n')

    # Extract car-following conflict data and save
    cfe = cf_extractor([data_ego, data_sur], meta)
    cfdata.append(cfe.cfdata)

cfdata = pd.concat(cfdata).reset_index(drop=True)
cfdata['conflict'] = cfdata['conflict'].astype(bool)
cfdata.to_hdf(path_output + 'HundredCar_cfdata.h5', key='data', mode='w')
print(cfdata.head())
