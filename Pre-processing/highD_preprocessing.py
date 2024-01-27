'''
This file is used to preprocess the highD dataset.
'''

# Import libraries
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

# Define the directory of the project
parent_dir = r'U:/Vehicle Coordination Yiru'

# Read meta data
metadatafiles =  sorted(glob.glob(parent_dir + '/RawDatasets/highD/highd-dataset-v1.0/data/RecordingMetadata/*.csv'))
metadata = []
for metadatafile in metadatafiles:
    df = pd.read_csv(metadatafile)
    metadata.append(df)
metadata = pd.concat(metadata)

'''
In:
metadata.groupby('locationId').numVehicles.sum()
---
Out:
locationId
1    85962
2     3074
3     3747
4     4751
5    10079
6     2903
Name: numVehicles, dtype: int64
'''

# Preprocessing
for locid in tqdm(metadata.locationId.unique()):
    loc = 'highD_' + str(locid).zfill(2)
    data_files = [str(id).zfill(2) + '_tracks' for id in metadata[(metadata.locationId==locid)].id.values]
    metadata_files = [str(id).zfill(2) + '_tracksMeta' for id in metadata[(metadata.locationId==locid)].id.values]
    data = []
    for data_file, metadata_file in tqdm(zip(data_files, metadata_files), total=len(data_files)):
        df = pd.read_csv(parent_dir + '/RawDatasets/highD/highd-dataset-v1.0/data/' + data_file +'.csv')
        meta = pd.read_csv(parent_dir + '/RawDatasets/highD/highd-dataset-v1.0/data/' + metadata_file +'.csv')
        df = df.rename(columns={'frame':'frame_id','id':'track_id','xVelocity':'vx','yVelocity':'vy','xAcceleration':'ax','yAcceleration':'ay','width':'length','height':'width'})
        df['class'] = meta.set_index('id').reindex(df.track_id.values)['class'].values
        df['direction'] = meta.set_index('id').reindex(df.track_id.values)['drivingDirection'].values
        df = df[['track_id','frame_id','x','y','vx','vy','ax','ay','width','length','class','direction','dhw','thw','ttc','precedingId','followingId','laneId']]
        
        # replace 0 precedingId and followingId with nan
        df['precedingId'] = df.precedingId.replace(0,np.nan)
        df['followingId'] = df.followingId.replace(0,np.nan)

        # redefine indcies to be unique for later data combination
        df['track_id'] = int(data_file[:2])*10000+df.track_id
        df['precedingId'] = int(data_file[:2])*10000+df.precedingId
        df['followingId'] = int(data_file[:2])*10000+df.followingId
        df['frame_id'] = int(data_file[:2])*100000+df.frame_id
        data.append(df)
    data = pd.concat(data).reset_index(drop=True)
    data.to_hdf(parent_dir + '/InputData/highD/'+loc+'.h5', key='data')

    # # correct direction
    # data.loc[data['direction']==1, 'x'] = data.x.max()-data.loc[data['direction']==1, 'x']
    # data.loc[data['direction']==1, ['vx','ax']] = -data.loc[data['direction']==1, ['vx','ax']]
    # data.loc[data['direction']==2, 'x'] = data.loc[data['direction']==2, 'x']-data.x.min()

    # position based on speed
    data['speed'] = np.sqrt(data['vx']**2+data['vy']**2)
    data = data.sort_values(['track_id','frame_id']).set_index('track_id')
    for trackid in tqdm(data.index.unique()):
        track = data.loc[trackid].reset_index()
        direction = np.sign(track['x'].iloc[-1]-track['x'].iloc[0])
        data.loc[trackid,'position'] = (((track['x'].iloc[0]+direction*track['speed'].cumsum()/25)+
                                        (track['x'].iloc[-1]-direction*track['speed'].iloc[-1::-1].cumsum()/25))/2).values
    data = data.reset_index()
    data[['pre_position','pre_speed','pre_length']] = data.set_index(['frame_id','track_id']).reindex(pd.MultiIndex.from_arrays(data[['frame_id','precedingId']].values.T, names=('frame_id', 'precedingId')))[['position','speed','length']].values

    # car following only
    data = data[~data['precedingId'].isna()]
    no_lane_change = data.groupby('track_id').laneId.transform('nunique')<=1
    single_leader = data.groupby('track_id').precedingId.transform('nunique')==1
    data = data[no_lane_change&single_leader]
    data = data[(data['length']<=7.5)&(data['pre_length']<=7.5)]

    data = data[['laneId','frame_id','track_id','position','speed','length','precedingId','pre_position','pre_speed','pre_length']]
    data['precedingId'] = data['precedingId'].astype(int)
    data.to_hdf(parent_dir + '/OutputData/ADAS/FCW/data/highD/'+loc+'.h5', key='data')


print('highD preprocessing done!')