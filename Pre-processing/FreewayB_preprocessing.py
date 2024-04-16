'''
This file is used to preprocess the Freeway B dataset.
'''

# Import libraries
import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

# Define the directory of the project
path_rawdata = './localdata/rawdata/'
path_inputdata = './localdata/inputdata/'
path_outputdata = './localdata/outputdata/'


print(os.path.abspath(path_rawdata))
print(os.path.abspath(path_inputdata))
print(os.path.abspath(path_outputdata))

# Preprocessing
data_files = glob.glob(path_rawdata + '*.csv')
for data_file in data_files:
    data = pd.read_csv(data_file)
    suffix = int(data_file[-6:-4])
    data['suffix'] = suffix
    data['frameNum'] = (suffix*100000 + data['frameNum']).astype(int)
    data['carId'] = (suffix*100000 + data['carId']).astype(int)

    data['length'] = (np.sqrt((data.boundingBox1Xft-data.boundingBox2Xft)**2+(data.boundingBox1Yft-data.boundingBox2Yft)**2)+
                    np.sqrt((data.boundingBox3Xft-data.boundingBox4Xft)**2+(data.boundingBox3Yft-data.boundingBox4Yft)**2))/2
    data['length'] = data.groupby('carId')['length'].transform('median')
    data['headXft'] = (data.boundingBox1Xft + data.boundingBox4Xft)/2
    data['headYft'] = (data.boundingBox1Yft + data.boundingBox4Yft)/2
    data['tailXft'] = (data.boundingBox2Xft + data.boundingBox3Xft)/2
    data['tailYft'] = (data.boundingBox2Yft + data.boundingBox3Yft)/2

    data = data.rename(columns={'frameNum':'frame_id','carId':'track_id'})

    # preceding vehicle
    for idx in tqdm(data.index, total=len(data), desc=data_file[-5:-3]):
        alternatives = data[(data['frame_id']==data.frame_id.loc[idx])&
                            (data['laneId']==data.laneId.loc[idx])&
                            (data['track_id']!=data.track_id.loc[idx])]
        if data.laneId.loc[idx]<2.5:
            preceding = alternatives[alternatives.headXft<data.headXft.loc[idx]].sort_values(by='headXft', ascending=False)
            if len(preceding)>0:
                preceding = preceding.iloc[0]
                data.loc[idx, 'precedingId'] = preceding['track_id']
        elif data.laneId.loc[idx]>2.5:
            preceding = alternatives[alternatives.headXft>data.headXft.loc[idx]].sort_values(by='headXft', ascending=True)
            if len(preceding)>0:
                preceding = preceding.iloc[0]
                data.loc[idx, 'precedingId'] = preceding['track_id']

    data = data[['track_id','frame_id','headXft','headYft','length','speed','laneId','precedingId']]
    distance = ['headXft','headYft','length']
    data[distance] = data[distance]/3.281 # Feet to meters
    data['speed'] = data['speed']/2.237 # Miles/hour to m/s

    data.to_hdf(path_inputdata + 'FreewayB_'+data_file[-6:-4]+'.h5', key='data')

    # position based on speed
    data = data.drop_duplicates(['track_id','frame_id'])
    data = data.rename(columns={'headXft':'x','headYft':'y'})
    data = data.sort_values(['track_id','frame_id']).set_index('track_id')
    for trackid in tqdm(data.index.unique()):
        track = data.loc[trackid]
        if len(track.shape)==1:
            data.loc[trackid,'position'] = track['x']
        else:
            track = track.reset_index()
            direction = np.sign(track['x'].iloc[-1]-track['x'].iloc[0])
            data.loc[trackid,'position'] = (((track['x'].iloc[0]+direction*track['speed'].cumsum()/30)+
                                            (track['x'].iloc[-1]-direction*track['speed'].iloc[-1::-1].cumsum()/30))/2).values

    data = data.reset_index()

    # check if a preceding vehicle has more than one following vehicle
    num_following = data.groupby(['frame_id','precedingId'])['track_id'].count().reset_index().rename(columns={'track_id':'num_following'})
    num_following = num_following[num_following['num_following']>1]
    print('There are '+str(len(num_following['precedingId'].unique()))+' preceding vehicles with more than one following vehicle.')
    data = data[~data['precedingId'].isin(num_following['precedingId'].unique())]

    data[['pre_position','pre_speed','pre_length']] = data.set_index(['frame_id','track_id']).reindex(pd.MultiIndex.from_arrays(data[['frame_id','precedingId']].values.T, names=('frame_id', 'precedingId')))[['position','speed','length']].values


    # car following only
    data = data[~data['precedingId'].isna()]
    no_lane_change = data.groupby('track_id').laneId.transform('nunique')<=1
    single_leader = data.groupby('track_id').precedingId.transform('nunique')==1
    data = data[no_lane_change&single_leader]
    data = data[(data['length']<=7.5)&(data['pre_length']<=7.5)]

    data = data[['laneId','frame_id','track_id','position','speed','length','precedingId','pre_position','pre_speed','pre_length']]
    data[['laneId','frame_id','track_id','precedingId']] = data[['laneId','frame_id','track_id','precedingId']].astype(int)
    data.to_hdf(path_outputdata + 'FreewayB_'+data_file[-6:-4]+'.h5', key='data')


print('FreewayB preprocessing done!')
