'''
This file is used to determine conflicts and sample data for spacing inferences.
'''

import sys
import pandas as pd
import numpy as np
from tqdm import tqdm


# Define the directory of the project
data_path = './Conflict-detection-MFaM/localdata/'


# Define functions
def determine_conflicts(samples):
    samples['conflict_1'] = False
    relative_speed = (samples['v']>0)
    samples.loc[relative_speed&(samples['s']<=(3*samples['v'])), 'conflict_1'] = True

    samples['conflict_2'] = False
    relative_speed = (samples['v']>5)
    samples.loc[relative_speed&(samples['s']<=(2.5*samples['v'])), 'conflict_2'] = True
    relative_speed = (samples['v']>2)&(samples['v']<=5)
    samples.loc[relative_speed&(samples['s']<=(3*samples['v'])), 'conflict_2'] = True
    relative_speed = (samples['v']>0)&(samples['v']<=2)
    samples.loc[relative_speed&(samples['s']<=(3.5*samples['v'])), 'conflict_2'] = True

    samples['conflict_3'] = False
    relative_speed = (samples['v']>5)
    samples.loc[relative_speed&(samples['s']<=(2.5*samples['v'])), 'conflict_3'] = True
    relative_speed = (samples['v']>2)&(samples['v']<=5)
    absolute_speed = (samples['speed']>25)
    samples.loc[relative_speed&absolute_speed&(samples['s']<=(3.5*samples['v'])), 'conflict_3'] = True
    absolute_speed = (samples['speed']>10)&(samples['speed']<=25)
    samples.loc[relative_speed&absolute_speed&(samples['s']<=(3.*samples['v'])), 'conflict_3'] = True
    absolute_speed = (samples['speed']<=10)
    samples.loc[relative_speed&absolute_speed&(samples['s']<=(2.5*samples['v'])), 'conflict_3'] = True
    relative_speed = (samples['v']>0)&(samples['v']<=2)
    absolute_speed = (samples['speed']>5)
    samples.loc[relative_speed&absolute_speed&(samples['s']<=(0.5*samples['speed'])), 'conflict_3'] = True
    absolute_speed = (samples['speed']>2)&(samples['speed']<=5)
    samples.loc[relative_speed&absolute_speed&(samples['s']<=(0.3*samples['speed'])), 'conflict_3'] = True
    absolute_speed = (samples['speed']>1)&(samples['speed']<=2)
    samples.loc[relative_speed&absolute_speed&(samples['s']<=0.6), 'conflict_3'] = True

    return samples


def Grouping(samples, vehnum):
    samples = samples.sort_values(by='v')
    samples['round_v'] = np.round(samples.v,1)
    groups = samples.groupby('round_v').v.count()
    try:
        threshold = groups[groups>=vehnum].index[-1]
        sample1 = []
        for roundv in samples[samples.round_v<=threshold].round_v.unique():
            sample1.append(samples[samples.round_v==roundv])
        sample1 = pd.concat(sample1)
        print('--- '+str(roundv)+' ----')
    except:
        threshold = 0
        sample1 = samples[samples.round_v<0].copy()
    sample2 = samples[samples.round_v>threshold].copy()
    sample2['round_v'] = np.arange(len(sample2))//vehnum
    sample2['round_v'] = (np.round(sample2.groupby('round_v').v.mean(),1)).reindex(sample2.round_v).values
    samples = pd.concat((sample1, sample2))

    return samples


# FreewayB (synthetic conflicts)

## Load data
data_all = []
for loc in tqdm(['FreewayB_0'+str(i) for i in range(7,0,-1)]):
    data = pd.read_hdf(data_path + 'outputdata/'+loc+'.h5', key='data')
    data_all.append(data)
data_all = pd.concat(data_all).reset_index(drop=True)

data_all['s'] = abs(data_all['pre_position'] - data_all['position']) - data_all['pre_length'] # net distance
data_all['v'] = data_all['speed'] - data_all['pre_speed']
samples_FreewayB = data_all[(data_all['s']>0)&(data_all['v']>0)].copy()
data_all = []

## Determine conflicts
samples_FreewayB['ttc'] = samples_FreewayB['s']/samples_FreewayB['v']
print('The percentage of TTCs that are inf: ', np.isinf(samples_FreewayB['ttc']).sum()/len(samples_FreewayB))
samples_FreewayB = determine_conflicts(samples_FreewayB.copy())
samples_FreewayB.to_hdf(data_path + 'samples/samples_FreewayB.h5', key='data')

## Sample data
samples = Grouping(samples_FreewayB, 7500)
samples = samples.sort_values(by='v').reset_index(drop=True)
print('--- '+str(len(samples[samples.round_v<=10].round_v.unique()))+' ----')
samples[['s','round_v','conflict_1','conflict_2','conflict_3']].to_hdf(data_path + 'samples/samples_toinfer_FreewayB.h5', key='samples')


# 100Car (real conflicts)

## Load data
data = pd.read_hdf(data_path + 'outputdata/HundredCar_CFData.h5', key='data')
samples_100Car = data[(data['s']>0)&(data['v']>0)].copy()

## Conflicts are empirical
samples_100Car['conflict'] = samples_100Car['conflict'].astype(bool)
samples_100Car['ttc'] = samples_100Car['s']/samples_100Car['v']
print('The percentage of TTCs that are inf: ', np.isinf(samples_100Car['ttc']).sum()/len(samples_100Car))
samples_100Car.to_hdf(data_path + 'samples/samples_100Car.h5', key='data')

## Sample data
samples = Grouping(samples_100Car, 800)
samples = samples.sort_values(by='v').reset_index(drop=True)
print('--- '+str(len(samples[samples.round_v<=10].round_v.unique()))+' ----')
samples[['s','round_v','conflict']].to_hdf(data_path + 'samples/samples_toinfer_100Car.h5', key='samples')
