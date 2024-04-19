'''
This file is used to compute pma and pfa at each time moment.
'''

# Import libraries
import sys
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy import optimize
from scipy import stats

# Define the directory of the project
data_path = './localdata/'


# Define functions
def round_speed(samples, roundvs):
    vs = samples[samples['v']<roundvs.max()]['v'].values
    
    index_sorted = np.argsort(roundvs)
    roundvs_sorted = roundvs[index_sorted]

    idx1 = np.searchsorted(roundvs_sorted, vs)
    idx2 = np.clip(idx1 - 1, 0, len(roundvs_sorted)-1)

    diff1 = roundvs_sorted[idx1] - vs
    diff2 = vs - roundvs_sorted[idx2]

    indices = index_sorted[np.where(diff1 <= diff2, idx1, idx2)]
    samples.loc[samples['v']<roundvs.max(),'round_v'] = roundvs[indices]
    samples.loc[samples['v']>=roundvs.max(),'round_v'] = roundvs.max()
    
    return samples


def load_data(loc):
    samples = pd.read_hdf(data_path + '/samples/samples_'+loc+'.h5', key='data')
    samples = samples.reset_index(drop=True)
    samples_toinfer = pd.read_hdf(data_path + '/samples/samples_toinfer_'+loc+'.h5', key='samples')
    roundvs = samples_toinfer['round_v'].unique()
    samples = round_speed(samples, roundvs)
    return samples, samples_toinfer, roundvs


def spacing_inference_s(samples_toinfer, roundvs, loc):
    prob_s = {key:[] for key in roundvs}
    len_s_list = []

    for i in tqdm(range(len(roundvs)), desc=loc):
        roundv = roundvs[i]

        sample = samples_toinfer[samples_toinfer['round_v']==roundv]
        kde_s = stats.gaussian_kde(sample['s'].values)
        prob_s[roundv] = kde_s
        len_s_list.append(len(sample))
        
    len_s_list = np.array(len_s_list)
        
    return prob_s, len_s_list
    
    
def spacing_inference_sc(samples_toinfer, roundvs, loc, conflict_type):
    prob_sc = {key:[] for key in roundvs}
    smax_list, c_list, idx_empty = [], [], []

    for i in tqdm(range(len(roundvs)), desc=loc):
        roundv = roundvs[i]
        
        sample = samples_toinfer[(samples_toinfer['round_v']==roundv)&(samples_toinfer[conflict_type])]
        if len(sample) <= 5:
            idx_empty.append(i)
            smax_list.append(0)
            c_list.append(np.nan)
        else:
            kde_sc = stats.gaussian_kde(sample['s'].values)
            prob_sc[roundv] = kde_sc
            smax_list.append(sample['s'].max())
            c_list.append(len(sample))

    idx_empty = np.array(idx_empty)
    idx_unempty = np.setdiff1d(np.arange(len(roundvs)), idx_empty)
    for idx in idx_empty:
        fillin_idx = idx_unempty[np.argmin(np.abs(idx_unempty-idx))]
        prob_sc[roundvs[idx]] = prob_sc[roundvs[fillin_idx]]

    c_list = np.array(c_list)
    c_list[np.isnan(c_list)] = np.nanmin(c_list)

    return prob_sc, smax_list, c_list


def solve_threshold(prob_s, prob_sc, smax, c):
    range_s = np.arange(0, 200, 0.1)
    density_s = prob_s(range_s)
    smax = max(smax, range_s[np.argmax(density_s)])

    cum_smax_s = prob_s.integrate_box_1d(0, smax)
    cum_smax_sc = prob_sc.integrate_box_1d(0, smax)

    range_sc = np.arange(0, smax, 0.1)
    pma, pfa = [], []
    for i in range(len(range_sc)):
        pma0 = prob_sc.integrate_box_1d(range_sc[i], smax)
        pma.append(pma0)
        pfa0 = (prob_s.integrate_box_1d(0, range_sc[i])-c*prob_sc.integrate_box_1d(0, range_sc[i]))/(cum_smax_s-c*cum_smax_sc)
        pfa.append(pfa0)
    pma = np.array(pma)
    pfa = np.array(pfa)

    thresholds = np.zeros((19,6))
    alpha_list = np.arange(0.05,1.,0.05)
    for i in range(19):
        alpha = alpha_list[i]
        threshold = range_sc[np.argmin(alpha*pma + (1-alpha)*pfa)]
        thresholds[i,:] = [alpha, threshold, smax, cum_smax_s, cum_smax_sc, c]

    return thresholds


def compute_thresholds(roundvs, prob_s, prob_sc, smax_list, c_list, loc):
    parameters = pd.DataFrame(np.zeros((len(roundvs)*19, 7)), columns=['round_v','alpha','threshold','smax','cum_smax_s','cum_smax_sc','c'])
    parameters['round_v'] = np.repeat(roundvs, 19)
    parameters = parameters.set_index('round_v')
    
    for roundv, smax, c in tqdm(zip(roundvs, smax_list, c_list), desc=loc, total=len(roundvs)):
        prob_s_rv = prob_s[roundv]
        prob_sc_rv = prob_sc[roundv]
        thresholds = solve_threshold(prob_s_rv, prob_sc_rv, smax, c)
        parameters.loc[roundv, ['alpha','threshold','smax','cum_smax_s','cum_smax_sc','c']] = thresholds

    return parameters.reset_index()


# Freeway B
## Load data
print('Loading FreewayB data...')
samples_FreewayB, samples_toinfer_FreewayB, roundvs_FreewayB = load_data('FreewayB')

## Create dictionaries for spacing inference
print('Inferencing spacing...')
prob_s_FreewayB, len_s = spacing_inference_s(samples_toinfer_FreewayB, roundvs_FreewayB, 'FreewayB')
prob_sc1_FreewayB, smax_c1_FreewayB, c1_FreewayB = spacing_inference_sc(samples_toinfer_FreewayB, roundvs_FreewayB, 'FreewayB', 'conflict_1')
prob_sc2_FreewayB, smax_c2_FreewayB, c2_FreewayB = spacing_inference_sc(samples_toinfer_FreewayB, roundvs_FreewayB, 'FreewayB', 'conflict_2')
prob_sc3_FreewayB, smax_c3_FreewayB, c3_FreewayB = spacing_inference_sc(samples_toinfer_FreewayB, roundvs_FreewayB, 'FreewayB', 'conflict_3')
c1_FreewayB, c2_FreewayB, c3_FreewayB = c1_FreewayB/len_s, c2_FreewayB/len_s, c3_FreewayB/len_s

## Compute pma and pfa
print('Computing thresholds...')
parameters_FreewayB = []
ctype_list = ['conflict_1', 'conflict_2','conflict_3']
prob_sc_list = [prob_sc1_FreewayB, prob_sc2_FreewayB, prob_sc3_FreewayB]
smax_c_list = [smax_c1_FreewayB, smax_c2_FreewayB, smax_c3_FreewayB]
c_list = [c1_FreewayB, c2_FreewayB, c3_FreewayB]
for ctype, prob_sc, smax_c, c in zip(ctype_list, prob_sc_list, smax_c_list, c_list):
    parameters = compute_thresholds(roundvs_FreewayB, prob_s_FreewayB, prob_sc, smax_c, c, 'FreewayB')
    parameters['ctype'] = ctype
    parameters_FreewayB.append(parameters)
parameters_FreewayB = pd.concat(parameters_FreewayB).reset_index(drop=True)
parameters_FreewayB.to_csv(data_path + 'spacing/parameters_FreewayB.csv', index=False)


# 100Car
## Load data
samples_100Car, samples_toinfer_100Car, roundvs_100Car = load_data('100Car')

## Create dictionaries for spacing inference
print('Inferencing spacing...')
prob_s_100Car, len_s = spacing_inference_s(samples_toinfer_100Car, roundvs_100Car, '100Car')
prob_sc_100Car, smax_c_100Car, c_100Car = spacing_inference_sc(samples_toinfer_100Car, roundvs_100Car, '100Car', 'conflict')
c_100Car = c_100Car/len_s

## Compute pma and pfa
print('Computing thresholds...')
parameters_100Car = compute_thresholds(roundvs_100Car, prob_s_100Car, prob_sc_100Car, smax_c_100Car, c_100Car, '100Car')
parameters_100Car.to_csv(data_path + 'spacing/parameters_100Car.csv', index=False)
