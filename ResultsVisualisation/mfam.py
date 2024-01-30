'''
This file is used for once computation for visualisation.
'''

# Import libraries
from tqdm import tqdm
import numpy as np
from scipy import stats


# Define functions
def spacing_inference(samples_toinfer, roundvs, loc):
    prob_s = {key:[] for key in roundvs}
    prob_sc1 = {key:[] for key in roundvs}
    prob_sc2 = {key:[] for key in roundvs}
    smax_c1_list, smax_c2_list = [], []
    c1_list, c2_list = [], []

    idx_empty_c1, idx_empty_c2 = [], []
    for i in tqdm(range(len(roundvs)), desc=loc):
        roundv = roundvs[i]

        ## p_s
        sample = samples_toinfer[samples_toinfer['round_v']==roundv]
        kde_s = stats.gaussian_kde(sample['s'].values)
        prob_s[roundv] = kde_s
        
        ## p_sc1, smax_c1, c1
        sample_c1 = sample[sample['conflict_1']]
        if len(sample_c1) <= 5:
            idx_empty_c1.append(i)
            smax_c1_list.append(0)
            c1_list.append(np.nan)
        else:
            kde_sc1 = stats.gaussian_kde(sample_c1['s'].values)
            prob_sc1[roundv] = kde_sc1
            smax_c1_list.append(sample_c1['s'].max())
            c1_list.append(len(sample_c1)/len(sample))

        ## p_sc2, smax_c2, c2
        sample_c2 = sample[sample['conflict_2']]
        if len(sample_c2) <= 5:
            idx_empty_c2.append(i)
            smax_c2_list.append(0)
            c2_list.append(np.nan)
        else:
            kde_sc2 = stats.gaussian_kde(sample_c2['s'].values)
            prob_sc2[roundv] = kde_sc2
            smax_c2_list.append(sample_c2['s'].max())
            c2_list.append(len(sample_c2)/len(sample))

    idx_empty_c1 = np.array(idx_empty_c1)
    idx_unempty_c1 = np.setdiff1d(np.arange(len(roundvs)), idx_empty_c1)
    for idx in idx_empty_c1:
        fillin_idx = idx_unempty_c1[np.argmin(np.abs(idx_unempty_c1-idx))]
        prob_sc1[roundvs[idx]] = prob_sc1[roundvs[fillin_idx]]

    idx_empty_c2 = np.array(idx_empty_c2)
    idx_unempty_c2 = np.setdiff1d(np.arange(len(roundvs)), idx_empty_c2)
    for idx in idx_empty_c2:
        fillin_idx = idx_unempty_c2[np.argmin(np.abs(idx_unempty_c2-idx))]
        prob_sc2[roundvs[idx]] = prob_sc2[roundvs[fillin_idx]]

    c1_list = np.array(c1_list)
    c2_list = np.array(c2_list)
    c1_list[np.isnan(c1_list)] = np.nanmin(c1_list)
    c2_list[np.isnan(c2_list)] = np.nanmin(c2_list)

    return (prob_s, prob_sc1, prob_sc2), (smax_c1_list, smax_c2_list, c1_list, c2_list)


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

    return range_sc, pma, pfa
