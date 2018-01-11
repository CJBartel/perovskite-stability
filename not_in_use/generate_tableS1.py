# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:16:37 2018

@author: Chris
"""

from PredictABX3_script import PredictABX3
import pandas as pd

# read in TableS1.csv which contains our experimental dataset and associated information
df = pd.read_csv('TableS1.csv')
# generate t using the current script
df['t_check'] = [PredictABX3(ABX3).t for ABX3 in df.ABX3.values]
# compare t generated here to t in TableS1.csv
df['t_diff'] = abs(df['t'] - df['t_check'])
print('The max difference for t = %.3f' % df.t_diff.max())

# generate tau using the current script
df['tau_check'] = [PredictABX3(ABX3).tau for ABX3 in df.ABX3.values]
# compare tau generated here to tau in TableS1.csv
df['tau_diff'] = abs(df['tau'] - df['tau_check'])
print('The max difference for tau = %.3f' % df.tau_diff.max())

# fit calibrated classifier for tau
clf = PredictABX3('').calibrate_tau

# generate tau  probabilities using the current script
df['tau_prob_check'] = [PredictABX3(ABX3).tau_prob(clf) for ABX3 in df.ABX3.values]
# compare tau probabilities generated here to those in TableS1.csv
df['tau_prob_diff'] = abs(df['tau_prob'] - df['tau_prob_check'])
print('The max difference for tau probabilities = %.3f' % df.tau_prob_diff.max())
print('Note probabilities can vary slightly due to variation in CV splits')