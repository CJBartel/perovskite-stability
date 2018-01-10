# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:16:37 2018

@author: Chris
"""

from PredictAABBXX6_script import PredictAABBXX6
import pandas as pd

# read in TableS3.csv which contains our experimental dataset and associated information
df = pd.read_csv('TableS3.csv')

df['compound'] = [cmpd.replace('MA2', 'Ma2') for cmpd in df['compound'].values]
df['A'] = [A if A != 'MA' else 'Ma' for A in df.A.values]

df['A1'] = df['A']
df['A2'] = df['A']
df['B1'] = df['B1']
df['X1'] = df['X']
df['X2'] = df['X']

# generate t using the current script  
df['t_check'] = [PredictAABBXX6(df.A1.values[idx], 
                                df.A2.values[idx],
                                df.B1.values[idx],
                                df.B2.values[idx],
                                df.X1.values[idx],
                                df.X2.values[idx]).t for idx in range(len(df))]
## compare t generated here to t in TableS1.csv
df['t_diff'] = abs(df['t'] - df['t_check'])
print('The max difference for t = %.3f' % df.t_diff.max())

# generate t using the current script  
df['tau_check'] = [PredictAABBXX6(df.A1.values[idx], 
                                df.A2.values[idx],
                                df.B1.values[idx],
                                df.B2.values[idx],
                                df.X1.values[idx],
                                df.X2.values[idx]).tau for idx in range(len(df))]
## compare t generated here to t in TableS1.csv
df['tau_diff'] = abs(df['tau'] - df['tau_check'])
print('The max difference for tau = %.3f' % df.tau_diff.max())
#
# fit calibrated classifier for tau
clf = PredictAABBXX6(df.A1.values[0], 
                                df.A2.values[0],
                                df.B1.values[0],
                                df.B2.values[0],
                                df.X1.values[0],
                                df.X2.values[0]).calibrate_tau

# generate tau  probabilities using the current script
df['tau_prob_check'] = [PredictAABBXX6(df.A1.values[idx], 
                                df.A2.values[idx],
                                df.B1.values[idx],
                                df.B2.values[idx],
                                df.X1.values[idx],
                                df.X2.values[idx]).tau_prob(clf) for idx in range(len(df))]
# compare tau probabilities generated here to those in TableS1.csv
df['tau_prob_diff'] = abs(df['tau_prob'] - df['tau_prob_check'])
print('The max difference for tau probabilities = %.3f' % df.tau_prob_diff.max())
print('Note probabilities can vary slightly due to variation in CV splits')