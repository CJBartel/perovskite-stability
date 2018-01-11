# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:02:29 2017

@author: Chris
"""

import numpy as np
import pandas as pd
from make_radii_dict import ionic_radii_dict as Shannon_dict
import math
import re
from sklearn.calibration import CalibratedClassifierCV


class PredictABX3(object):
    """
    for undoped ABX3s, where X can be O, F, Cl, Br, or I
        -predicts which cation is A or B
        -determines whether compound can be charge-balanced
        -assigns oxidation states for A and B
        -predicts radii (and octahedral and Goldschmidt parameters)
    """
    
    def __init__(self, initial_form):
        """
        CCX3 to classify
        """
        self.initial_form = initial_form
    
    @property
    def els(self):
        """
        list of elements in formula (str)
        """
        return re.findall('[A-Z][a-z]?', self.initial_form)
    
    @property
    def X(self):
        """
        anion (str)
        """
        el_num_pairs = [[el_num_pair[idx] for idx in range(len(el_num_pair)) if el_num_pair[idx] != ''][0]
                                  for el_num_pair in re.findall('([A-Z][a-z]\d*)|([A-Z]\d*)', self.initial_form)]
        return [el_num_pair.replace('3', '') for el_num_pair in el_num_pairs if '3' in el_num_pair][0]
    
    @property
    def cations(self):
        """
        list of cations (str)
        """
        els = self.els
        return [el for el in els if el != self.X]
        
    @property
    def X_ox_dict(self):
        """
        returns {el (str): oxidation state (int)} for allowed anions
        """
        return {'N' : -3,
                'O' : -2,
                'S' : -2,
                'Se' : -2,
                'F' : -1,
                'Cl' : -1,
                'Br' : -1,
                'I' : -1}
    
    @property
    def plus_one(self):
        """
        returns list of elements (str) likely to be 1+
        """
        return ['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Ag']
    
    @property
    def plus_two(self):
        """
        returns list of elements (str) likely to be 2+
        """        
        return ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra']
    
    @property
    def plus_three(self):
        """
        returns list of elements (str) likely to be 3+
        """        
        return ['Sc', 'Y', 'La', 'Al', 'Ga', 'In',
                'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
                'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']    
            
    @property
    def chi_dict(self):
        """
        returns {el (str) : Pauling electronegativity (float)} for cations
        """
        cations = self.cations
        chi_dict = {}
        with open('electronegativities.csv') as f:
            for line in f:
                line = line.split(',')
                if line[0] in cations:
                    chi_dict[line[0]] = float(line[1][:-1])
        return chi_dict
    
    @property
    def allowed_ox(self):
        """
        returns {el (str) : list of allowed oxidation states (int)} for cations
        """
        X = self.X
        cations = self.cations
        allowed_ox_dict = {}
        for cation in cations:
            # if cation is commonly 1+, make that the only allowed oxidation state
            if cation in self.plus_one:
                allowed_ox_dict[cation] = [1]
            # if cation is commonly 2+, make that the only allowed oxidation state
            elif cation in self.plus_two:
                allowed_ox_dict[cation] = [2]
            # otherwise, use the oxidation states that have corresponding Shannon radii                
            else:
                allowed_ox_dict[cation] = [val for val in list(Shannon_dict[cation].keys()) if val > 0]
        allowed_ox_dict[X] = [self.X_ox_dict[X]]
        return allowed_ox_dict
    
    @property
    def charge_bal_combos(self):
        """
        returns list of oxidation state pairs (tuple) which charge-balance X or NaN if none found
        """
        cations = self.cations
        X = self.X
        allowed_ox = self.allowed_ox
        ox1s = allowed_ox[cations[0]]
        ox2s = allowed_ox[cations[1]]
        oxX = allowed_ox[X][0]
        bal_combos = []
        for ox1 in ox1s:
            for ox2 in ox2s:
                if ox1 + ox2 == -3*oxX:
                    bal_combos.append((ox1, ox2))
        if len(bal_combos) == 0:
            #print(self.initial_form)
            #print('No charge balanced combinations. . .')
            return np.nan
        else:
            return bal_combos 
    
    @property
    def chosen_ox_states(self):
        """
        returns {el (str) : assigned oxidation state (int)} for cations
        """
        combos = self.charge_bal_combos
        chi_dict = self.chi_dict
        cations = self.cations
        X = self.X
        plus_three =self.plus_three
        # if only one charge-balanced combination exists, use it
        if len(combos) == 1:
            ox_states = dict(zip(cations, combos[0]))
        # if two combos exists and they are the reverse of one another
        elif (len(combos) == 2) and (combos[0] == combos[1][::-1]):
            # assign the minimum oxidation state to the more electronegative cation
            min_ox = np.min(combos[0])
            max_ox = np.max(combos[1])
            epos_el = [el for el in cations if chi_dict[el] == np.min(list(chi_dict.values()))][0]
            eneg_el = [el for el in cations if el != epos_el][0]
            ox_states = {epos_el : max_ox,
                         eneg_el : min_ox}
        else:
            # if one of the cations is probably 3+, let it be 3+
            if (cations[0] in plus_three) or (cations[1] in plus_three):
                if X == 'O':
                    if (3,3) in combos:
                        combo = (3,3)
                        ox_states = dict(zip(cations, list(combo)))
            # else compare electronegativities - if 0.9 < chi1/chi2 < 1.1, minimize the oxidation state diff
            elif np.min(list(chi_dict.values())) > 0.9 * np.max(list(chi_dict.values())):
                diffs = [abs(combo[0] - combo[1]) for combo in combos]
                mindex = [idx for idx in range(len(diffs)) if diffs[idx] == np.min(diffs)]
                if len(mindex) == 1:
                    mindex = mindex[0]
                    combo = combos[mindex]
                    ox_states = dict(zip(cations, combo))
                else:
                    min_ox = np.min([combos[idx] for idx in mindex])
                    max_ox = np.max([combos[idx] for idx in mindex])
                    epos_el = [el for el in cations if chi_dict[el] == np.min(list(chi_dict.values()))][0]
                    eneg_el = [el for el in cations if el != epos_el][0]
                    ox_states = {epos_el : max_ox,
                                 eneg_el : min_ox} 
            else:
                diffs = [abs(combo[0] - combo[1]) for combo in combos]
                maxdex = [idx for idx in range(len(diffs)) if diffs[idx] == np.max(diffs)]
                if len(maxdex) == 1:
                    maxdex = maxdex[0]
                    combo = combos[maxdex]
                    ox_states = dict(zip(cations, combo))
                else:
                    min_ox = np.min([combos[idx] for idx in maxdex])
                    max_ox = np.max([combos[idx] for idx in maxdex])
                    epos_el = [el for el in cations if chi_dict[el] == np.min(list(chi_dict.values()))][0]
                    eneg_el = [el for el in cations if el != epos_el][0]
                    ox_states = {epos_el : max_ox,
                                eneg_el : min_ox}
        return ox_states
    
    @property
    def AB_radii_dict(self):
        """
        returns {el (str) : {'A_rad' : radius if A (float),
                             'B_rad' : radius if B (float)}}
        """
        ox_dict = self.chosen_ox_states
        if isinstance(ox_dict, float):
            return np.nan
        radii_dict = {}
        for el in ox_dict:
            tmp_dict = {}
            # get the oxidation state
            ox = ox_dict[el]
            coords = list(Shannon_dict[el][ox].keys())
            # get the B CN as the one available nearest 6
            B_coords = [abs(coord - 6) for coord in coords]
            mindex = [idx for idx in range(len(B_coords)) if B_coords[idx] == np.min(B_coords)][0]
            B_coord = coords[mindex]
            # get the A CN as the one available nearest 12
            A_coords = [abs(coord - 12) for coord in coords]
            mindex = [idx for idx in range(len(A_coords)) if A_coords[idx] == np.min(A_coords)][0]
            A_coord = coords[mindex]
            # produce the equivalent B-site and A-site radii
            B_rad = Shannon_dict[el][ox][B_coord]['only_spin']
            A_rad = Shannon_dict[el][ox][A_coord]['only_spin']
            tmp_dict['A_rad'] = A_rad
            tmp_dict['B_rad'] = B_rad
            radii_dict[el] = tmp_dict
        return radii_dict

    @property
    def pred_A(self):
        """
        returns predicted A (str)
        """
        ox_dict = self.chosen_ox_states
        if isinstance(ox_dict, float):
            return np.nan        
        radii_dict = self.AB_radii_dict
        el1 = list(radii_dict.keys())[0]
        el2 = list(radii_dict.keys())[1]
        if (radii_dict[el1]['A_rad'] > radii_dict[el2]['B_rad']) and (radii_dict[el1]['B_rad'] > radii_dict[el2]['A_rad']):
            return el1
        elif (radii_dict[el1]['A_rad'] < radii_dict[el2]['B_rad']) and (radii_dict[el1]['B_rad'] < radii_dict[el2]['A_rad']):
            return el2
        elif (radii_dict[el1]['A_rad'] > radii_dict[el2]['A_rad']) and (radii_dict[el1]['B_rad'] > radii_dict[el2]['B_rad']):
            return el1
        elif (radii_dict[el1]['A_rad'] < radii_dict[el2]['A_rad']) and (radii_dict[el1]['B_rad'] < radii_dict[el2]['B_rad']):
            return el2
        elif (radii_dict[el1]['B_rad'] > radii_dict[el2]['B_rad']):
            return el1
        elif (radii_dict[el1]['B_rad'] < radii_dict[el2]['B_rad']):
            return el2
        elif (radii_dict[el1]['A_rad'] > radii_dict[el2]['A_rad']):
            return el1
        elif (radii_dict[el1]['A_rad'] < radii_dict[el2]['A_rad']):
            return el2  
        else:
            if ox_dict[el1] < ox_dict[el2]:
                return el1
            else:
                return el2
    
    @property
    def pred_B(self):
        """
        returns predicted B (str)
        """
        cations = self.cations
        pred_A = self.pred_A
        if pred_A in cations:
            return [cation for cation in cations if cation != pred_A][0]
        else:
            return np.nan
    
    @property
    def nA(self):
        """
        returns oxidation state assigned to A (int)
        """
        if isinstance(self.chosen_ox_states, float):
            return np.nan
        else:
            return self.chosen_ox_states[self.pred_A]
    
    @property
    def nB(self):
        """
        returns oxidation state assigned to B (int)
        """
        if isinstance(self.chosen_ox_states, float):
            return np.nan
        else:        
            return self.chosen_ox_states[self.pred_B]
    
    @property
    def rA(self):
        """
        returns predicted Shannon ionic radius for A (float)
        """
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:        
            return self.AB_radii_dict[self.pred_A]['A_rad']
    
    @property
    def rB(self):
        """
        returns predicted Shannon ionic radius for B (float)
        """
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:      
            return self.AB_radii_dict[self.pred_B]['B_rad']
    
    @property
    def rX(self):
        """
        returns Shannon ionic radius for X (float)
        """
        return Shannon_dict[self.X][self.X_ox_dict[self.X]][6]['only_spin']
    
    @property
    def mu(self):
        """
        returns the predicted octahedral factor (float)
        """
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:          
            return self.rB / self.rX
    
    @property
    def t(self):
        """
        returns the predicted Goldschmidt tolerance factor (float)
        """
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:          
            return (self.rA + self.rX) / (np.sqrt(2) * (self.rB + self.rX))
        
    @property
    def tau(self):
        """
        returns tau (float)
        """
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:
            try:
                return ((1/self.mu) - (self.nA)**2 + (self.nA) * (self.rA/self.rB)/(np.log(self.rA/self.rB)))
            except:
                return np.nan
    
    @property
    def tau_pred(self):
        """
        returns prediction of 1 (perovskite) or -1 (nonperovskite) based on tau
        """
        if math.isnan(self.tau):
            return np.nan
        else:
            return [1 if self.tau < 4.18 else -1][0]
        
    @property
    def t_pred(self):
        """
        returns prediction of 1 (perovskite) or -1 (nonperovskite) based on t
        """        
        if math.isnan(self.t):
            return np.nan
        else:
            return [1 if (self.t > 0.825) and (self.t < 1.059) else -1][0]  
        
    @property
    def calibrate_tau(self):
        """
        returns a calibrated classifier to yield tau probabilities
        """
        df = pd.read_csv('TableS1.csv')
        df['tau'] = [PredictABX3(ABX3).tau for ABX3 in df.ABX3.values]
        X, y = df['tau'].values.reshape(-1, 1), df['exp_label'].values
        clf = CalibratedClassifierCV(cv=3)
        clf.fit(X, y)
        return clf
    
    def tau_prob(self, clf):
        """
        Args:
            clf (sklearn object) - calibrated classifier based on tau
        Returns:
            probability of perovskite based on tau (float)
        """
        X = [[self.tau]]
        return clf.predict_proba(X)[0][1]
    
            
def main():
    
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
    return df
    
if __name__ == '__main__':
    df = main()
    
