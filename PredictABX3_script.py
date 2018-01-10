# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:02:29 2017

@author: Chris
"""

import numpy as np
import os 
from CompAnalyzer_script import CompAnalyzer
import pandas as pd
from make_radii_dict import ionic_radii_dict as Shannon_dict
import math

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
        ABX3 to get A, B, oxA, oxB, rA, rB for
        """
        self.initial_form = initial_form
    
    @property
    def ca_obj(self):
        """
        returns CompAnalyzer object
        """
        return CompAnalyzer(self.initial_form)
    
    @property
    def good_form(self):
        """
        returns well-formatted formula
        """
        return self.ca_obj.good_form
    
    @property
    def atom_names(self):
        """
        returns alphabetical list of elements
        """
        return self.ca_obj.atom_names
    
    @property
    def atom_nums(self):
        """
        returns number of each element in compound (same order as names)
        """
        return self.ca_obj.atom_nums
    
    @property
    def num_els(self):
        """
        returns number of elements in compound
        """
        return self.ca_obj.num_els
    
    @property
    def X_ox_dict(self):
        """
        returns {el (str): oxidation state (int)} for allowed anions
        """
        return {'O' : -2,
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
    def X(self):
        """
        returns anion (str)
        """
        return self.atom_names[self.atom_nums.index(3)]
    
    @property
    def cations(self):
        """
        returns list of cations (str)
        """
        return [el for el in self.atom_names if el != self.X]
    
    @property
    def chi_dict(self):
        """
        returns {el (str) : Pauling electronegativity (float)} for cations
        """
        return {cation : CompAnalyzer(cation).feat_lst('X')[0] for cation in self.cations}    
    
    @property
    def allowed_ox(self):
        """
        returns {el (str) : list of allowed oxidation states (int)} for cations
        """
        X = self.X
        cations = self.cations
        allowed_ox_dict = {}
        for cation in cations:
            if cation in self.plus_one:
                allowed_ox_dict[cation] = [1]
            elif cation in self.plus_two:
                allowed_ox_dict[cation] = [2]
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
    def choose_combo(self):
        """
        returns {el (str) : assigned oxidation state (int)} for cations
        """
        combos = self.charge_bal_combos
        chi_lst = list(self.chi_dict.values())
        cations = self.cations
        X = self.X
        if isinstance(combos, float):
            return np.nan
        AB_lst = cations
        if len(combos) == 1:
            return dict(zip(AB_lst, list(combos[0])))
        elif (len(combos) == 2) and (combos[0] == combos[1][::-1]):
            min_ox = np.min(combos[0])
            max_ox = np.max(combos[0])
            epos_idx = chi_lst.index(np.min(chi_lst))
            eneg_idx = chi_lst.index(np.max(chi_lst))
            return {AB_lst[epos_idx] : max_ox,
                    AB_lst[eneg_idx] : min_ox}
        else:
            if (AB_lst[0] in self.plus_three) or (AB_lst[1] in self.plus_three):
                if X == 'O':
                    if (3,3) in combos:
                        combo = (3,3)
                        return dict(zip(AB_lst, list(combo)))
            elif np.min(chi_lst) > 0.9 * np.max(chi_lst):
                diffs = [abs(combo[0] - combo[1]) for combo in combos]
                mindex = [idx for idx in range(len(diffs)) if diffs[idx] == np.min(diffs)]
                if len(mindex) == 1:
                    mindex = mindex[0]
                    combo = combos[mindex]
                else:
                    min_ox = np.min([combos[idx] for idx in mindex])
                    max_ox = np.max([combos[idx] for idx in mindex])
                    epos_idx = chi_lst.index(np.min(chi_lst))
                    eneg_idx = chi_lst.index(np.max(chi_lst))
                    return {AB_lst[epos_idx] : max_ox,
                            AB_lst[eneg_idx] : min_ox}
                return dict(zip(AB_lst, list(combos[mindex])))
            else:
                diffs = [abs(combo[0] - combo[1]) for combo in combos]
                maxdex = [idx for idx in range(len(diffs)) if diffs[idx] == np.max(diffs)]
                if len(maxdex) == 1:
                    maxdex = maxdex[0]
                    combo = combos[maxdex]
                else:
                    min_ox = np.min([combos[idx] for idx in maxdex])
                    max_ox = np.max([combos[idx] for idx in maxdex])
                    epos_idx = chi_lst.index(np.min(chi_lst))
                    eneg_idx = chi_lst.index(np.max(chi_lst))
                    return {AB_lst[epos_idx] : max_ox,
                            AB_lst[eneg_idx] : min_ox}
                return dict(zip(AB_lst, list(combos[maxdex])))
    
    @property
    def AB_radii_dict(self):
        """
        returns {el (str) : {'A_rad' : radius if A (float),
                             'B_rad' : radius if B (float)}}
        """
        ox_dict = self.choose_combo
        if isinstance(ox_dict, float):
            return np.nan
        radii_dict = {}
        for el in ox_dict:
            tmp_dict = {}
            ox = ox_dict[el]
            coords = list(Shannon_dict[el][ox].keys())
            B_coords = [abs(coord - 6) for coord in coords]
            mindex = [idx for idx in range(len(B_coords)) if B_coords[idx] == np.min(B_coords)][0]
            B_coord = coords[mindex]
            A_coords = [abs(coord - 12) for coord in coords]
            mindex = [idx for idx in range(len(A_coords)) if A_coords[idx] == np.min(A_coords)][0]
            A_coord = coords[mindex]
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
        ox_dict = self.choose_combo
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
        if isinstance(self.choose_combo, float):
            return np.nan
        else:
            return self.choose_combo[self.pred_A]
    
    @property
    def nB(self):
        """
        returns oxidation state assigned to B (int)
        """
        if isinstance(self.choose_combo, float):
            return np.nan
        else:        
            return self.choose_combo[self.pred_B]
    
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
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:
            try:
                return ((1/self.mu) - (self.nA)**2 + (self.nA) * (self.rA/self.rB)/(np.log(self.rA/self.rB)))
            except:
                return np.nan
    
    @property
    def is_perov(self):
        if math.isnan(self.tau):
            return np.nan
        else:
            return [1 if self.tau < 4.18 else -1][0]
        

    @property
    def inv_rA(self):
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:        
            return self.AB_radii_dict[self.pred_A]['B_rad']
        
    @property
    def inv_rB(self):
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:
            return self.AB_radii_dict[self.pred_B]['A_rad']
    
def main():
    print('-----THIS IS A DEMONSTRATION-----')
    test_formula = 'LaGaO3'
    obj = PredictABX3(test_formula)
    good_form = obj.good_form
    A = obj.pred_A
    B = obj.pred_B
    oxA = obj.nA
    oxB = obj.nB
    rA = obj.rA
    rB = obj.rB
    mu = obj.mu
    t = obj.t
    tau = obj.tau
    is_perov = obj.is_perov
    
    print('The formula for demonstration is %s' % test_formula)
    print('A is predicted to be %s' % A)
    print('B is predicted to be %s' % B)
    print('The oxidation state of A is predicted to be %s' % oxA)
    print('The oxidation state of B is predicted to be %s' % oxB)
    print('The radius of A is predicted to be %.2f A' % rA)
    print('The radius of B is predicted to be %.2f A' % rB)
    print('which gives a Goldschmidt factor of %.3f' % t)
    print('and an octahedral factor of %.3f' % mu)
    
    return obj
    
if __name__ == '__main__':
    obj = main()
    
