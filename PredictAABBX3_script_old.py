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
from math import gcd
from itertools import combinations, product
from sklearn.calibration import CalibratedClassifierCV
import math
from sklearn.externals import joblib


class PredictAABBXX3(object):
    """
    for doped ABX3s, where X can be O, F, Cl, Br, or I
        -predicts which cation is A or B
        -determines whether compound can be charge-balanced
        -assigns oxidation states for A and B
        -predicts radii (and octahedral and Goldschmidt parameters)
    """
    
    def __init__(self, A1, A2,
                       B1, B2,
                       X1, X2):
        """
        ABX3 to get A, B, oxA, oxB, rA, rB for
        """
        self.A1 = A1
        self.A2 = A2
        self.B1 = B1
        self.B2 = B2
        self.X1 = X1
        self.X2 = X2
        
    @property
    def As(self):
        return list(set([self.A1, self.A2]))
    
    @property
    def Bs(self):
        return list(set([self.B1, self.B2]))
    
    @property
    def Xs(self):
        return list(set([self.X1, self.X2]))
        
    @property
    def conc_dict(self):
        if self.A1 == self.A2:
            x = 0
        else:
            x = 0.5
        if self.B1 == self.B2:
            y = 0
        else:
            y = 0.5
        if self.X1 == self.X2:
            z = 0
        else:
            z = 1.5
        conc_dict = {}
        if x != 0:
            conc_dict[self.A1] = 1-x
            conc_dict[self.A2] = x
        else:
            conc_dict[self.A1] = 1
        if y != 0:
            conc_dict[self.B1] = 1-y
            conc_dict[self.B2] = y
        else:
            conc_dict[self.B1] = 1            
        if z != 0:
            conc_dict[self.X1] = 3-z
            conc_dict[self.X2] = z
        else:
            conc_dict[self.X1] = 3
        return conc_dict
    
    @property
    def int_conc_dict(self):
        conc_dict = self.conc_dict
        big_conc_dict = {}
        for el in conc_dict:
            big_conc_dict[el] = int(1000*conc_dict[el])
        els = list(big_conc_dict.keys())
        amts = [big_conc_dict[el] for el in els]
        list_to_join = []
        for idx in range(len(els)):
            list_to_join.append(els[idx])
            if amts[idx] != 0:  
                list_to_join.append(amts[idx])
        list_to_join = [str(val) for val in list_to_join]
        big_initial_form = ''.join(list_to_join)
        nums = CompAnalyzer(big_initial_form).atom_nums
        names = CompAnalyzer(big_initial_form).atom_names
        combos = list(combinations(nums, 2))
        factors = [gcd(combo[0], combo[1]) for combo in combos]
        min_factor = np.min(factors)
        small_nums = [int(num/min_factor) for num in nums]
        names_to_nums = dict(zip(names, small_nums))
        factor = names_to_nums[self.A1] / conc_dict[self.A1]        
        return {key : int(factor * conc_dict[key]) for key in conc_dict}
    
    @property
    def initial_form(self):
        conc_dict = self.int_conc_dict
        if len(self.As) == 2:
            if len(self.Bs) == 2:
                if len(self.Xs) == 2:
                    list_to_join = [self.A1, conc_dict[self.A1],
                                    self.A2, conc_dict[self.A2],
                                    self.B1, conc_dict[self.B1],
                                    self.B2, conc_dict[self.B2],
                                    self.X1, conc_dict[self.X1],
                                    self.X2, conc_dict[self.X2]]
                else:
                    list_to_join = [self.A1, conc_dict[self.A1],
                                    self.A2, conc_dict[self.A2],
                                    self.B1, conc_dict[self.B1],
                                    self.B2, conc_dict[self.B2],
                                    self.X1, conc_dict[self.X1]]
            else:
                if len(self.Xs) == 2:
                    list_to_join = [self.A1, conc_dict[self.A1],
                                    self.A2, conc_dict[self.A2],
                                    self.B1, conc_dict[self.B1],
                                    self.X1, conc_dict[self.X1],
                                    self.X2, conc_dict[self.X2]]
                else:
                    list_to_join = [self.A1, conc_dict[self.A1],
                                    self.A2, conc_dict[self.A2],
                                    self.B1, conc_dict[self.B1],
                                    self.X1, conc_dict[self.X1]]  
        elif len(self.Bs) == 2:
            if len(self.Xs) == 2:
                list_to_join = [self.A1, conc_dict[self.A1],
                                self.B1, conc_dict[self.B1],
                                self.B2, conc_dict[self.B2],
                                self.X1, conc_dict[self.X1],
                                self.X2, conc_dict[self.X2]]
            else:
                list_to_join = [self.A1, conc_dict[self.A1],
                                self.B1, conc_dict[self.B1],
                                self.B2, conc_dict[self.B2],
                                self.X1, conc_dict[self.X1]]
        else:
            if len(self.Xs) == 2:
                list_to_join = [self.A1, conc_dict[self.A1],
                                self.B1, conc_dict[self.B1],
                                self.X1, conc_dict[self.X1],
                                self.X2, conc_dict[self.X2]]
            else:
                print("Hey there, stranger. That composition isn't even doped. . .")
                list_to_join = [self.A1, conc_dict[self.A1],
                                self.B1, conc_dict[self.B1],
                                self.X1, conc_dict[self.X1]]           
                
        
        return ''.join([str(val) for val in list_to_join])
        
    
    @property
    def ca_obj(self):
        """
        returns CompAnalyzer object
        """
        return CompAnalyzer(self.initial_form)
    
    @property
    def good_form(self):
        """
        returns alphabetical list of elements
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
                'I' : -1,
                'S' : -2,
                'Se' : -2}
    
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
    def cations(self):
        """
        returns list of cations (str)
        """
        return [el for el in self.atom_names if el not in self.Xs]
    
    @property
    def chi_dict(self):
        """
        returns {el (str) : Pauling electronegativity (float)} for cations
        """
        return {cation : CompAnalyzer(cation).feat_lst('X')[0] for cation in self.cations}

    @property
    def site_dict(self):
        conc_dict = self.int_conc_dict
        tmp_dict = {el : list(np.arange(conc_dict[el])) for el in conc_dict}
        site_dict = {el : [] for el in tmp_dict}
        for key in tmp_dict:
            for val in tmp_dict[key]:
                site_dict[key].append('_'.join([key, str(val)]))
                
        return site_dict  
    
    @property
    def allowed_ox(self):
        """
        returns {el (str) : list of allowed oxidation states (int)} for cations
        """
        site_dict = self.site_dict
        Xs = self.Xs
        cations = self.cations
        ox_dict = {}
        for cation in cations:
            tmp_dict1 = {}
            sites = site_dict[cation]
            for site in sites:
                tmp_dict2 = {}
                if cation in self.plus_one:
                    oxs = [1]
                elif cation in self.plus_two:
                    oxs = [2]
                else:
                    oxs = [val for val in list(Shannon_dict[cation].keys()) if val > 0]
                tmp_dict2['oxs'] = oxs
                tmp_dict1[site] = tmp_dict2
            ox_dict[cation] = tmp_dict1
        for X in Xs:
            tmp_dict1 = {}
            sites = site_dict[X]
            for site in sites:
                tmp_dict2 = {}
                tmp_dict2['oxs'] = [self.X_ox_dict[X]]
                tmp_dict1[site] = tmp_dict2
            ox_dict[X] = tmp_dict1
                
        return ox_dict
    
    @property
    def X_charge(self):
        charge = 0
        conc_dict = self.int_conc_dict
        allowed_ox = self.allowed_ox
        for key in allowed_ox:
            if key in self.Xs:
                X_sites = allowed_ox[key]
                for X_site in X_sites:
                    charge += allowed_ox[key][X_site]['oxs'][0]
        return charge
    
    @property
    def idx_dict(self):
        cations = self.cations
        allowed_ox = self.allowed_ox
        idx_dict = {}
        count = 0
        for key in cations:
            num_sites = len(allowed_ox[key].keys())
            indices = list(np.arange(count, count + num_sites))
            count += num_sites
            idx_dict[key] = indices
        return idx_dict
    
    @property
    def bal_combos(self):
        X_charge = self.X_charge
        allowed_ox = self.allowed_ox
        idx_dict = self.idx_dict
        cations = self.cations
        lists = [allowed_ox[key][site]['oxs'] for key in cations for site in list(allowed_ox[key].keys())]
        combos = list(product(*lists))
        isovalent_combos = []
        suitable_combos = []
        for combo in combos:
            iso_count = 0
            suit_count = 0
            for key in idx_dict:
                curr_oxs = [combo[idx] for idx in idx_dict[key]]
                if np.min(curr_oxs) == np.max(curr_oxs):
                    iso_count += 1
                if np.min(curr_oxs) >= np.max(curr_oxs) - 1:
                    suit_count += 1
            if iso_count == len(cations):
                isovalent_combos.append(combo)
            if suit_count == len(cations):
                suitable_combos.append(combo)
        bal_combos = [combo for combo in isovalent_combos if np.sum(combo) == -X_charge]
        if len(bal_combos) > 0:
            combo_to_idx_ox = {}
            for combo in bal_combos:
                idx_to_ox = {}
                for key in idx_dict:
                    idx_to_ox[key] = sorted([combo[idx] for idx in idx_dict[key]])
                    if idx_to_ox not in list(combo_to_idx_ox.values()):
                        combo_to_idx_ox[combo] = idx_to_ox                    
                combo_to_idx_ox[combo] = idx_to_ox
            return combo_to_idx_ox
        else:
            bal_combos = [combo for combo in suitable_combos if combo not in isovalent_combos if np.sum(combo) == -X_charge]
            combo_to_idx_ox = {}
            for combo in bal_combos:
                idx_to_ox = {}
                for key in idx_dict:
                    idx_to_ox[key] = sorted([combo[idx] for idx in idx_dict[key]])
                    if idx_to_ox not in list(combo_to_idx_ox.values()):
                        combo_to_idx_ox[combo] = idx_to_ox
            return combo_to_idx_ox
        
    @property
    def unique_combos(self):
        combos = self.bal_combos
        unique_combos = {}
        for combo in combos:
            if combos[combo] not in list(unique_combos.values()):
                unique_combos[combo] = combos[combo]
        return unique_combos
        
        
    @property
    def combos_near_isovalency(self):
        combos = self.unique_combos
        cations = self.cations
        hetero_dict = {}
        for combo in combos:
            sum_states = 0
            for cation in cations:
                sum_states += len(list(set(combos[combo][cation])))
            hetero_dict[combo] = sum_states - len(set(cations))
        min_heterovalency = np.min(list(hetero_dict.values()))
        near_iso_dict = {}
        for combo in combos:
            if hetero_dict[combo] == min_heterovalency:
                near_iso_dict[combo] = combos[combo]
        return near_iso_dict
    
    @property
    def choice_dict(self):
        combos = self.combos_near_isovalency
        cations = self.cations
        choices = {cation : [] for cation in cations}

        for cation in cations:
            for combo in combos:
                choices[cation].extend(combos[combo][cation])
                choices[cation] = list(set(choices[cation]))
        return choices
    
    @property
    def choose_combo(self):
        cations = self.cations
        conc_dict = self.conc_dict
        As = self.As
        Bs = self.Bs
        els = list(set(cations + [self.X1, self.X2]))
        choices = self.choice_dict
        X_ox_dict = self.X_ox_dict
        ox_dict = {}
        ox_dict[self.X1] = X_ox_dict[self.X1]
        if self.X1 != self.X2:
            ox_dict[self.X2] = X_ox_dict[self.X2]
        for cation in cations:
            if len(choices[cation]) == 1:
                ox_dict[cation] = choices[cation][0]
        if len(ox_dict) == len(els):
            return ox_dict
        else:
            unspec_els = [el for el in els if el not in ox_dict]
            unspec_charge = -np.sum([conc_dict[el]*ox_dict[el] for el in ox_dict])
            unspec_amt = np.sum([conc_dict[el] for el in unspec_els])
            unspec_charge_sum = unspec_charge*unspec_amt
            if len(unspec_els) == 2:
                unspec_combos = list(product(choices[unspec_els[0]], choices[unspec_els[1]]))
            elif len(unspec_els) == 3:
                unspec_combos = list(product(choices[unspec_els[0]], choices[unspec_els[1]], choices[unspec_els[2]]))
            elif len(unspec_els) == 4:
                unspec_combos = list(product(choices[unspec_els[0]], choices[unspec_els[1]], choices[unspec_els[2]], choices[unspec_els[3]]))
            good_combos = []
            for combo in unspec_combos:
                amt = 0
                for idx in range(len(unspec_els)):
                    amt += conc_dict[unspec_els[idx]]*combo[idx]
                if amt == unspec_charge:
                    good_combos.append(combo) 
            
            biggest_spread = np.max([np.max(combo) - np.min(combo) for combo in good_combos])
            smallest_spread = np.min([np.max(combo) - np.min(combo) for combo in good_combos])
            spread_combos = [combo for combo in good_combos if np.max(combo) - np.min(combo) == biggest_spread]
            tight_combos = [combo for combo in good_combos if np.max(combo) - np.min(combo) == smallest_spread]
            chis = [CompAnalyzer(el).feat_lst('X')[0] for el in unspec_els]
            maxdex = chis.index(np.max(chis))
            mindex = chis.index(np.min(chis))
            if np.min(chis) <= 0.9*np.max(chis):
                if len(spread_combos) > 1:
                    for combo in spread_combos:
                        if combo[mindex] == np.max(combo):
                            if combo[maxdex] == np.min(combo):
                                for idx in range(len(unspec_els)):
                                    el = unspec_els[idx]
                                    ox_dict[el] = combo[idx]
                else:
                    combo = spread_combos[0]
                    for idx in range(len(unspec_els)):
                        el = unspec_els[idx]
                        ox_dict[el] = combo[idx]
            else:
                if len(tight_combos) > 1:
                    for combo in tight_combos:
                        if combo[mindex] == np.max(combo):
                            if combo[maxdex] == np.min(combo):
                                for idx in range(len(unspec_els)):
                                    el = unspec_els[idx]
                                    ox_dict[el] = combo[idx]
                else:
                    combo = tight_combos[0]
                    for idx in range(len(unspec_els)):
                        el = unspec_els[idx]
                        ox_dict[el] = combo[idx]
        return ox_dict
    
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
            if el not in [self.X1, self.X2]:
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
    def nA1(self):
        """
        returns oxidation state assigned to A (int)
        """
        if isinstance(self.choose_combo, float):
            return np.nan
        else:
            return self.choose_combo[self.A1]
        
    @property
    def nA2(self):
        """
        returns oxidation state assigned to A (int)
        """
        if isinstance(self.choose_combo, float):
            return np.nan
        else:
            return self.choose_combo[self.A2]    
        
    @property
    def nA(self):
        """
        returns predicted Shannon ionic radius for B (float)
        """
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:      
            return np.mean([self.nA1, self.nA2])    
        
    @property
    def nB(self):
        """
        returns predicted Shannon ionic radius for B (float)
        """
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:      
            return np.mean([self.nB1, self.nB2])         
    
    @property
    def nB1(self):
        """
        returns oxidation state assigned to A (int)
        """
        if isinstance(self.choose_combo, float):
            return np.nan
        else:
            return self.choose_combo[self.B1]
        
    @property
    def nB2(self):
        """
        returns oxidation state assigned to A (int)
        """
        if isinstance(self.choose_combo, float):
            return np.nan
        else:
            return self.choose_combo[self.B2]   
    
    @property
    def rA1(self):
        """
        returns predicted Shannon ionic radius for A (float)
        """
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:        
            return self.AB_radii_dict[self.A1]['A_rad']
        
    @property
    def rA2(self):
        """
        returns predicted Shannon ionic radius for A (float)
        """
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:        
            return self.AB_radii_dict[self.A2]['A_rad']

    @property
    def rA(self):
        """
        returns predicted Shannon ionic radius for B (float)
        """
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:      
            return np.mean([self.rA1, self.rA2])        
        
    @property
    def rB1(self):
        """
        returns predicted Shannon ionic radius for A (float)
        """
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:        
            return self.AB_radii_dict[self.B1]['B_rad']
        
    @property
    def rB2(self):
        """
        returns predicted Shannon ionic radius for A (float)
        """
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:        
            return self.AB_radii_dict[self.B2]['B_rad']       
    
    @property
    def rB(self):
        """
        returns predicted Shannon ionic radius for B (float)
        """
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:      
            return np.mean([self.rB1, self.rB2])
    
    @property
    def rX1(self):
        """
        returns Shannon ionic radius for X (float)
        """
        return Shannon_dict[self.X1][self.X_ox_dict[self.X1]][6]['only_spin']
    
    @property
    def rX2(self):
        """
        returns Shannon ionic radius for X (float)
        """
        return Shannon_dict[self.X2][self.X_ox_dict[self.X2]][6]['only_spin']
    
    @property
    def rX(self):
        """
        returns predicted Shannon ionic radius for B (float)
        """
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:      
            return np.mean([self.rX1, self.rX2])    
    
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
    def t_pred(self):
        """
        returns the predicted Goldschmidt tolerance factor (float)
        """
        if math.isnan(self.t):
            return np.nan
        else: 
            t = self.t
            if (t <= 0.825) or (t >= 1.059):
                return 'nonperovskite'
            elif (t > 0.825) & (t < 1.059):
                return 'perovskite'
            else:
                return np.nan
        
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
    def tau_pred(self):
        if math.isnan(self.tau):
            return np.nan
        else:
            return ['perovskite' if self.tau < 4.18 else 'nonperovskite'][0]
        
    @property
    def tau_prob(self):
        if math.isnan(self.tau):
            return np.nan
        else:
            clf = joblib.load('calibrated_tau.pkl')
            X = [[self.tau]]
            return clf.predict_proba(X)[0][1]    
        
def main():
    A1 = 'Cs'
    A2 = 'Cs'
    
    B1 = 'Ag'
    B2 = 'In'
    
    X1 = 'Cl'
    X2 = 'Cl'
    
    obj = PredictAABBXX3(A1, A2,
                         B1, B2,
                         X1, X2)
    return obj
    
if __name__ == '__main__':
    obj = main()
    
    
    
    
