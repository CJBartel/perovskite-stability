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


class PredictAABBXX3(object):
    """
    for doped ABX3s, where X can be O, F, Cl, Br, or I
        -predicts which cation is A or B
        -determines whether compound can be charge-balanced
        -assigns oxidation states for A and B
        -predicts radii (and octahedral and Goldschmidt parameters)
    """
    
    def __init__(self, A1, A2, x,
                       B1, B2, y,
                       X1, X2, z):
        """
        ABX3 to get A, B, oxA, oxB, rA, rB for
        """
        self.A1 = A1
        self.A2 = A2
        self.x = x
        self.B1 = B1
        self.B2 = B2
        self.y = y
        self.X1 = X1
        self.X2 = X2
        self.z = z
        
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
        x = self.x
        y = self.y
        z = self.z
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
        # try no heterovalency
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
                combo_to_idx_ox[combo] = idx_to_ox
            return combo_to_idx_ox
        else:
            bal_combos = [combo for combo in suitable_combos if combo not in isovalent_combos if np.sum(combo) == -X_charge]
            combo_to_idx_ox = {}
            for combo in bal_combos:
                idx_to_ox = {}
                for key in idx_dict:
                    idx_to_ox[key] = sorted([combo[idx] for idx in idx_dict[key]])
                combo_to_idx_ox[combo] = idx_to_ox
            return combo_to_idx_ox

    @property
    def unique_combos(self):
        unique_combos = []
        combo_dict = self.bal_combos
        cations = self.cations
        idx_dict = self.idx_dict
        for combo in combo_dict:
            if len(unique_combos) == 0:
                unique_combos.append(combo)
            else:
                big_count = 0
                for unique_combo in unique_combos:
                    count = 0
                    for el in combo_dict[combo].keys():
                        if combo_dict[combo][el] != combo_dict[unique_combo][el]:
                            count += 1
                    if count == len(cations):
                        big_count += 1
                if big_count == len(unique_combos):
                    unique_combos.append(combo)
        combo_to_idx_ox = {}
        for combo in unique_combos:
            idx_to_ox = {}
            for key in idx_dict:
                idx_to_ox[key] = sorted([combo[idx] for idx in idx_dict[key]])
            combo_to_idx_ox[combo] = idx_to_ox    
        return combo_to_idx_ox
        
#    @property
#    def choose_combo(self):
#        """
#        returns {el (str) : assigned oxidation state (int)} for cations
#        """
#        combos = self.unique_combos
#        if isinstance(combos, float):
#            return np.nan
#        else:
#            return combos[list(combos.keys())[0]]
    
    @property
    def AB_radii_dict(self):
        """
        returns {el (str) : {'A_rad' : radius if A (float),
                             'B_rad' : radius if B (float)}}
        """
        ox_dict = self.unique_combos
        if isinstance(ox_dict, float):
            return np.nan
        radii_dict = {}
        combos = list(ox_dict.keys())
        for combo in combos:
            combo_dict = {}
            for el in ox_dict[combo]:
                tmp_dict1 = {}
                for ox in ox_dict[combo][el]:
                    tmp_dict2 = {}
                    coords = list(Shannon_dict[el][ox].keys())
                    B_coords = [abs(coord - 6) for coord in coords]
                    mindex = [idx for idx in range(len(B_coords)) if B_coords[idx] == np.min(B_coords)][0]
                    B_coord = coords[mindex]
                    A_coords = [abs(coord - 12) for coord in coords]
                    mindex = [idx for idx in range(len(A_coords)) if A_coords[idx] == np.min(A_coords)][0]
                    A_coord = coords[mindex]
                    B_rad = Shannon_dict[el][ox][B_coord]['only_spin']
                    A_rad = Shannon_dict[el][ox][A_coord]['only_spin']
                    tmp_dict2['A_rad'] = A_rad
                    tmp_dict2['B_rad'] = B_rad
                    tmp_dict1[ox] = tmp_dict2
                combo_dict[el] = tmp_dict1
            radii_dict[combo] = combo_dict
            return radii_dict
    
    @property
    def rdict(self):
        AB_radii_dict = self.AB_radii_dict
        ox_dict = self.unique_combos
        if isinstance(AB_radii_dict, float) or (AB_radii_dict is None):
            return np.nan
        else:        
            rdict = {}
            combos = list(AB_radii_dict.keys())
            for combo in combos:
                combo_dict = {}
                for el in self.As:
                    rs = []
                    for ox in ox_dict[combo][el]:
                        rs.append(AB_radii_dict[combo][el][ox]['A_rad'])
                    combo_dict[el] = np.mean(rs)
                for el in self.Bs:
                    rs = []
                    for ox in ox_dict[combo][el]:
                        rs.append(AB_radii_dict[combo][el][ox]['B_rad'])
                    combo_dict[el] = np.mean(rs)
                for el in self.Xs:
                    combo_dict[el] = Shannon_dict[el][self.X_ox_dict[el]][6]['only_spin']
                rdict[combo] = combo_dict
            return rdict
    
    @property
    def ndict(self):
        ox_dict = self.unique_combos
        cations = self.cations
        if isinstance(ox_dict, float):
            return np.nan
        else:
            combos = list(ox_dict.keys())
            ndict = {combo : {cation : np.mean(ox_dict[combo][cation]) for cation in cations} for combo in combos}
            for combo in combos:
                for el in self.Xs:
                    ndict[combo][el] = self.X_ox_dict[el]
            return ndict
          
    @property
    def nA(self):
        As = self.As
        conc_dict = self.conc_dict
        ndict = self.ndict
        if isinstance(ndict, float):
            return np.nan
        else: 
            try:            
                return {combo : np.sum([conc_dict[A]*ndict[combo][A] for A in As]) for combo in list(ndict.keys())}
            except:
                return np.nan        

    @property
    def nB(self):
        Bs = self.Bs
        conc_dict = self.conc_dict
        ndict = self.ndict
        if isinstance(ndict, float):
            return np.nan
        else:
            try:
                return {combo : np.sum([conc_dict[B]*ndict[combo][B] for B in Bs]) for combo in list(ndict.keys())} 
            except:
                return np.nan

    @property
    def nX(self):
        Xs = self.Xs
        conc_dict = self.conc_dict
        ndict = self.ndict
        if isinstance(ndict, float):
            return np.nan
        else:
            try:            
                return {combo : np.sum([conc_dict[X]*ndict[combo][X]/3 for X in Xs]) for combo in list(ndict.keys())}
            except:
                return np.nan        
    
    @property
    def rA(self):
        As = self.As
        conc_dict = self.conc_dict
        rdict = self.rdict
        if isinstance(rdict, float):
            return np.nan
        else:    
            try:            
                return {combo : np.sum([conc_dict[A]*rdict[combo][A] for A in As]) for combo in list(rdict.keys())}
            except:
                return np.nan        
    
    @property
    def rB(self):
        Bs = self.Bs
        conc_dict = self.conc_dict
        rdict = self.rdict
        if isinstance(rdict, float):
            return np.nan
        else:
            try:            
                return {combo : np.sum([conc_dict[B]*rdict[combo][B] for B in Bs]) for combo in list(rdict.keys())}
            except:
                return np.nan        

    @property
    def rX(self):
        Xs = self.Xs
        conc_dict = self.conc_dict
        rdict = self.rdict
        if isinstance(rdict, float):
            return np.nan
        else:
            try:            
                return {combo : np.sum([conc_dict[X]/3*rdict[combo][X] for X in Xs]) for combo in list(rdict.keys())}
            except:
                return np.nan        
    
    @property
    def mu(self):
        """
        returns the predicted octahedral factor (float)
        """
        rB = self.rB
        rX = self.rX
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:
            try:            
                return {combo : rB[combo] / rX[combo] for combo in list(rB.keys())}
            except:
                return np.nan        
    
    @property
    def t(self):
        """
        returns the predicted Goldschmidt tolerance factor (float)
        """
        rA = self.rA
        rB = self.rB
        rX = self.rX     
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:
            try:            
                return {combo : (rA[combo] + rX[combo]) / (np.sqrt(2) * (rB[combo] + rX[combo])) for combo in list(rA.keys())}
            except:
                return np.nan        
        
    @property
    def tau(self):
        rA = self.rA
        rB = self.rB
        rX = self.rX
        nA = self.nA
        if (isinstance(self.AB_radii_dict, float)) or (rA == rB):
            return np.nan
        else:
            try:            
                return {combo : (rX[combo] / rB[combo] - nA[combo] * (nA[combo] - (rA[combo]/rB[combo])/np.log(rA[combo]/rB[combo]))) for combo in list(rA.keys())}
            except:
                return np.nan        
        
    @property
    def pred(self):
        rA = self.rA
        rB = self.rB        
        new_t = self.new_t
        if (isinstance(self.AB_radii_dict, float)) or (rA == rB):
            return np.nan
        else:
            try:
                return {combo : 1 if new_t[combo] <= 4.179 else -1 for combo in list(new_t.keys())}
            except:
                return np.nan
     
def main():
    print('-----THIS IS A DEMONSTRATION-----')
    
    Zunger_stable = ['K2AgInCl6', 'Rb2AgInCl6', 'Cs2AgInCl6',
                     'Rb2AgInBr6', 'Cs2AgInBr6', 'Rb2AgGaCl6', 'Cs2AgGaCl6',
                     'Cs2AgGaBr6', 'Cs2AgInI6', 'K2CuInCl6', 'Rb2CuInCl6',
                     'Cs2CuInCl6', 'Rb2CuInBr6', 'Cs2CuInBr6']
    
    Zunger_unstable = ['K2AgInBr6', 'K2AgGaCl6', 'K2AgGaBr6', 'Rb2AgGaBr6',
                       'K2AgInI6', 'Rb2AgInI6', 'K2AgGaI6', 'Rb2AgGaI6',
                       'K2CuInBr6', 'Rb2AgGaI6', 'Cs2AgGaI6',
                       'K2CuGaCl6', 'Rb2CuGaCl6', 'Cs2CuGaCl6',
                       'K2CuInI6', 'Rb2CuInI6', 'K2CuGaBr6', 
                       'Rb2CuGaBr6', 'Cs2CuGaBr6',
                       'K2CuGaI6', 'Rb2CuGaI6', 'Cs2CuGaI6']
    
    def get_inputs(formula):
        names = CompAnalyzer(formula).atom_names
        nums = CompAnalyzer(formula).atom_nums
        Xidx = nums.index(6)
        Aidx = nums.index(2)
        X = names[Xidx]
        A = names[Aidx]
        Bs = [name for name in names if name not in [A, X]]
        B1 = Bs[0]
        B2 = Bs[1]
        A1 = A
        A2 = A
        X1 = X
        X2 = X
        x = 0
        y = 0.5
        z = 0
        return A1, A2, x, B1, B2, y, X1, X2, z
    
    info_dict = {}
    for cmpd in Zunger_stable:
        tmp_dict = {}
        A1, A2, x, B1, B2, y, X1, X2, z = get_inputs(cmpd)
        obj = PredictAABBXX3(A1, A2, x,
                             B1, B2, y,
                             X1, X2, z)
        tau = obj.tau
        tmp_dict['stability'] = 1
        tmp_dict['tau'] = tau
        pred = [1 if tau[list(tau.keys())[0]] < 4.179 else -1][0]
        tmp_dict['pred'] = pred
        info_dict[cmpd] = tmp_dict
    for cmpd in Zunger_unstable:
        tmp_dict = {}
        A1, A2, x, B1, B2, y, X1, X2, z = get_inputs(cmpd)
        obj = PredictAABBXX3(A1, A2, x,
                             B1, B2, y,
                             X1, X2, z)
        tau = obj.tau
        tmp_dict['stability'] = -1
        tmp_dict['tau'] = tau
        pred = [1 if tau[list(tau.keys())[0]] < 4.179 else -1][0]
        tmp_dict['pred'] = pred        
        info_dict[cmpd] = tmp_dict
    good_cmpds = []
    bad_cmpds = []
    for cmpd in info_dict:
        if info_dict[cmpd]['stability'] == info_dict[cmpd]['pred']:
            good_cmpds.append(cmpd)
        else:
            bad_cmpds.append(cmpd)
    return Zunger_stable, Zunger_unstable, info_dict, good_cmpds, bad_cmpds
    
if __name__ == '__main__':
    Zunger_stable, Zunger_unstable, info_dict, good_cmpds, bad_cmpds = main()
    
    
    
    
