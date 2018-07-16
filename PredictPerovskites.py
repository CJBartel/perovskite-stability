import numpy as np
import pandas as pd
fjson = 'Shannon_radii_dict.json'
if not os.path.exists(fjson):
    from make_radii_dict import ionic_radii_dict as Shannon_dict    
else:
    with open(fjson) as f:
        Shannon_dict = json.load(f)
import math
import re
from sklearn.calibration import CalibratedClassifierCV
from itertools import combinations, product
from math import gcd


class PredictABX3(object):
    """
    for undoped ABX3s
        -predicts which cation is A or B
        -determines whether compound can be charge-balanced
        -assigns oxidation states for A and B
        -predicts radii
        -generates t and tau
        -classifies as perovskite/nonperovskite based on t and tau
        -generates tau-derived probability of stability in the perovskite structure
    """
    
    def __init__(self, initial_form):
        """
        Args:
            initial_form (str) - CC'X3 to classify
        """
        self.initial_form = initial_form
        
    @property
    def good_form(self):
        """
        returns standard formula (str); alphabetized, "1s", etc.
        """
        el_num_pairs = re.findall('([A-Z][a-z]\d*)|([A-Z]\d*)', self.initial_form)
        el_num_pairs = [[pair[idx] for idx in range(len(pair))if pair[idx] != ''][0] for pair in el_num_pairs]
        el_num_pairs = [pair+'1' if bool(re.search(re.compile('\d'), pair)) == False else pair for pair in el_num_pairs]
        el_num_pairs = sorted(el_num_pairs)
        formula = ''.join(el_num_pairs)
        nums = list(map(int, re.findall('\d+', formula)))
        if 1 not in nums:
            names = re.findall('[A-Z][a-z]?', formula)        
            combos = list(combinations(nums, 2))
            factors = [gcd(combo[0], combo[1]) for combo in combos]
            gcf = np.min(factors)
            new_nums = [int(np.round(num/gcf)) for num in nums]        
            el_num_pairs = []
            for idx in range(len(names)):
                el_num_pairs.append(''.join([names[idx], str(new_nums[idx])]))
            el_num_pairs = [str(pair) for pair in el_num_pairs]
            return ''.join(sorted(el_num_pairs))
        else:
            return formula
    
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
                'I' : -1,
                'Fo' : -1}
    
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
        returns {el (str) : list of allowed oxidation states (int)} for each ion
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
        returns list of oxidation state pairs (tuple of ints) which charge-balance X3
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
        if isinstance(combos, float):
            return np.nan
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
                if self.X_ox_dict[X] == -2:
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
    def nX(self):
        """
        returns oxidation state assigned to X (int)
        """
        if isinstance(self.chosen_ox_states, float):
            return np.nan
        else:        
            return self.X_ox_dict[self.X]        
    
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
        if self.X != 'N':
            return Shannon_dict[self.X][self.X_ox_dict[self.X]][6]['only_spin']
        else:
            return Shannon_dict[self.X][self.X_ox_dict[self.X]][4]['only_spin']
    
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
            if self.rA <= self.rB:
                return np.nan
            else:
                return ((1/self.mu) - (self.nA)**2 + (self.nA) * (self.rA/self.rB)/(np.log(self.rA/self.rB)))
    
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
        if math.isnan(self.tau):
            return np.nan
        X = [[self.tau]]
        return clf.predict_proba(X)[0][1]


class PredictAABBXX6(object):
    """
    classifies the following compounds:
        -ABX3 (defaults to PredictABX3(CC'X3))
        -A2BB'X6
        -AA'B2X6
        -A2B2(XX')6
        -AA'BB'X6
        -AA'B2(XX')6
        -A2BB'(XX')6
        -AA'BB'(XX')6
    """
    
    def __init__(self, A1, A2, B1, B2, X1, X2):
        """
        Args:
            A1 (str) - element A
            A2 (str) - element A' if applicable, otherwise A
            B1 (str) - element B
            B2 (str) - element B' if applicable, otherwise B
            X1 (str) - element X
            X2 (str) - element X' if applicable, otherwise X           
        """
        self.A1 = A1
        self.A2 = A2
        self.B1 = B1
        self.B2 = B2
        self.X1 = X1
        self.X2 = X2
        
    @property
    def is_single(self):
        if (self.A1 == self.A2) and (self.B1 == self.B2) and (self.X1 == self.X2):
            return 1
        else:
            return -1
        
    @property
    def A(self):
        if self.is_single == 1:
            return PredictABX3(self.good_form).pred_A
        else:
            return np.nan
        
    @property
    def B(self):
        if self.is_single == 1:
            return PredictABX3(self.good_form).pred_B
        else:
            return np.nan
        
    @property
    def As(self):
        """
        returns list of A cations (str)
        """
        return list(set([self.A1, self.A2]))
    
    @property
    def Bs(self):
        """
        returns list of B cations (str)
        """        
        return list(set([self.B1, self.B2]))
    
    @property
    def Xs(self):
        """
        returns list of X anions (str)
        """        
        return list(set([self.X1, self.X2]))
    
    @property
    def X(self):
        if self.is_single == 1:
            return self.Xs[0]
        else:
            return self.Xs
    
    @property
    def els(self):
        """
        returns list of elements (str) in As, Bs, Xs order
        """        
        return self.As + self.Bs + self.Xs
    
    @property
    def formula(self):
        """
        returns pretty chemical formula in AA'BB'X3X'3 format (str)
        """
        if len(self.As) == 1:
            A_piece = ''.join([self.As[0], '2'])
        else:
            A_piece = ''.join(self.As)
        if len(self.Bs) == 1:
            B_piece = ''.join([self.Bs[0], '2'])
        else:
            B_piece = ''.join(self.Bs)
        if len(self.Xs) == 1:
            X_piece = ''.join([self.Xs[0], '6'])
        else:
            X_piece = ''.join([self.Xs[0], '3', self.Xs[1], '3'])
        return ''.join([A_piece, B_piece, X_piece])
    
    @property
    def good_form(self):
        """
        returns standard formula (str); alphabetized, "1s", etc.
        """
        el_num_pairs = re.findall('([A-Z][a-z]\d*)|([A-Z]\d*)', self.formula)
        el_num_pairs = [[pair[idx] for idx in range(len(pair))if pair[idx] != ''][0] for pair in el_num_pairs]
        el_num_pairs = [pair+'1' if bool(re.search(re.compile('\d'), pair)) == False else pair for pair in el_num_pairs]
        el_num_pairs = sorted(el_num_pairs)
        formula = ''.join(el_num_pairs)
        nums = list(map(int, re.findall('\d+', formula)))
        if 1 not in nums:
            names = re.findall('[A-Z][a-z]?', formula)        
            combos = list(combinations(nums, 2))
            factors = [gcd(combo[0], combo[1]) for combo in combos]
            gcf = np.min(factors)
            new_nums = [int(np.round(num/gcf)) for num in nums]        
            el_num_pairs = []
            for idx in range(len(names)):
                el_num_pairs.append(''.join([names[idx], str(new_nums[idx])]))
            el_num_pairs = [str(pair) for pair in el_num_pairs]
            return ''.join(sorted(el_num_pairs))  
        else:
            return formula
    
    @property
    def atom_names(self):
        """
        returns alphabetical list (str) of atomic symbols in composition
        e.g., good_form = 'Al2O3', atom_names = ['Al','O']
        """
        return re.findall('[A-Z][a-z]?', self.good_form)
    
    @property
    def atom_nums(self):
        """
        returns list (int) corresponding with number of each element in composition
            order of list corresponds with alphabetized atomic symbols in composition
        e.g., good_form = 'Al2O3', atom_nums = [2, 3]
        """
        return list(map(int, re.findall('\d+', self.good_form)))
    
    @property
    def frac_atom_nums(self):
        """
        returns list (float) of mol fraction of each element in composition
            order of list corresponds with alphabetized atomic symbols in composition
        e.g., good_form = 'Al2O3', frac_atom_nums = [0.4, 0.6]
        """
        atom_nums = self.atom_nums
        num_atoms = self.num_atoms
        return [num / num_atoms for num in atom_nums]

    @property
    def conc_dict(self):
        """
        returns dictionary of {el (str) : concentration in AxA'1-xByB'1-yXzX'3-z format (float)}
        """
        els = self.atom_names
        conc = self.frac_atom_nums
        natoms = self.num_atoms
        return {els[idx] : conc[idx] *natoms/2 for idx in range(len(els))}

    @property
    def num_els(self):
        """
        returns how many unique elements in composition (int)
        e.g., good_form = 'Al2O3', num_els = 2
        """
        return len(self.atom_names)

    @property
    def num_atoms(self):
        """
        returns how many atoms in composition (int)
        e.g., good_form = 'Al2O3', num_atoms = 5
        """
        return np.sum(self.atom_nums)    
    
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
                'I' : -1,
                'Fo' : -1}
    
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
        return self.As + self.Bs
    
    @property
    def anions(self):
        """
        returns list of anions (str)
        """        
        return self.Xs
    
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
    def site_dict(self):
        """
        returns dictionary of {el : [el_SITE0, el_SITE1, ...]}
        """
        els = self.atom_names
        nums = self.atom_nums
        site_dict = {els[idx] : ['_'.join([els[idx], str(counter)]) for counter in range(nums[idx])] for idx in range(len(els))}
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
        """
        returns the total charge of anions (float)
        """
        charge = 0
        allowed_ox = self.allowed_ox
        for key in allowed_ox:
            if key in self.Xs:
                X_sites = allowed_ox[key]
                for X_site in X_sites:
                    charge += allowed_ox[key][X_site]['oxs'][0]
        return charge
    
    @property
    def idx_dict(self):
        """
        returns dictionary of {el : [idx0, idx1, ...]}
        """
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
        """
        returns dictionary of {ox state combo (tup) : {el : [ox state by site (float)]}}
        """
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
        """
        returns unique version of self.bal_combos
        """
        combos = self.bal_combos
        if isinstance(combos, float) or len(combos) == 0:
            return np.nan
        unique_combos = {}
        for combo in combos:
            if combos[combo] not in list(unique_combos.values()):
                unique_combos[combo] = combos[combo]
        return unique_combos
        
        
    @property
    def combos_near_isovalency(self):
        """
        returns dictionary of most isovalent (within element) unique combos
        """
        combos = self.unique_combos
        if isinstance(combos, float) or len(combos) == 0:
            return np.nan        
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
        """
        returns dictionary of {el (str) : [potential ox states (float)]}
        """
        combos = self.combos_near_isovalency
        if isinstance(combos, float) or len(combos) == 0:
            return np.nan        
        cations = self.cations
        choices = {cation : [] for cation in cations}
        for cation in cations:
            for combo in combos:
                choices[cation].extend(combos[combo][cation])
                choices[cation] = list(set(choices[cation]))
        return choices
    
    @property
    def chosen_ox_states(self):
        """
        returns dictionary of {el (str) : chosen ox state (float)}
        """
        cations = self.cations
        conc_dict = self.conc_dict
        els = self.els
        choices = self.choice_dict
        if isinstance(choices, float):
            return np.nan        
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
            if len(unspec_els) == 1:
                unspec_combos = list(product(choices[unspec_els[0]]))
            elif len(unspec_els) == 2:
                unspec_combos = list(product(choices[unspec_els[0]], choices[unspec_els[1]]))
            elif len(unspec_els) == 3:
                unspec_combos = list(product(choices[unspec_els[0]], choices[unspec_els[1]], choices[unspec_els[2]]))
            elif len(unspec_els) == 4:
                unspec_combos = list(product(choices[unspec_els[0]], choices[unspec_els[1]], choices[unspec_els[2]], choices[unspec_els[3]]))
            elif len(unspec_els) == 5:
                unspec_combos = list(product(choices[unspec_els[0]], choices[unspec_els[1]], choices[unspec_els[2]], choices[unspec_els[3]], choices[unspec_els[4]]))  
            elif len(unspec_els) == 6:
                unspec_combos = list(product(choices[unspec_els[0]], choices[unspec_els[1]], choices[unspec_els[2]], choices[unspec_els[3]], choices[unspec_els[4]], choices[unspec_els[5]]))                 
            good_combos = []
            for combo in unspec_combos:
                amt = 0
                for idx in range(len(unspec_els)):
                    amt += conc_dict[unspec_els[idx]]*combo[idx]
                if amt == unspec_charge:
                    good_combos.append(combo) 
            if len(good_combos) == 0:
                return np.nan
            biggest_spread = np.max([np.max(combo) - np.min(combo) for combo in good_combos])
            smallest_spread = np.min([np.max(combo) - np.min(combo) for combo in good_combos])
            spread_combos = [combo for combo in good_combos if np.max(combo) - np.min(combo) == biggest_spread]
            tight_combos = [combo for combo in good_combos if np.max(combo) - np.min(combo) == smallest_spread]
            chi_dict = self.chi_dict
            chis = [chi_dict[el] for el in unspec_els]
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
                                return ox_dict
                    min_ox_most_elec = np.min([combo[maxdex] for combo in spread_combos])
                    for combo in spread_combos:
                        if (combo[maxdex] == min_ox_most_elec):
                            for idx in range(len(unspec_els)):
                                el = unspec_els[idx]
                                ox_dict[el] = combo[idx]
                            return ox_dict
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
        ox_dict = self.chosen_ox_states
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
        returns oxidation state assigned to A (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).nA        
        if isinstance(self.chosen_ox_states, float):
            return np.nan
        else:
            return self.chosen_ox_states[self.A1]
        
    @property
    def nA2(self):
        """
        returns oxidation state assigned to A (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).nA        
        if isinstance(self.chosen_ox_states, float):
            return np.nan
        else:
            return self.chosen_ox_states[self.A2]    
        
    @property
    def nA(self):
        """
        returns predicted Shannon ionic radius for B (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).nA
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:      
            return np.mean([self.nA1, self.nA2])    
    
    @property
    def nB1(self):
        """
        returns oxidation state assigned to A (int)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).nB          
        if isinstance(self.chosen_ox_states, float):
            return np.nan
        else:
            return self.chosen_ox_states[self.B1]
        
    @property
    def nB2(self):
        """
        returns oxidation state assigned to A (int)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).nB        
        if isinstance(self.chosen_ox_states, float):
            return np.nan
        else:
            return self.chosen_ox_states[self.B2]
        
    @property
    def nB(self):
        """
        returns predicted Shannon ionic radius for B (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).nB        
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:      
            return np.mean([self.nB1, self.nB2])
        
    @property
    def nX1(self):
        """
        returns oxidation state assigned to X1 (int)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).nX        
        if isinstance(self.chosen_ox_states, float):
            return np.nan
        else:
            return self.X_ox_dict[self.X1]    
        
    @property
    def nX2(self):
        """
        returns oxidation state assigned to X2 (int)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).nX        
        if isinstance(self.chosen_ox_states, float):
            return np.nan
        else:
            return self.X_ox_dict[self.X2]          

    @property
    def nX(self):
        """
        returns predicted Shannon ionic radius for X (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).nX       
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:      
            return np.mean([self.nX1, self.nX2])            
    
    @property
    def rA1(self):
        """
        returns predicted Shannon ionic radius for A (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).rA             
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:        
            return self.AB_radii_dict[self.A1]['A_rad']
        
    @property
    def rA2(self):
        """
        returns predicted Shannon ionic radius for A (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).rA             
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:        
            return self.AB_radii_dict[self.A2]['A_rad']

    @property
    def rA(self):
        """
        returns predicted Shannon ionic radius for B (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).rA        
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:      
            return np.mean([self.rA1, self.rA2])        
        
    @property
    def rB1(self):
        """
        returns predicted Shannon ionic radius for B (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).rB           
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:        
            return self.AB_radii_dict[self.B1]['B_rad']
        
    @property
    def rB2(self):
        """
        returns predicted Shannon ionic radius for B' (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).rB           
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:        
            return self.AB_radii_dict[self.B2]['B_rad']       
    
    @property
    def rB(self):
        """
        returns predicted Shannon ionic radius for B (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).rB        
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:      
            return np.mean([self.rB1, self.rB2])
    
    @property
    def rX1(self):
        """
        returns Shannon ionic radius for X (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).rX         
        if self.X1 != 'N':
            return Shannon_dict[self.X1][self.X_ox_dict[self.X1]][6]['only_spin']
        else:
            return Shannon_dict[self.X1][self.X_ox_dict[self.X1]][4]['only_spin']
    
    @property
    def rX2(self):
        """
        returns Shannon ionic radius for X' (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).rX          
        if self.X2 != 'N':
            return Shannon_dict[self.X2][self.X_ox_dict[self.X2]][6]['only_spin']
        else:
            return Shannon_dict[self.X2][self.X_ox_dict[self.X2]][4]['only_spin']
    
    @property
    def rX(self):
        """
        returns predicted Shannon ionic radius for X (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).rX        
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:
            return np.mean([self.rX1, self.rX2])    
    
    @property
    def mu(self):
        """
        returns the predicted octahedral factor (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).mu      
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:          
            return self.rB / self.rX
    
    @property
    def t(self):
        """
        returns the predicted Goldschmidt tolerance factor (float)
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).t        
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:          
            return (self.rA + self.rX) / (np.sqrt(2) * (self.rB + self.rX))
        
    @property
    def t_pred(self):
        """
        returns 1 if perovskite or -1 if nonperovskite by t
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).t_pred        
        if math.isnan(self.t):
            return np.nan
        else:
            return [1 if (self.t > 0.825) and (self.t < 1.059) else -1][0]
        
    @property
    def tau(self):
        """
        returns tau
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).tau      
        if isinstance(self.AB_radii_dict, float):
            return np.nan
        else:
            if self.rA <= self.rB:
                return np.nan
            else:
                return ((1/self.mu) - (self.nA)**2 + (self.nA) * (self.rA/self.rB)/(np.log(self.rA/self.rB)))
            
    @property
    def tau_pred(self):
        """
        returns 1 if perovskite or -1 if nonperovskite by tau
        """
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).tau_pred        
        if math.isnan(self.tau):
            return np.nan
        else:
            return [1 if self.tau < 4.18 else -1][0]
        
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
        if math.isnan(self.tau):
            return np.nan        
        if self.is_single == 1:
            CCX3 = ''.join(self.As + self.Bs + self.Xs + ['3'])
            return PredictABX3(CCX3).tau_prob(clf) 
        else:
            X = [[self.tau]]
            return clf.predict_proba(X)[0][1]             

   
            
def main():
    CCX3 = 'TiTaO3'
    A1, A2, B1, B2, X1, X2 = 'Pb', 'Pb', 'Mg', 'Te', 'O', 'O'
    single_obj = PredictABX3(CCX3)
    double_obj = PredictAABBXX6(A1, A2, B1, B2, X1, X2)
    props = ['A', 'B',
            'nA', 'nB', 'nX',
            'rA', 'rB', 'rX',
            't', 't_pred',
            'tau', 'tau_pred']
    return single_obj, double_obj
    
if __name__ == '__main__':
    single_obj, double_obj = main()
    
