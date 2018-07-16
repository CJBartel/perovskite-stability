import numpy as np
import pandas as pd
import json

df = pd.read_csv('Shannon_Effective_Ionic_Radii.csv')

df = df.rename(columns = {'OX. State': 'ox',
                          'Coord. #': 'coord',
                          'Crystal Radius': 'rcryst',
                          'Ionic Radius': 'rion',
                          'Spin State' : 'spin'})
    
df['spin'] = [spin if spin in ['HS', 'LS'] else 'only_spin' for spin in df.spin.values]

def get_el(row):
    ION = row['ION']
    if ' ' in ION:
        return ION.split(' ')[0]
    elif '+' in ION:
        return ION.split('+')[0]
    elif '-' in ION:
        return ION.split('-')[0]

df['el'] = df.apply(lambda row: get_el(row), axis = 1)

el_to_ox = {}
for el in df.el.values:
    el_to_ox[el] = list(set(df.ox.get((df['el'] == el)).tolist()))
 
ionic_radii_dict = {}
for el in el_to_ox:
    oxs = el_to_ox[el]
    ox_to_coord = {}
    for ox in oxs:
        coords = df.coord.get((df['el'] == el) & (df['ox'] == ox)).tolist()
        ox_to_coord[ox] = coords
        coord_to_spin = {}
        for coord in ox_to_coord[ox]:
            spin = df.spin.get((df['el'] == el) & (df['ox'] == ox) & (df['coord'] == coord)).tolist()
            coord_to_spin[coord] = spin
            spin_to_rad = {}
            for spin in coord_to_spin[coord]:
                rad = df.rion.get((df['el'] == el) & (df['ox'] == ox) & (df['coord'] == coord) & (df['spin'] == spin)).tolist()[0]
                spin_to_rad[spin] = rad  
                coord_to_spin[coord] = spin_to_rad
                ox_to_coord[ox] = coord_to_spin
    ionic_radii_dict[el] = ox_to_coord

spin_els = ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu']
starting_d = [4, 5, 6, 7, 8, 9]
d_dict = dict(zip(spin_els, starting_d))
for el in spin_els:
    for ox in ionic_radii_dict[el].keys():
        for coord in ionic_radii_dict[el][ox].keys():
            if len(ionic_radii_dict[el][ox][coord].keys()) > 1:
                num_d = d_dict[el] + 2 - ox
                if num_d in [4, 5, 6, 7]:
                    ionic_radii_dict[el][ox][coord]['only_spin'] = ionic_radii_dict[el][ox][coord]['HS']
                else:
                    ionic_radii_dict[el][ox][coord]['only_spin'] = ionic_radii_dict[el][ox][coord]['LS']
            elif 'HS' in ionic_radii_dict[el][ox][coord].keys():
                ionic_radii_dict[el][ox][coord]['only_spin'] = ionic_radii_dict[el][ox][coord]['HS']
            elif 'LS' in ionic_radii_dict[el][ox][coord].keys():
                ionic_radii_dict[el][ox][coord]['only_spin'] = ionic_radii_dict[el][ox][coord]['LS']
                
with open('Shannon_radii_dict.json', 'w') as f:
    json.dump(ionic_radii_dict, f)