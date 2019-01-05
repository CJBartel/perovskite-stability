# perovskite-stability

Repository associated with https://arxiv.org/abs/1801.07700 used to predict the stability of perovskites given composition


This repository is mostly static to coincide with the cited paper. For updates since this paper, please see https://github.com/CJBartel/compmatscipy which contains a similar (and more up-to-date module).


Also, a GUI utility that implements this code is available at https://analytics-toolkit.nomad-coe.eu/home/ .


## Data files from manuscript

  ### TableS1.csv 
    
    576 experimentally characterized ABX3 solids with classifications

  ### TableS2.csv

    comparison of tau predictions to calculated decomposition enthalpies
  
  ### TableS3.csv

    classification of Cs2BB'Cl6 and MA2BB'Br6 compounds

  
## Data files needed for classification
    
  ### electronegativities.csv

    file of elemental electronegativities

    
  ### Shannon_Effective_Ionic_Radii.csv

    file for extracting Shannon radii 
    
      inorganic cations from v.web.umkc.edu/vanhornj/Radii.xls - adapted from https://doi.org/10.1107/S0567739476001551
      Sn2+ from 10.1039/C5SC04845A 
      organic ions from 10.1039/C4SC02211D

    
  ### TableS1.csv

    needed for yielding tau probabilities (see above)


## Scripts imported for classification

  ### make_radii_dict.py
    converts Shannon_Effective_Ionic_Radii.csv into dictionary to be imported
  
  ### PredictPerovskites.py
    contains classes for classifying single and double perovskites
  
        PredictABX3(object) (input CC'X3; output A, B, X, nA, nB, nX, rA, rB, rX, t, tau, t_prediction, tau_prediction, tau_probability)
  
        PredictAABBXX6(object) (input A1, A2, B1, B2, X1, X2; output nA, nB, nX, rA, rB, rX, t, tau, t_prediction, tau_prediction, tau_probability)
  
## Tutorial scripts

  ### classify_CCX3_demo.ipynb 
    
    standalone demo of CC'X3 -> classification by tau
    
  ### classify_list_of_formulas.ipynb
    
    add classification to pandas DataFrame using classes in PredictPerovskites.py
    
## Script for re-creating manuscript data

  ### regenerate_supporting_tables.ipynb
  
## Data file generated in tutorial

  ### classified_formulas.csv
  
    file containing output from "classify_list_of_formulas.ipynb"
