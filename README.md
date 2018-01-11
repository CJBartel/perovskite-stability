# perovskite-stability


*in progress*


Data files from manuscript:

  -TableS1.csv (576 experimentally characterized ABX3 solids with classifications)
  
  -TableS2.csv (comparison of tau predictions to calculated decomposition enthalpies)
  
  -TableS3.csv (classification of Cs2BB'Cl6 and MA2BB'Br6 compounds)


Data files needed for classification:

  -electronegativities.csv (list of elemental electronegativities)
  
  -Shannon_Effective_Ionic_Radii.csv (file for extracting Shannon radii, from v.web.umkc.edu/vanhornj/Radii.xls)
  
  -TableS1.csv (see above; needed for yielding tau probabilities)
  
  
Scripts for classification:

  -make_radii_dict.py (converts Shannon_Effective_Ionic_Radii.csv into dictionary to be imported)
  
  -PredictPerovskites.py (contains classes for classifying single and double perovskites)
  
	-PredictABX3(object) (input CC'X3; output A, B, X, nA, nB, nX, rA, rB, rX, t, tau, t_prediction, tau_prediction, tau_probability)
  
	-PredictAABBXX6(object) (input A1, A2, B1, B2, X1, X2; output nA, nB, nX, rA, rB, rX, t, tau, t_prediction, tau_prediction, tau_probability)
  
  
Script for re-creating manuscript data:

  -regenerate_supporting_tables.ipynb
  
  
Tutorial script

  -classify_CCX3_demo.ipynb (standalone demo of CC'X3 -> classification by tau)


Files no longer in use but potentially useful
	-see not_in_use/
