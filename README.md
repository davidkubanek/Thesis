# MSc Thesis on Geometric Deep Learning
Predicting bioactivity of small molecules.

- HTS_fingerprints.ipynb generates HTS fingerprints for molecules collected from PubChem bioassays as in (Helal et al., 2016).
- concerto.ipynb replicates the CONCERTO architecture applied to the HTSFP data generated in HTS_fingerprints.ipynb.
- run.py can be used to perform a single run for an arbitrary assay and model
- sweep_main.py can be used to perform a hyperparameter search to find the best model for an arbitrary assay and model pairing
- finish_train.py can be used to fetch the best model found and pre-trained in sweep_main.py and finish training for an additional custom number of epochs
- Data files required to run the scripts above are not published on GitHub at this time due to size constraints
