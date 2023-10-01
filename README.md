# MSc Thesis on Geometric Deep Learning

## Predicting bioactivity of small molecules in a diverse range of assays

## Preparing dataset
activity_matrix.ipynb
- generates the activity matrix from the assay and compound bioactivity data collected from PubChem bioassays as in (Helal et al., 2016).
grover_fingerprints.ipynb
- generates GROVER fingerprints for the compounds in the activity matrix using the pre-trained GROVER_large graph transformer model
create_dataset.ipynb
- creates chemical fingerprints and graph embeddigns for the compounds in the activity matrix and combines those with GROVER fingerprints to get 'data_list'. 'data_list' is a list of data objects for each compound that is used as the dataset for all experiments.

Data files required to run the scripts above are not published on GitHub at this time due to size constraints

## Running experiments

### Generic

main.py
- this is the most high-level file that can be used to run the full pipeline with the same presets as used in the thesis: just specifiy which assays and press run
- performs both hyperparameter search to find the best model. It then also fetches the best model found and pre-trained and finishes training for an additional custom number of epochs

### More custom
sweep_main.py
- can be used to perform a hyperparameter search to find the best model for an arbitrary assay and model pairing
finish_train_main.py
- can be used to fetch the best model found and pre-trained in sweep_main.py and finish training for an additional custom number of epochs
- can also be used to run training for any model from scratch

## Models

### Baselines

- 'LogReg': logistic regression on the chemical fingerprints
- 'RF': random forest classification on the chemical fingerprints
- 'GCN_base': graph embedding followed by a single classification layer

### Main

- 'FP': fingerprints embedding followed by MLP classification
- 'GROVER': graph transformer embedding followed by MLP classification
- 'GROVER_FP': graph transformer + fingerprints embedding followed by MLP classification
- 'GCN': graph embedding followed by MLP classification
- 'GCN_FP': graph + fingerprints embedding followed by MLP classification
- 'GCN_FP_GROVER': graph + fingerprints + graph transformer embedding followed by MLP classification
