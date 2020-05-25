# Predicting Parkinson's Disease Symptom Severity and Medication State With Subject Specific Neural Network Ensembles

This is the repository for team EasedisPD's work on the 2020 [Biomarker and Endpoint Assessment to Track Parkinson's Disease Challenge (BEAT-PD)](https://www.synapse.org/#!Synapse:syn20825169/wiki/596118). The team was comprised of Jonathan Hampton (East Tennessee State University Undergraduate), Dr. Jeff Knisley, and Dr. Debra Knisley. 

## Methods Summary
The complete write up of the team's work can be found [here](https://www.synapse.org/#!Synapse:syn21784049/wiki/603034).

For each subject in the challenge, two data representations are used with a neural network ensemble to train and make predictions for each of the sub-challenges (medication state, dyskinesia severity, tremor severity). The neural network ensemble is depicted in the graph below.

![Ensemble Graph](https://i.imgur.com/ygX8VIt.png)




## Requirements
- NumPy

- Pandas

- Matplotlib

- Torch

- Torch Vision

- Tensorboard

- PyTorch Lightning

- Librosa

- Scikit Learn

- tqdm

- Numba

- Imbalance learn

- Pillow

- Jupyter

## Usage Instructions

Each jupyter notebook corresponds to a subchallenge of the BEAT PD DREAM Challenge.

- Within notebooks, set appropriate paths to BEAT PD data.

- Also set path_to_null_preds to the path of null_preds.csv (included).

Notebooks must remain in same directory as .py files.




### CPU Training

While not practical, it is possible to train on CPU.
In the jupyter notebooks in the model settings section, set gpus=None.
Also will have to remove .cuda() calls within the jupyter notebook.

## Development Environment

- Conda environment running Python 3.7.5 on Windows 10.

- Models for final predictions were trained on a 6GB GTX 1060
