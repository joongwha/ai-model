# Training and Validation of an ECG-Based Deep-Learning Model for Myocardial Infarction Subtypes
This is the official repository for the analysis "_Training and validation of an ECG-based deep-learning model for myocardial infarction subtypes_" by the MALBEC Team at the University Hospital Basel, Switzerland.

This is the first 12-lead ECG-based deep learning model for the diagnosis of different subtypes of myocardial infarction (multi-label outcome). Prospective implementation of the model is needed to determine whether it can improve the diagnosis of myocardial infarction and patient care.

**Note**: This repository only contains inference code and pre-trained weights. Training code can be made available upon reasonable request to the corresponding authors.


## Directory Layout:
```
training-and-validation-of-a-12-lead-ecg-based-deep-learning-model-for-myocardial-infarction-subtypes/
├── LICENSE
├── README.md
├── data
│   ├── ecgs_dummy.h5
│   └── outcome_dummy.csv
├── environment.yml
├── models
│   ├── finetuned_18d5a8abc508439a82786289b8f67ccc.ckpt (on Zenodo)
│   ├── model_non_hd.pkl
│   ├── model_nstemi_hd.pkl
│   ├── model_nstemi_type1_hd.pkl
│   ├── model_nstemi_type2_hd.pkl
│   ├── model_other_hd.pkl
│   └── model_stemi_hd.pkl
└── src
    ├── ecg_processing.py
    ├── h5dataset.py
    ├── hparams.py
    ├── model.py
    ├── predict_recalibrated.ipynb
    └── recalibrate.py
```

## Predictions
`python 3.11.9` was used. Install the requirements using

```bash
conda env create -f environment.yml
```
You can run inference in the notebook `predict_recalibrated.ipynb` 

Furthermore, the non-calibrated implementation is included in `predict.ipynb`

## Dataset
We provide one representative ECG for each myocardial infarction subtype: STEMI, OMI, NSTEMI type 1, and NSTEMI type 2. Each ECG is 10 seconds long, sampled at 4096 datapoints, with an amplitude resolution of 1 µV/bit.


## Licensing
This project is licensed under the GPL 3.0 Licence - See LICENSE for more details.
