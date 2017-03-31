Kaggle seizure prediction
=========================

 

Kaggle competition:
<https://www.kaggle.com/c/melbourne-university-seizure-prediction>

Seizure forecasting in humans with epilepsy. Distinguishing interictal and
preictal states given ten-minutes EEG records.

 

Autors
------

Tomasz Tylec, Anna Przysiezna, Maciej Jaskowski

 

Requirements
------------

It is assumed that `data/` directory is sym-linked to the kaggle contest data.

 

Data is stored in .mat files

-   I_J_K.mat - training data file, K=0(1) for interictal(preictal), I=1,2,3 is
    the number of the patient, J is the number of the sample

-   I_J.mat - test data file, I=1,2,3 is the number of the patient, J is the
    number of the sample

 

Each file contains:

`data.Struct` - 240000 x 16 array = 10 min EEG signal sampled with 400Hz
frequency on 16 electrodes

 

Usage
-----

 

### **Computing features**: `compute-features.py`

 

-   **training data set**

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ python compute-features.py train_data_dir -f padaczka.featuresFFT -t  -o train_features.csv -n scaler.pkl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute features for all files in `train_data_dir` and save them in one file,
`train_features.csv`. Features are standarized, StandardScaler is calculated for
all features and saved in `scaler.pkl`. -t option will include target value as a
first attribute (prepare data for training).

 

-   **validation data set**

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ python compute-features.py validation_data_dir -f padaczka.featuresFFT -t -s  -o  validation_features_dir -n scaler.pkl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute features for all files in `validation_data_dir` and save them into
separate csv files under `validation_features_dir` directory. Features are
standarized with respect to StandardScaler calculated for training data, passed
in `scaler.pkl` file. -t option includes target value as the first attribute

 

-   **test data set**

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ python compute-features.py test_data_dir -f padaczka.featuresFFT -s  -o  test_features_dir -n scaler.pkl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute features for all files in `test_data_dir` and save them into separate
csv files under `test_features_dir` directory. Features are standarized with
respect to StandardScaler calculated for training data, passed in `scaler.pkl`
file.

 

 

### **Training and predicting**: `train-test-predict.py`

 

-   **training**

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ python train-test-predict.py train_features.csv --train padaczka.classifier_svm  -p "C=10,gamma=0.01,kernel='rbf'" -c seizures.clf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train a classifier from the module `padaczka.classifier_svm` and save it to
`seizures.clf` file. Features are taken from the single csv file
`train_features.csv` .

Parameters can be passed to the classifier by -p option. `-p params` will be
converted to dict using`eval(dict(params))` and passed to the constructor of
classifier.

 

-   **testing**

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ python train-test-predict.py validation_features_dir --test -c seizures.clf -r roc.png -o test.csv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test classifier from `seizures.clf` on data from `validation_features_dir` and
print the report (confusion matrix) on stdout. ROC curve can be plotted and
saved to `roc.png` file. Seizure probabilities for each validation file can be
saved with `-o` option in `test.csv` file.

 

-   **predicting**

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ python train-test-predict.py test_features_dir --predict -c seizures.clf -o predictions.csv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate `predictions.csv` file with seizure probabilities for the test data
given in `test_data_dir` calculated with `seizures.clf` classifier.

 

Comments
--------

We obtain the best results when the classifier is trained for each patient
separately.
