import os
import glob
import argparse
import pickle
import importlib
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd




description = """
Framework for model training, testing and predicting

Training
========

    $ python train-test-predict.py train_features.csv --train padaczka.classifier_svm  -p "C=10,gamma=0.01,kernel='rbf'" -c seizures.clf

Train a classifier from the module `padaczka.classifier_svm` and save it to `seizures.clf` file. Features are taken from the single csv file `train_features.csv` .
Parameters can be passed to the classifier by -p option. `-p params` will be converted to dict using`eval(dict(params))` and passed to the constructor of classifier. 


Testing
=======

    $ python train-test-predict.py validation_features_dir --test -c seizures.clf -r roc.png -o test.csv

Test classifier from `seizures.clf` on data from `validation_features_dir` and print the report (confusion matrix) on stdout. ROC curve can be plotted and saved to `roc.png`  file. Seizure probabilities for each validation file can be saved with `-o` option in `test.csv` file. 


Predicting
==========

    $ python train-test-predict.py test_data_dir --predict -c seizures.clf -o predictions.csv

Generate `predictions.csv`  file with seizure probabilities for the test data given in `test_data_dir` calculated with `seizures.clf` classifier. 
"""

report_template = """
p = {p:5.2f}        prediction
             +----------+----------+
             | positive | negative |
d +----------+----------+----------+
a | positive |  {tp:7d} |  {fn:7d} | TPR: {tpr:.3f}
t +----------+----------+----------+
a | negative |  {fp:7d} |  {tn:7d} | FPR: {fpr:.3f}
  +----------+----------+----------+

AUC: {auc:.3f}
"""


def test_dir(tdir, clf, output=None):
    """ Tests with respect to data files (one file
    per contest input file). """

    predictions = []
    pdict = {}
    for testfile in tqdm(glob.glob(os.path.join(tdir, '*.csv'))):
        db = pd.read_csv(testfile, sep=';')
        data = db.ix[:, 1:].values
        target = int(db.ix[0, 0])
        # extracting name of the file from the path
        filename = os.path.basename(testfile)
        predicted = clf.predict_proba(data)
        prob = predicted[:, 1].sum() / predicted.shape[0]

        predictions.append((target, prob))
        pdict[filename] = prob

    if output is not None:
        pd.DataFrame(pd.Series(pdict)).to_csv(output, header=False)

    return predictions


def predict_dir(tdir, clf, output):
    """ Tests with respect to data files (one file
    per contest input file). """

    pdict = {}
    for testfile in tqdm(glob.glob(os.path.join(tdir, '*.csv'))):
        db = pd.read_csv(testfile, sep=';')
        data = db.values
        # extract name of the file from the path
        filename = os.path.basename(testfile)

        predicted = clf.predict_proba(data)
        prob = predicted[:, 1].sum() / predicted.shape[0]

        pdict[filename] = prob

    pd.DataFrame(pd.Series(pdict)).to_csv(output, header=False)


def plot_roc_curve(fpr, tpr, roc_auc, rocfile='roc.png'):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(rocfile, dpi=300)


def report(predictions, ptreshold=[0.01, 0.10, 0.25, 0.5], rocfile='roc.png'):
    y_true, y_score = zip(*predictions)
    fpr, tpr, tr = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, rocfile=rocfile)

    target = pd.Series(y_true)
    for p in ptreshold:
        predicted = pd.Series(y_score) >= p

        tp = ((predicted == 1) & (target == 1)).sum()
        fp = ((predicted == 1) & (target == 0)).sum()
        fn = ((predicted == 0) & (target == 1)).sum()
        tn = ((predicted == 0) & (target == 0)).sum()

        print(report_template.format(tp=tp, fp=fp, fn=fn, tn=tn,
                                     tpr=float(tp)/(tp+fn),
                                     fpr=float(fp)/(fp+tn),
                                     auc=roc_auc, p=p))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('src')
    parser.add_argument('-c', '--classifier-file', action='store',
                        required=True, default='',
                        help='file to store/read classifier')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--train', action='store', default=None)
    action.add_argument('--test', action='store_true',
                        help='perform test of classifier')
    action.add_argument('--predict', action='store_true',
                        help='predict using classifier')
    parser.add_argument('-p', '--parameters', action='store',
                        default='',
                        help='additional parameters passed to classifier '
                        'constructor')
    parser.add_argument('-r', '--roc-file', action='store',
                        default='roc.png',
                        help='roc file name for --test action')
    parser.add_argument('-o', '--output_file', action='store',
                        default='predictions.csv',
                        help='output file name for predictions in --sub action')
    args = parser.parse_args()

    
    # train
    if args.train is not None: 
        clib = importlib.import_module(args.train)
        db = pd.read_csv(args.src, sep=';', low_memory=False)
        features = db.ix[:, 1:].values
        target = db.ix[:, 0].values.astype('int')
        clf = clib.Classifier(**eval('dict({})'.format(args.parameters)))
        clf.fit(features, target)
        with open(args.classifier_file, 'wb') as f:
            pickle.dump(clf, f)

    # test & predict
    else:
        # load the calssifier
        with open(args.classifier_file, 'rb') as f: 
            clf = pickle.load(f)

        # test
        if args.test:
            pred = test_dir(args.src, clf, args.output_file)
            report(pred, rocfile=args.roc_file)
            
        # predict
        elif args.predict:
            pred = predict_dir(args.src, clf, args.output_file)
