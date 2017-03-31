#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
import shutil
import pickle
import glob
import importlib
import argparse
import scipy.io as sio
from tqdm import tqdm

from padaczka.common import data_sequences

from sklearn.preprocessing import StandardScaler


description = """
Framework for feature calculation from the EEG signal given in .mat files

There are three use cases:


Generate features for one file
==============================

    $ python compute-features.py data.mat -t -f padaczka.featuresFFT -s  -o  output_data.csv -n scaler.pkl

Creates output_data.csv file in a current directory with features computed for data.mat using module `padaczka.featuresFFT`. 
If the output file not specified by -o option, creates .csv file with the input filename (data.csv).

-t option will include target value as a first attribute (prepare data for training).

-n option will standarize the features with respect to StandardScaler passed in `scaler.pkl`  file.


Merge features from single files
================================

    $ compute-features.py data_dir -m 

Merges all csv files in `data_dir` into one file `data_dir.csv` (unless explicit -o is passed).


Compute features for all files in a directory
===========================================

    $ python compute-features.py data_dir -f padaczka.featuresFFT -t -n scaler.pkl

Computes features for all files in data_dir and saves them into a csv files speciffied (optionally) with -o option (`data_dir.csv` by default).

-s option allows to save features from different data files into separate csv files. 
If -o is passed, then generated csv files are under specified directory. 

Features are standarized, sklearn.preprocessing.StandardScaler  is calculated for all features and saved in scaler.pkl file.


Feature computation modules
===========================

Module passed to -f option is required to define a function `compute_features`
that takes a np.array of data and returns a single vector of features.
"""


def datafile_features(sfile, compute_features, sequencer=data_sequences,
                      include_target=False):
    fileid = os.path.splitext(os.path.basename(sfile))[0]
    preictal = fileid[-1] == '1'
    # load .mat file, return dictionary
    content = sio.loadmat(sfile, struct_as_record=False,
                          verify_compressed_data_integrity=False,
                          squeeze_me=True)['dataStruct']
    # tak data matrix
    data = content.data.astype(float)
    features = np.array(filter(lambda f: f.shape != (0,),
                               [compute_features(d) for d in sequencer(data)])) # d is a 50s subset of the data
    return np.insert(features, 0, int(preictal), axis=1) \
        if include_target and features.shape != (0,) else features


def save_feature_matrix(data, fn, include_target=False):
    if data.shape == (0,):
        return
    
    if include_target:
        columns = ['target'] + ['A' + str(i) for i in xrange(data.shape[1]-1)]
    else:
        columns = ['A' + str(i) for i in xrange(data.shape[1])]
        
        
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(fn, sep=';', index=False)


def default_csv_file(fn):
    return os.path.splitext(os.path.basename(fn))[0] + '.csv'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('src', help='source of data')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('-f', '--features', default=None,
                        help='module(s) computing features')
    action.add_argument('-m', '--merge', action='store_true',
                        help='assuming src is a dir, this option '
                        'will cause all csv files in src to be merged ')
    parser.add_argument('-s', '--separate', action='store_true',
                        help='even if src is a directory, store features '
                        'for each .mat file separately')
    parser.add_argument('-t', '--include-target', action='store_true',
                        help='include train target as a first attribute')
    parser.add_argument('-o', '--output', default='',
                        help='output file.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be more verbose')
    parser.add_argument('-n', '--scaler_file', default='scaler.pkl',
                        help='scaler file for feature normalization')
    args = parser.parse_args()
    
 
    # specify compute_features function     
    # if we only merge files with already computed features:
    if args.merge:                          
        compute_features = lambda sfile: pd.read_csv(sfile, sep=';').values   
    # if we need to compute features (do not merge):
    else:
        # import module that calculates features:
        flib = importlib.import_module(args.features) 
        compute_features = lambda sfile: \
                datafile_features(sfile, flib.compute_features,
                                    include_target=args.include_target)
                                    
                                    
    # compute features when the source is a single file:
    if os.path.isfile(args.src): 
        dst = default_csv_file(args.src) if args.output == '' else args.output
        # to do: add scaler!
        features = compute_features(args.src)
        save_feature_matrix(features, dst)
        if args.verbose:
            print("Features for {} computed and saved to {}."
                  .format(args.src, dst))
                  
    # compute features when the source is a catalog:               
    else:                       
        datafiles = (
                # all the csv files for merging: 
                glob.glob(os.path.join(args.src, '*.csv')) if args.merge 
                # all the mat files for computing features:
                else glob.glob(os.path.join(args.src, '*.mat')) 
        ) 
            
        # progress bar (tqdm - progress meter)    
        pbar = tqdm if args.verbose else list 
        

        # set the name of the output file or directory (dir_name.csv if the output is not specified):
        dst = os.path.normpath(args.src).split('/')[-1] + '.csv' \
            if (args.merge or not args.separate) and args.output == ''\
            else args.output
            
            
        # separate output files (for test files, scaled with the scaler calculated for train files):
        if args.separate:
            if dst != '':
                # remove the dst directory with all its contents
                shutil.rmtree(dst, ignore_errors=True) 
                # make (empty) output directory
                os.mkdir(dst)   

            do_compute_features = compute_features 
            
            
            def compute_features(sfile): 
                """ Define new compute_features function that also scales the features. """
                features = do_compute_features(sfile)
                pkl_file = open(args.scaler_file, 'rb')
                scaler = pickle.load(pkl_file)  #load scaler 
                pkl_file.close()
                
                if  features.ndim==2:
                    if args.include_target:
                        # scale the features:
                        features[:,1:] = scaler.transform(features[:,1:]) 
                    else:
                        features = scaler.transform(features)
                        
                    # save  separate files:
                    save_feature_matrix(features, os.path.join(dst,   default_csv_file(sfile)), args.include_target)
                        
                return features
                
        #compute the features      
        features = filter(lambda f: f.shape != (0,),
                          [compute_features(sfile)
                           for sfile in pbar(datafiles)])
                           
                           
        # one output file 
        #(features for all data samples in one place, rescaled -> scaler is calculated and saved)
        if not args.separate:
            output_sc = open(args.scaler_file, 'wb')
            all_features = np.concatenate(features)
            scaler = StandardScaler()
            #scale the features:
            all_features[:,1:] = scaler.fit_transform(all_features[:,1:]) 
            # save the scaler
            pickle.dump(scaler, output_sc)  
            output_sc.close()
            #save the features in one csv file
            save_feature_matrix(all_features, dst) 
