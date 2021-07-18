# -*- coding: utf-8 -*-
import argparse
import logging
import os
import re
import timeit
import numpy as np
import sys;
import array



from metodos import Reader
from metodos.caracteristicas import Caracteristicas





def main():
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        default='{}/../../audios'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--save_path',
                        default='{}/../../resultados/dataset/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--log_path',
                        default='{}/../'.format(os.path.dirname(os.path.abspath(__file__))))

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)
    save_path = os.path.normpath(args.save_path)
    log_path = os.path.normpath(args.log_path)

    # Set up logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S %p',
                        filename=os.path.join(log_path, 'logs_pc_methods_feat_eng.log'),
                        filemode='w',
                        level=logging.INFO)

    # READ FILES IN SUB-FOLDERS of load_path and FEATURE ENGINEERING

    # list load_path sub-folders
    regex = re.compile(r'^[0-9]')
    directory_list = [i for i in os.listdir(load_path) if regex.search(i)]

    # initialize empty array for features
    X = np.empty([1,24])
    X1 = np.empty([1,24])
    X2 = np.empty([1,24])
    X3 = np.empty([1,24])
    
    
    # initialise empty array for labels
    y = []

    logging.info('Creating training set...')
    start = timeit.default_timer()

    
    # iteration on sub-folders
    for directory in directory_list:

        # Instantiate FeatureEngineer
        caracteristica = Caracteristicas(label=directory)

        file_list = os.listdir(os.path.join(load_path, directory))

        # iteration on audio files in each sub-folder

        for audio_file in file_list:
            file_reader = Reader(os.path.join(load_path, directory, audio_file))
            data, sample_rate = file_reader.read_audio_file()
            
            avg_features, label , loudness_feat, sharpness_feat, roughness_feat= caracteristica.caracteristicas(audio_data=data, sample_rate=sample_rate)
            
            X = np.concatenate((X, avg_features), axis=0)
            X1=np.concatenate((X1, loudness_feat), axis=0)
            X2=np.concatenate((X2, sharpness_feat), axis=0)
            X3=np.concatenate((X3, roughness_feat), axis=0)

            y.append(label)

    
    X=X[1:, :]
    X1=X1[1:, :]
    X2=X2[1:, :]
    X3=X3[1:, :]



    
    
    

    stop = timeit.default_timer()
    logging.info('Time taken for reading files and feature engineering: {0}'.format(stop - start))

    # Save to numpy binary format
    logging.info('Saving training set...')
    np.save(os.path.join(save_path, 'dataset.npy'), X)
    np.save(os.path.join(save_path, 'datasetX1.npy'), X1)
    np.save(os.path.join(save_path, 'datasetX2.npy'), X2)
    np.save(os.path.join(save_path, 'datasetX3.npy'), X3)
    np.save(os.path.join(save_path, 'labels.npy'), y)

    logging.info('Saved! {0}'.format(os.path.join(save_path, 'dataset.npy')))
    logging.info('Saved! {0}'.format(os.path.join(save_path, 'labels.npy')))


    



if __name__ == '__main__':
    main()
