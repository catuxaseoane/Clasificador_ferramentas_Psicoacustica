# -*- coding: utf-8 -*-

import argparse
import logging
import os
import pickle
import timeit
import warnings


from metodos2 import Reader
from metodos2.caracteristicas import Caracteristicas
from metodos2.majority_voter import MajorityVoter

from metodos2.ferramenta_estado_predictor import FerramentaPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path_data',
                        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--load_path_model',
                        default='{}/../../resultados/modelo/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--save_path',
                        default='{}/../../resultados/prediccion/'.format(os.path.dirname(os.path.abspath(__file__))))
    
    #parser.add_argument('--file_name', default='Grabacion_catu.wav')
    #parser.add_argument('--file_name', default='afN5_R_Ux.wav')
    parser.add_argument('--file_name', default='buena1.wav')

    parser.add_argument('--log_path',
                        default='{}/../'.format(os.path.dirname(os.path.abspath(__file__))))

    # Arguments
    args = parser.parse_args()
    load_path_data = os.path.normpath(args.load_path_data)
    load_path_model = os.path.normpath(args.load_path_model)
    file_name = args.file_name
    save_path = os.path.normpath(args.save_path)
    log_path = os.path.normpath(args.log_path)

     # Set up logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S %p',
                        filename=os.path.join(log_path, 'logs_prediction_test_test_model.log'),
                        filemode='w',
                        level=logging.INFO)

    # READ RAW SIGNAL

    logging.info('Reading {0}'.format(file_name))
    start = timeit.default_timer()

    # Read signal 
    file_reader = Reader(os.path.join(load_path_data, file_name))

    play_list,sample_rate = file_reader.read_audio_file()

    #stop = timeit.default_timer()
    #logging.info('Time taken for reading file: {0}'.format(stop - start))

    # FEATURE ENGINEERING

    #logging.info('Starting caracteristicas')
    #start = timeit.default_timer()

    # Feature extraction
    caracteristica = Caracteristicas()

    play_list_processed = list()

    #for audio_file in play_list:
        # Instantiate FeatureEngineer
        
        #print(type(audio_file))
    tmp = caracteristica.caracteristicas(audio_data=play_list,sample_rate=sample_rate)
    play_list_processed.append(tmp)

    stop = timeit.default_timer()
    logging.info('Time taken for feature engineering: {0}'.format(stop - start))

    # MAKE PREDICTION

    #logging.info('Predicting...')
    #start = timeit.default_timer()

    # https://stackoverflow.com/questions/41146759/check-sklearn-version-before-loading-model-using-joblib
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)

      with open((os.path.join(load_path_model, 'model.pkl')), 'rb') as fp:
          model = pickle.load(fp)

    predictor = FerramentaPredictor(model)

    predictions = list()

    for signal in play_list_processed:
        tmp = predictor.classify(signal[0])
        predictions.append(tmp)

    # MAJORITY VOTE

    majority_voter = MajorityVoter(predictions)
    majority_vote = majority_voter.vote()

    stop = timeit.default_timer()
    #logging.info('Time taken for majority vote: {0} '.format(stop - start, majority_vote))
    #logging.info('Time taken forall the prediction: {0} '.format(stop - start))


    # SAVE

    logging.info('Saving prediction...')

    # Save prediction result
    with open(os.path.join(save_path, 'prediction.txt'), 'wb') as text_file:
        text_file.write("{}".format(majority_vote).encode('utf-8'))

    logging.info('Saved! {}'.format(os.path.join(save_path, 'prediction.txt')))


if __name__ == '__main__':
    main()
