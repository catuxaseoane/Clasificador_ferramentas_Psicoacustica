# -*- coding: utf-8 -*-

import numpy as np
import logging
import timeit
import soundfile as sf
import pyloudnorm as pyln



from pyloudnorm.meter import Meter
from timbral_models import timbral_util
from roughness_danielweber.comp_roughness import comp_roughness
#from /../timbral_models/ import Timbral_Roughness

# Add MOSQITO to the Python path
import sys
sys.path.append('..')

# Import useful packages
import math
import numpy as np
import matplotlib.pyplot as plt




__all__ = [
    'Caracteristicas'
]


class Caracteristicas:
  

    def __init__(self, label=None):
        # Necessary to re-use code for training and prediction
        if label is None:
            self.label = ''
        else:
            self.label = label

    def caracteristicas(self, audio_data,sample_rate):
        
        
        loudness_feat=self.compute_loudness(audio_data=audio_data,sample_rate=sample_rate, feat_name='loudness')
        sharpness_feat=self.compute_sharpness(audio_data=audio_data,sample_rate=sample_rate, feat_name='sharpness')
        roughness_feat=self.compute_roughness(audio_data=audio_data,sample_rate=sample_rate, feat_name='roughness')
        
        concat_feat = np.concatenate((loudness_feat,
                                    sharpness_feat,
                                    roughness_feat
                                    ), axis=0)

        logging.info('Comezando coa extracción de características')
        start = timeit.default_timer()
       
        mean_feat = np.mean(concat_feat, axis=0, keepdims=True)
        loudness_feat=np.mean(loudness_feat, axis=0, keepdims=True)
        sharpness_feat=np.mean(sharpness_feat, axis=0, keepdims=True)
        roughness_feat=np.mean(roughness_feat, axis=0, keepdims=True)

        

        stop = timeit.default_timer()
        
        logging.info('Time taken: {0}'.format(stop - start))

        
        return mean_feat, self.label

    
################################################## LOUDNESS ##################################################
    def compute_loudness(self, audio_data, sample_rate, feat_name):    
          
        if feat_name == 'loudness':
                
                meter=pyln.Meter(sample_rate,block_size=0.060)
                
                loudness_result=Meter.integrated_loudness(meter,audio_data)
                return loudness_result

################################################## SHARPNESS ##################################################
              
    def compute_sharpness(self, audio_data, sample_rate, feat_name):
       
        # window the audio file into XXXXXX sample sections
        windowed_audio = timbral_util.window_audio(audio_data, window_length=830)

        windowed_sharpness = []
        windowed_rms = []
        
        for i in range(windowed_audio.shape[0]):
            samples = windowed_audio[i, :]
            
            # calculate the rms and append to list
            windowed_rms.append(np.sqrt(np.mean(samples * samples)))

            # calculate the specific loudness
            N_entire, N_single = timbral_util.specific_loudness(samples, Pref=100.0, fs=sample_rate, Mod=0)
            
            # calculate the sharpness if section contains audio
            if N_entire > 0:
                sharpness = self.sharpness_Fastl(N_single)
            else:
                sharpness = 0

            windowed_sharpness.append(sharpness)

        # convert lists to numpy arrays for fancy indexing
        windowed_rms = np.array(windowed_rms)
        windowed_sharpness = np.array(windowed_sharpness)
        
       
        windowed_sharpness=np.array(windowed_sharpness).transpose() 
        windowed_sharpness=windowed_sharpness.reshape(1,24)

       
        return windowed_sharpness
    
    def sharpness_Fastl(self,loudspec):
    
        n = len(loudspec)
        gz = np.ones(140)
        z = np.arange(141,n+1)
        gzz = 0.00012 * (z/10.0) ** 4 - 0.0056 * (z/10.0) ** 3 + 0.1 * (z/10.0) ** 2 -0.81 * (z/10.0) + 3.5
        gz = np.concatenate((gz, gzz))
        z = np.arange(0.1, n/10.0+0.1, 0.1)

        sharp = 0.11 * np.sum(loudspec * gz * z * 0.1) / np.sum(loudspec * 0.1)
        return sharp

################################################## ROUGHNESS ##################################################
  

    def compute_roughness(self, audio_data, sample_rate, feat_name):
        

        roughness=comp_roughness(audio_data,sample_rate,overlap=0.005)
        roughness2=roughness.get('values')

        roughness_numpy = np.array(roughness2)
        roughness_numpy=np.array(roughness_numpy).transpose() 
        roughness_numpy=roughness_numpy.reshape(1,24)
        
      
        
        return roughness_numpy



   