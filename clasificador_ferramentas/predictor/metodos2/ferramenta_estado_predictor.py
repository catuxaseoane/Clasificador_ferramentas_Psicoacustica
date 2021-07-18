# -*- coding: utf-8 -*-

import re


__all__ = [
    'FerramentaPredictor'
]


class FerramentaPredictor:
   

    def __init__(self, model):
        self.model = model

    def classify(self, new_signal):
        

        category = self.model.predict(new_signal)

        
        return self._is_good(category[0])

    @staticmethod
    def _is_good(string):
       
        match = re.search('audios_GOOD', string)
        if match:
            return 1
        else:
            match = re.search('audios_BAD', string)
            if match:
                return -1
            else:
                return 0    
                
       
