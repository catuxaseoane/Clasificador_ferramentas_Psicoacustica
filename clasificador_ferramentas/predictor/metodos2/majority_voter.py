# -*- coding: utf-8 -*-
# from __future__ import division

__all__ = [
    'MajorityVoter'
]


class MajorityVoter:
    
    def __init__(self, prediction_list):
        self.predictions = prediction_list
        
       

    def vote(self):
  

        if sum(self.predictions) > len(self.predictions)/2.5:
            return 1
        else:
            return 0
