# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:30:05 2018

@author: Frank
"""

import os

directory = './shortterm_models/'
output = './temp/modelcount.txt'

models = str(len([name for name in os.listdir(directory)]))

with open(output, 'w') as f:
  f.write(models)