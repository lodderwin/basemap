# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:22:20 2018

@author: Frank
"""

import yaml
import os
from slackclient import SlackClient

TOKEN = yaml.load(open('./configs/slack_token.yml'))['token']
SC = SlackClient(TOKEN)

countfile = './temp/modelcount.txt'
#%%

#import old count from countfile and calculate number of new models
lastcount = int((open(countfile, 'r')).read())
directory = directory = './shortterm_models/'
models = len([name for name in os.listdir(directory)])
newmodels = models - lastcount

#slackbot message
msg = str(newmodels) + ' new models have been pushed. There are now ' + str(models) + ' models in git.'
SC.api_call(
    'chat.postMessage',
     channel='training',
     text=msg,
     username='pyntbot',
     footer='pyntbot')

#write new model count to count file
with open(countfile, 'w') as f:
  f.write(str(models))