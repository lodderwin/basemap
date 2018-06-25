import yaml

from slackclient import SlackClient


token = yaml.load(open('./configs/slack_token.yml'))['token']

#%%

sc = SlackClient(token)

sc.api_call(
  'chat.postMessage',
  channel='slackbot_test',
  text="Hello from Python! :tada:",
  username='pyntbot'
)