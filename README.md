## pynt

### Important

**Any time you make changes to "industry_owners.csv" commit and push changes immediately.**


## Training Models

- Run the following in terminal/powershell:

```
git pull
```
**Always git pull before running or changing any .py file.**

- Create a folder called _"configs"_ in the main directory.

- In configs folder create user_settings.yml file with the following content:

```
user: insert_your_name_here (erwin, frank or howard) 
```

- Claim some industries by opening _"industry_owners.csv"_ and adding a row with industry and your name, **do not** use capitals.

- Run _"CreateModels.py"_ and wait for program to finish. Ending midway through is also an option.

- Run the following in terminal/powershell:

```
git add results/*
git add investment_sim/* 
git add shortterm_models/*
git commit -m 'adding latest training results'
git push
```

- Run the following code in terminal/powershell:

```
git status
```

**Output of above status should look something like:**

```
nothing to commit, working tree clean
```

**If the output does not say _"nothing to commit"_ please contact your local git master**

## Slack Token
latest.py uses the slack api to send a notification to the pynt workspace. This means you will need to create a _"Legacy Token"_. Do so at the following url: [https://api.slack.com/custom-integrations/legacy-tokens](https://api.slack.com/custom-integrations/legacy-tokens)

Save you token in a file called ./configs/slack_token.yml. The file should look like the following:

```
token: insert_token_here
```

## Daily Predictions
Make the day's predictions using _latest.py_. Running this file will make predictions for all models available. The results will be shared via slack.

Once you have run _latest.py_ you will need to commit some files to git. Run the following code in the terminal.

```
git add predictions/*
git commit -m 'adding latest prediction results'
git push
```

Then run the following in the terminal:

```
git status
```

**If the output does not say _"nothing to commit"_ please contact your local git master**