#pynt

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
