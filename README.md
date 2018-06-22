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

- Run the following code in terminal/powershell:

```
git status
```

**Output of above status should look something like:**
```
nothing to commit, working tree clean
```

**If the output does not say _"nothing to commit"_ please contact your local git master**
