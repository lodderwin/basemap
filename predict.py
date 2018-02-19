import pandas as pd
import numpy as np
import yahoo_reader as yr
import processing as ps
from sklearn.model_selection import RandomizedSearchCV, train_test_split, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint as sp_randint
import plots

#%%
# Get Stock Data
yr = yr.finance_data()
df = yr.getData()

#%%
# Process for predictions
df = ps.prepData(df)
df = ps.genFeatures(df)
df = ps.featureProcessing(df)
df = ps.genTargets(df)

#%%
# Define X and y
X = df[['open_sc','high_sc','low_sc','close_sc','volume_sc',
        'year_sc',
        'ce_month_x','ce_month_y','ce_dow_x','ce_dow_y','ce_doy_x','ce_doy_y',
        'high_low_sc','open_min1_sc','close_min1_sc','volume_min1_sc',
        'high_low_min1_sc','open_min5_sc','close_min5_sc','volume_min5_sc',
        'high_low_min5_sc','open_cagr_sc','close_cagr_sc','ticker_en']]
y = df['regressor_y']

#%%
# Split df
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.8,
                                                    shuffle=False)

#%%
# Set parameters for Random search
param_dist = {"max_depth": sp_randint(1,20),
              "n_estimators": sp_randint(1,20),
              "max_features": sp_randint(1, 20),
              "bootstrap":[True,False]
              }
    
# run randomized search
rs = RandomizedSearchCV(RandomForestRegressor(n_jobs=2), 
                        param_distributions=param_dist,
                        n_iter=50, 
                        n_jobs=4)
# Fit RS to model
rs.fit(X_train, y_train)
# get best model
rf = rs.best_estimator_
# Fit model to data
rf.fit(X_train, y_train)

#%%
plots.barFeatureImportance(feature_importances=rf.feature_importances_,
                           features=list(X.columns))

#%%
# Draw Histogram of errors on test and train
plots.dualHist(errors1 = rs.predict(X_test) - y_test, 
               errors2 = rs.predict(X_train) - y_train,
               label1 = 'test', 
               label2 = 'train', 
               title = 'Prediction Error: test vs train',
               xlabel = '$',
               hist_range = [-15,15])

#%%
title = 'Learning Curves (Random Forest Regression)'
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

plt = plots.plot_learning_curve(rf, title, X, y, cv=cv, n_jobs=5)

plt.show()
plt.close()

#%%
# Use classifier
from sklearn.ensemble import RandomForestClassifier

y = df['classification_y']

# Split df
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.8,
                                                    shuffle=False)

rc = RandomForestClassifier(n_jobs=2)

rc.fit(X_train, y_train)

