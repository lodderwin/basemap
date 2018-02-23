import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def barFeatureImportance(feature_importances : np.ndarray, features : list):
    """
    Uses sklearn RandomForestRegressor's feature importances to draw a horizontal bar chart
    ordered by feature importance
    
    Parameters
    --------
    feature_importances : numpy array of feature importances
    features : list of features
    
    """
    # Create numpy array sort index
    sort = feature_importances.argsort()
    # Apply that sort to feature importances and features
    feature_importances = feature_importances[sort]
    features = np.asarray(features)[sort]
    
    # Define number of bars, bar heights and y labels for horizotal bar plot
    bars = np.arange(len(features))
    heights = feature_importances
    
    # Plot horizontal bar chart for feature importances
    plt.rc('axes', axisbelow=True)
    plt.barh(bars, heights, align="center", alpha=0.8, color="#3F5D7D")
    plt.yticks(bars, features)
    # Remove padding at top and bottom of chart
    plt.margins(y=0)
    # Plot titles and labels
    plt.title("Feature Importance")
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.grid(axis='x', linestyle='--', linewidth=1)
    # Show plt
    plt.show()
    
def scatter(x, y, title :str, xlabel : str, ylabel :str):
    """
    """
    # Draw scatter plot of predicted vs actuals
    plt.plot(x, y, 
             color="#3F5D7D", linestyle='', marker='o', markersize=1, alpha=0.6)
    # Configure plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def errorHist(errors, title :str, xlabel : str, rmse=None, unit=None, hist_range=None):
    """
    """
    # Create histogram label
    if rmse:
        label = 'rmse: ' + rmse.astype(str) + unit
    else:
        label=None   
    # Plot Histogram   
    plt.hist(errors, bins=60, label=label,
             color="#3F5D7D", edgecolor='black', linewidth=0.5, normed=True, alpha=0.8,
             range=hist_range)
    # Configure plot
    plt.title(title)
    plt.ylabel('%')
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()
    
def dualHist(errors1, errors2, rmse1, rmse2, label1, label2, 
             title : str, xlabel : str, hist_range=None):
    """
    """
    # Draw Histogram of errors
    plt.hist(errors1, bins=100, label=label1 + ' (rmse: $' + rmse1.astype(str) + ')',
             color="#3F5D7D", edgecolor='black', linewidth=0.5, normed=True, alpha=0.6,
            range=hist_range)
    # Draw Hist of baseline errors
    plt.hist(errors2, bins=100, label=label2 + ' (rmse: $' + rmse2.astype(str) + ')',
             color='g',edgecolor='black', linewidth=0.5, normed=True, alpha=0.6,
            range=hist_range)
    # Configure plot
    plt.title(title)
    plt.ylabel('%')
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores = np.sqrt(train_scores * -1)
    test_scores = np.sqrt(test_scores * -1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    
    return plt