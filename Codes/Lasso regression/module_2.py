# Functions:
# 1. plot_curve(..) --- developed by Nilotpal Chakraborty

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from pathlib import Path

## Function for plotting learning curves of a linear regression model 
def plot_curve(X,Y,alpha, outdir):
    size = 5
    cv = KFold(size, shuffle=True)
    ## Lasso regression model
    mlr = Lasso(alpha=alpha)   
    mlr.fit(X, Y)
    
    train_sizes, train_scores, test_scores = learning_curve(mlr, X, Y, n_jobs=-1, cv=cv, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.title("Lasso regression")
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()
    
    # box-like grid
    plt.grid()
    
    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.gca().legend(("Training score","Cross-validation score"))
    # sizes the window for readability and displays the plot
    # shows error from 0 to 1.1
    plt.ylim(-.1,1.1)
    plt.savefig(Path(outdir+"learning_curve.png"), dpi=300)
    plt.show()
    
   
    
    
