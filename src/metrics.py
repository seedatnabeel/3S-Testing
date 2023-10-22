from typing import Any, Tuple, Type
# synthcity absolute
from synthcity.metrics.eval_statistical import (
    AlphaPrecision,
    InverseKLDivergence,
    JensenShannonDistance,
    MaximumMeanDiscrepancy,
    PRDCScore,
    WassersteinDistance,
)
from synthcity.plugins import Plugin, Plugins
from synthcity.plugins.core.dataloader import (
    DataLoader,
    GenericDataLoader,
    create_from_info,
)

import pandas as pd
import numpy as np


def eval_plugin(
    evaluator_t: Type, X: DataLoader, X_syn: DataLoader, **kwargs: Any
) -> Tuple:
    
    """Computes Synthetic Data Metrics"""
    evaluator = evaluator_t(**kwargs)

    syn_score = evaluator.evaluate(X, X_syn)

    sz = len(X_syn)
    X_rnd = create_from_info(
        pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns), X.info()
    )
    rnd_score = evaluator.evaluate(
        X,
        X_rnd,
    )

    return syn_score, rnd_score


def perf_measure(y_actual, y_hat):
    """
    The `perf_measure` function calculates the true positive rate, false positive rate, true negative
    rate, and false negative rate for a binary classification model.
    
    Args:
      y_actual: The actual values of the target variable (ground truth).
      y_hat: The predicted values for the target variable.
    
    Returns:
      The function `perf_measure` returns four values: the true positive rate (TPR), false positive rate
    (FPR), true negative rate (TNR), and false negative rate (FNR).
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP/len(y_hat), FP/len(y_hat), TN/len(y_hat), FN/len(y_hat))


def compute_interval_metrics(lb, ub, true):
    """
    The function computes interval metrics such as excess, deficit, and width based on lower bound (lb),
    upper bound (ub), and true values.
    
    Args:
      lb: The lower bound of the interval. It represents the minimum value that the true value can take.
      ub: The parameter "ub" stands for the upper bound of the interval.
      true: The true value is the actual value that we are comparing to the lower bound (lb) and upper
    bound (ub).
    
    Returns:
      three values: excess, deficit, and width.
    """

    if true >= lb and true <= ub:
        excess = (np.min([true - lb, ub - true]))
    else:
        excess = -9999

    if true <= lb or true >= ub:
        deficet =  np.min([np.abs(true - lb), np.abs(true - ub)])            
    else:
        deficet = -9999

    width = np.mean(abs(ub - lb))

    return excess, deficet, width


