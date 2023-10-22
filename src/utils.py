from __future__ import annotations
from copy import deepcopy
import pandas as pd
import numpy as np





def enable_dropout(model):
    """Enables dropout at test time - for Monte Carlo Dropout"""
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()
    return model

def seed_everything(seed: int):
    """Seed all"""
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_models(X_train, model_dict):
    """
    Trains multiple models using the given training data and returns a
    dictionary of trained models.
    
    Args:
      X_train: X_train is a pandas DataFrame containing the training data. It should have the features
    as columns and the target variable 'y' as a separate column.
      model_dict: The `model_dict` parameter is a dictionary where the keys are model names and the
    values are instances of those models. Each model instance should have a `fit` method that takes in
    the training data and labels as arguments.
    
    Returns:
      a dictionary containing trained models.
    """
    trained_model_dict = {}

    for model in model_dict.keys():
        clf = model_dict[model]
        clf.fit(X_train.drop('y', axis=1), X_train['y'])

        trained_model_dict[model] = deepcopy(clf)

    return trained_model_dict




def bootstrap(y_true: pd.DataFrame | np.array, y_pred: pd.DataFrame | np.array, metric, R=10000):
    """
    The function `bootstrap` performs bootstrap resampling on the true and predicted values and
    calculates a metric for each resampled dataset.
    
    Args:
      y_true (pd.DataFrame | np.array): The parameter `y_true` represents the true values of the target
    variable. It can be either a pandas DataFrame or a numpy array.
      y_pred (pd.DataFrame | np.array): The `y_pred` parameter represents the predicted values for a
    target variable. It can be either a pandas DataFrame or a numpy array.
      metric: The "metric" parameter is a function that calculates a performance metric between the true
    values (y_true) and the predicted values (y_pred). This function can be any valid performance
    metric, such as mean squared error (MSE), mean absolute error (MAE), or any other custom metric you
      R: The parameter R represents the number of bootstrap resamples to perform. It determines how many
    times the resampling process will be repeated to obtain different samples from the original data.
    Defaults to 10000
    
    Returns:
      a DataFrame containing the scores obtained from resampling the true and predicted values and
    calculating the metric for each resampled set.
    """
 
    y = np.concatenate([y_true, y_pred], axis=1)

    n = len(y)
    scores = []
    for i in range(R):
        idx = np.random.choice(n, n, replace=True)
        y_resampled = y[idx]
        y_true_resampled = y_resampled[:, 0]
        y_pred_resampled = y_resampled[:, 1]
        score = metric(y_true_resampled, y_pred_resampled)
        scores.append(score)
    
    scores = pd.DataFrame(scores)
    return scores


def confidence_intervals(scores, confidence_level=0.95): 
    """
    Calculates the confidence intervals for a given set of scores at a specified confidence level.
    
    Args:
      scores: The "scores" parameter is a list or array of numerical values for which you want to
    calculate confidence intervals.
      confidence_level: The confidence level is the probability that the true population parameter falls
    within the calculated confidence interval. It is typically expressed as a decimal between 0 and 1.
    For example, a confidence level of 0.95 corresponds to a 95% confidence interval.
    
    Returns:
      the bottom percentile and top percentile of the scores, which represent the confidence interval.
    """
    
    low_end = (1 - confidence_level)/2
    high_end = 1 - low_end
    bottom_percentile = scores.quantile(low_end)
    top_percentile = scores.quantile(high_end)
 
    return bottom_percentile, top_percentile