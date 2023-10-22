import numpy as np
from scipy.stats import gaussian_kde


def rejection_sample(D, mean, std, feat_id=[0]):
    """
    Performs rejection sampling to generate a shifted dataset based on a
    source dataset, mean, and standard deviation.
    
    Args:
      D: D is the source dataset, which is a pandas DataFrame. It contains the original data that we
    want to sample from.
      mean: The mean parameter represents the mean value of the target distribution that you want to
    shift your data towards. It is used to generate the target dataset in the function.
      std: The `std` parameter represents the standard deviation of the target distribution. It is used
    to generate the target dataset when `manual_shift` is set to False.
      feat_id: The `feat_id` parameter is a list that specifies the feature(s) to condition on. It can
    be a single feature or a list of multiple features. The function will generate a shifted
    distribution based on these features.
    
    Returns:
      the shifted oracle dataset, which is a subset of the original dataset that has been sampled using
    rejection sampling.
    """
 
    # source dataset
    oracle = D.to_numpy()

    # feature to condition on (can also be boolean/int array for multiple)
    X_c = np.array(feat_id).astype(int)

    # whether to do shift using target data of X_c, or with manual shift (define below)
    manual_shift = False

    # get original distribution
    p_ = gaussian_kde(oracle[:, X_c].T)

    # Annoying that kde takes features x samples (instead of other way around). Fix:
    p = lambda x: p_(x.T)

    # Define your shifted distribution here:
    if manual_shift:
        mean_shift = 60 # can also be multivariate for multivariate X_c
        p_shifted = lambda x: p(x-mean_shift)
        # Create target dataset (just for checking it works)
        target = np.random.randn(n,d)
        target[:, X_c] += mean_shift
    else:
        # Or if target features/label are observed, simply train it on CUTRACT variables
        target = np.random.normal(loc=mean, scale=std/2, size=10000) #D_cutract.to_numpy() ## change if manual
        #p_shifted_ = gaussian_kde(target[:,X_c].T)
        p_shifted_ = gaussian_kde(target.T)
        p_shifted = lambda x: p_shifted_(x.T)

    first_run = True
    test_set_size = 1000
    shifted_oracle = []
    eps = 1e-8
    # choose M, see https://en.wikipedia.org/wiki/Rejection_sampling. 50 is probably fine for our purposes; I'm raising an error if it's not
    M = 60 #80
    i=0
    while len(shifted_oracle)<test_set_size:
        ratio = p_shifted(oracle[:,X_c])/(M*p(oracle[:, X_c])+eps)
        
        if np.max(ratio) > 1:
            raise ValueError(f'Increase M to at least {M*np.max(ratio)}')
        
        accept = ratio > np.random.rand(*ratio.shape)
        if first_run:
            shifted_oracle = oracle[accept]
            first_run = False
        else:
            shifted_oracle = np.concatenate([shifted_oracle, oracle[accept]],axis=0)

        i+=1

        if i>=20:
          break

    shifted_oracle = shifted_oracle[:test_set_size]

    return shifted_oracle
