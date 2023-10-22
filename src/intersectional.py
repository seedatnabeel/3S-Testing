def intersection_matrix(data, model, trained_model_dict, cols, discrete_columns, n_min=1):
    """
    Helper to compute the intersectional matrix
    
    Args:
      data: The input data for the intersection matrix. It should be a pandas DataFrame.
      model: The "model" parameter is the name of the machine learning model that you want to use for
    prediction. It should be a string that matches one of the keys in the "trained_model_dict"
    dictionary.
      trained_model_dict: The parameter "trained_model_dict" is a dictionary that contains trained
    machine learning models. The keys of the dictionary are the names of the models, and the values are
    the trained models themselves.
      cols: The `cols` parameter is a list of column names that you want to include in the intersection
    matrix. These columns will be used to calculate the intersection between different categories.
      discrete_columns: The `discrete_columns` parameter is a list of column names that represent
    discrete variables in your dataset. These are variables that have a finite number of distinct
    values, such as categories or labels.
      n_min: The parameter `n_min` is the minimum number of samples required for evaluation. If there
    are not enough samples for a particular combination of categories, an error will be raised. Defaults
    to 1
    
    Returns:
      a matrix and a tuple. The matrix represents the intersection matrix, which contains the accuracy
    scores for different combinations of categories in the input data. The tuple contains three
    elements: df_discretised, cat_per_feat, and all_cats. df_discretised is a copy of the input data
    with discretized values, cat_per_feat is a dictionary that maps each feature to its corresponding
    categories
    """
    from copy import deepcopy
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    import matplotlib.pyplot as plt
    from copy import copy

    gen_data = data

    gen_data_eval = deepcopy(gen_data).reset_index(drop=True)

    model_name=model
    clf = trained_model_dict[model_name]

    y_pred = clf.predict(gen_data.drop('y', axis=1))
    gen_data_eval['yhat'] = y_pred

    data_gen = gen_data_eval[cols]

  

    num_cats_for_cont = 5
    cat_per_feat = {}
    all_cats = []
    df_discretised = copy(data_gen)
    bins = {}
    for col in data_gen.columns:
        if col == 'y' or col=='yhat':
            continue
        elif col in discrete_columns:
            cat_names = np.unique(data_gen[col]).astype(int)
            cat_per_feat[col] = cat_names
            all_cats += list(cat_names)
        else:
            thresholds = np.linspace(data_gen[col].min(), data_gen[col].max(), num_cats_for_cont, endpoint=False)
            midpoints = list(thresholds + (thresholds[1]-thresholds[0])/2)
            all_cats += list(range(len(midpoints)))
            cat_per_feat[col] = midpoints
            df_discretised[col], bins[col] = pd.cut(data_gen[col], num_cats_for_cont, labels=midpoints, retbins=True)

    count=0
    raise_error=True
    allow_more_samples = True
    n_evaluation_samples=n_min
    max_samples=10000
    matrix = np.zeros((len(all_cats), len(all_cats)))
    features = cat_per_feat.keys()
    i = 0
    for x1 in features:
        for cat1 in cat_per_feat[x1]:
            j=0
            for x2 in features:
                for cat2 in cat_per_feat[x2]:
                    if j > i:
                        break
                    elif x1 == x2 and cat1 != cat2:
                        matrix[i,j] = None
                        matrix[j,i] = matrix[i,j]

                        
                    else:
                        condition = np.where((df_discretised[x1]==cat1) & (df_discretised[x2] == cat2))[0]
                        df_subset = data_gen.loc[condition]



                        if len(df_subset) < n_evaluation_samples:
                            if raise_error:
                                continue
                                #raise ValueError('Not enough samples')
                            else:
                                print('Not enough samples')
                        else:
                            if not allow_more_samples:
                                df_subset = df_subset.sample(n=max_samples, replace=True)

                            acc = precision_score(df_subset['y'], df_subset['yhat'])
                            matrix[i,j] = acc
                            matrix[j,i] = matrix[i,j]
                    j+=1
            i+=1

    return deepcopy(matrix), (df_discretised,cat_per_feat,all_cats)