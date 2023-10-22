import numpy as np
from copy import deepcopy
import random
from sklearn.model_selection import train_test_split
import logging

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy

def load_seer_cutract_dataset(name="seer", seed=42):
    import sklearn

    def aggregate_grade(row):
        if row["grade_1.0"] == 1:
            return 1
        if row["grade_2.0"] == 1:
            return 2
        if row["grade_3.0"] == 1:
            return 3
        if row["grade_4.0"] == 1:
            return 4
        if row["grade_5.0"] == 1:
            return 5

    def aggregate_stage(row):
        if row["stage_1"] == 1:
            return 1
        if row["stage_2"] == 1:
            return 2
        if row["stage_3"] == 1:
            return 3
        if row["stage_4"] == 1:
            return 4
        if row["stage_5"] == 1:
            return 5

    def aggregate_treatment(row):
        if row["treatment_CM"] == 1:
            return 1
        if row["treatment_Primary hormone therapy"] == 1:
            return 2
        if row["treatment_Radical Therapy-RDx"] == 1:
            return 3
        if row["treatment_Radical therapy-Sx"] == 1:
            return 4

    features = [
        "age",
        "psa",
        "comorbidities",
        "treatment_CM",
        "treatment_Primary hormone therapy",
        "treatment_Radical Therapy-RDx",
        "treatment_Radical therapy-Sx",
        "grade",
        "stage",
    ]

    features = [
        "age",
        "psa",
        "comorbidities",
        "treatment",
        "grade",
        "stage",
    ]

    # features = ['age', 'psa', 'comorbidities', 'treatment_CM', 'treatment_Primary hormone therapy',
    #         'treatment_Radical Therapy-RDx', 'treatment_Radical therapy-Sx', 'grade', 'stage']
    label = "mortCancer"
    try:
        df = pd.read_csv(f"../data/{name}.csv")
    except BaseException:
        df = pd.read_csv(f"../data/{name}.csv")

    df["grade"] = df.apply(aggregate_grade, axis=1)
    df["stage"] = df.apply(aggregate_stage, axis=1)
    df["treatment"] = df.apply(aggregate_treatment, axis=1)
    df["mortCancer"] = df["mortCancer"].astype(int)
    df["mort"] = df["mort"].astype(int)

    mask = df[label] == True
    df_dead = df[mask]
    df_survive = df[~mask]

    if name == "seer":
        n_samples = 10000
    else:
        n_samples = 1000
        
    df = pd.concat(
        [
            df_dead.sample(n_samples, random_state=seed),
            df_survive.sample(n_samples, random_state=seed),
        ]
    )
    df = sklearn.utils.shuffle(df, random_state=seed)
    df = df.reset_index(drop=True)
    return df[features], df[label]

def load_adult_data(split_size=0.3):
    """
    > This function loads the adult dataset, removes all the rows with missing values, and then splits the data into
    a training and test set
    Args:
      split_size: The proportion of the dataset to include in the test split.
    Returns:
      X_train, X_test, y_train, y_test, X, y
    """

    def process_dataset(df, random_state=42):
        """
        > This function takes a dataframe, maps the categorical variables to numerical values, and returns a
        dataframe with the numerical values
        Args:
          df: The dataframe to be processed
        Returns:
          a dataframe after the mapping
        """

        data = [df]

        salary_map = {" <=50K": 1, " >50K": 0}
        df["salary"] = df["salary"].map(salary_map).astype(int)

        df["sex"] = df["sex"].map({" Male": 1, " Female": 0}).astype(int)

        df["country"] = df["country"].replace(" ?", np.nan)
        df["workclass"] = df["workclass"].replace(" ?", np.nan)
        df["occupation"] = df["occupation"].replace(" ?", np.nan)

        df.dropna(how="any", inplace=True)

        for dataset in data:
            dataset.loc[
                dataset["country"] != " United-States",
                "country",
            ] = "Non-US"
            dataset.loc[
                dataset["country"] == " United-States",
                "country",
            ] = "US"

        df["country"] = df["country"].map({"US": 1, "Non-US": 0}).astype(int)

        df["marital-status"] = df["marital-status"].replace(
            [
                " Divorced",
                " Married-spouse-absent",
                " Never-married",
                " Separated",
                " Widowed",
            ],
            "Single",
        )
        df["marital-status"] = df["marital-status"].replace(
            [" Married-AF-spouse", " Married-civ-spouse"],
            "Couple",
        )

        df["marital-status"] = df["marital-status"].map(
            {"Couple": 0, "Single": 1},
        )

        rel_map = {
            " Unmarried": 0,
            " Wife": 1,
            " Husband": 2,
            " Not-in-family": 3,
            " Own-child": 4,
            " Other-relative": 5,
        }

        df["relationship"] = df["relationship"].map(rel_map)

        race_map = {
            " White": 0,
            " Amer-Indian-Eskimo": 1,
            " Asian-Pac-Islander": 2,
            " Black": 3,
            " Other": 4,
        }

        df["race"] = df["race"].map(race_map)

        def f(x):
            if (
                x["workclass"] == " Federal-gov"
                or x["workclass"] == " Local-gov"
                or x["workclass"] == " State-gov"
            ):
                return "govt"
            elif x["workclass"] == " Private":
                return "private"
            elif (
                x["workclass"] == " Self-emp-inc"
                or x["workclass"] == " Self-emp-not-inc"
            ):
                return "self_employed"
            else:
                return "without_pay"

        df["employment_type"] = df.apply(f, axis=1)

        employment_map = {
            "govt": 0,
            "private": 1,
            "self_employed": 2,
            "without_pay": 3,
        }

        df["employment_type"] = df["employment_type"].map(employment_map)
        df.drop(
            labels=[
                "workclass",
                "education",
                "occupation",
            ],
            axis=1,
            inplace=True,
        )
        df.loc[(df["capital-gain"] > 0), "capital-gain"] = 1
        df.loc[(df["capital-gain"] == 0, "capital-gain")] = 0

        df.loc[(df["capital-loss"] > 0), "capital-loss"] = 1
        df.loc[(df["capital-loss"] == 0, "capital-loss")] = 0

        return df

    try:
        df = pd.read_csv("data/adult.csv", delimiter=",")
    except BaseException:
        df = pd.read_csv("../data/adult.csv", delimiter=",")

    df = process_dataset(df)

    df_sex_1 = df.query("sex ==1")

    salary_1_idx = df.query("sex == 0 & salary == 1")
    salary_0_idx = df.query("sex == 0 & salary == 0")

    X = df.drop(["salary"], axis=1)
    X = X.drop('fnlwgt', axis=1)
    y = df["salary"]

    # Creation of Train and Test dataset
    random.seed(a=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_size,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test, X, y



def load_covid_dataset(seed=42, drop_SG_UF_NOT=True):

    df_ALL = pd.read_csv("../data/covid.csv")
    x_ids = [2,3,5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
    df = df_ALL.iloc[:,x_ids]
    y = df_ALL.iloc[:,0]

    if drop_SG_UF_NOT:
        df= df.drop(columns=['Race', 'SG_UF_NOT'])
    else:
        df= df.drop(columns=['Race'])

    df['y'] = y 
    df.columns

    races = ['Branca', 'Preta', 'Amarela', 'Parda', 'Indigena']
    regions = []
    for i in range(df.shape[0]):
        if df.iloc[i][races[0]]==1:
            regions.append(1)
        if df.iloc[i][races[1]]==1:
            regions.append(2)
        if df.iloc[i][races[2]]==1:
            regions.append(3)
        if df.iloc[i][races[3]]==1:
            regions.append(4)
        if df.iloc[i][races[4]]==1:
            regions.append(5)

    df['region']=regions


    ages = ['Age_40', 'Age_40_50', 'Age_50_60', 'Age_60_70','Age_70']
    regions = []
    for i in range(df.shape[0]):
        if df.iloc[i][ages[0]]==1:
            regions.append(1)
        if df.iloc[i][ages[1]]==1:
            regions.append(2)
        if df.iloc[i][ages[2]]==1:
            regions.append(3)
        if df.iloc[i][ages[3]]==1:
            regions.append(4)
        if df.iloc[i][ages[4]]==1:
            regions.append(5)

    df['age_group']=regions

    df= df.drop(columns=races)
    df= df.drop(columns=ages)
    
    return df.drop(columns=['y']), df['y'], df



def load_drug_dataset():
    from sklearn.impute import SimpleImputer
    data = pd.read_csv('../data/Drug_Consumption.csv')

    #Drop overclaimers, Semer, and other nondrug columns
    data = data.drop(data[data['Semer'] != 'CL0'].index)
    data = data.drop(['Semer', 'Caff', 'Choc'], axis=1)
    data.reset_index()

    # Binary encode gender
    data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'M' else 0)

    # Encode ordinal features
    ordinal_features = ['Age', 
                        'Education',
                        'Alcohol',
                        'Amyl',
                        'Amphet',
                        'Benzos',
                        'Cannabis',
                        'Coke',
                        'Crack',
                        'Ecstasy',
                        'Heroin',
                        'Ketamine',
                        'Legalh',
                        'LSD',
                        'Meth',
                        'Mushrooms',
                        'Nicotine',
                        'VSA'    ]

    # Define ordinal orderings
    ordinal_orderings = [
        ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
        ['Left school before 16 years', 
        'Left school at 16 years', 
        'Left school at 17 years', 
        'Left school at 18 years',
        'Some college or university, no certificate or degree',
        'Professional certificate/ diploma',
        'University degree',
        'Masters degree',
        'Doctorate degree'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6']
    ]

    # Nominal features
    nominal_features = ['Country',
                        'Ethnicity']

    #Create function for ordinal encoding
    def ordinal_encoder(df, columns, ordering):
        df = df.copy()
        for column, ordering in zip(ordinal_features, ordinal_orderings):
            df[column] = df[column].apply(lambda x: ordering.index(x))
        return df

    def cat_converter(df, columns):
        df = df.copy()
        for column in columns:
            df[column] = df[column].astype('category').cat.codes
        return df

    data = ordinal_encoder(data, ordinal_features, ordinal_orderings)
    data = cat_converter(data, nominal_features)

    nic_df = data.copy()
    nic_df['y'] = nic_df['Nicotine'].apply(lambda x: 1 if x not in [0,1] else 0)
    nic_df = nic_df.drop(['ID','Nicotine'], axis=1)

    return nic_df.drop(columns=['y']), nic_df['y'], nic_df


def load_bank_dataset(seed=0):
    import pandas as pd

    df = pd.read_csv('../data/Base.csv')
    for col in ["payment_type", "employment_status", "housing_status", "source", "device_os"]:
        df[col] = df[col].astype("category").cat.codes

    df.rename(columns={'fraud_bool': 'y'}, inplace=True)

    mask = df['y'] == True
    df_fraud = df[mask]
    df_no = df[~mask]

    n_samples = 5000
    df = pd.concat(
        [
            df_fraud.sample(n_samples, random_state=seed),
            df_no.sample(n_samples, random_state=seed),
        ]
    )
    from sklearn.utils import shuffle
    df = shuffle(df, random_state=seed)

    return df.drop(columns=['y']), df['y'], df


def load_support_dataset():

    df = pd.read_csv('../data/support_data.csv')

    df['salary'] = df[['under $11k', '$11-$25k', '$25-$50k', '>$50k']].values.argmax(1)+1

    df['race'] =df[[ 'white','black', 'asian', 'hispanic']].values.argmax(1)+1

    df.drop(['under $11k', '$11-$25k', '$25-$50k', '>$50k', 'white','black', 'asian', 'd.time', 'Unnamed: 0', 'hispanic'], axis=1, inplace=True)

    # rename dataframe column
    df.rename(columns={'death': 'y'}, inplace=True)

    return df.drop(columns=['y']), df['y'], df