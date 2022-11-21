######################### IMPORTS #########################

import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# import preprocessing
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

from env import user, password, host



######################### TABLE OF CONTENTS #########################
def TOC():
    print('ACQUIRE DATA')
    print('* get_db_url')
    print('* new_wrangle_zillow_2017')
    print('* get_wrangle_zillow_2017')
    print('* wrangle_zillow')
    print()
    
    print('PREPARE DATA')
    print('* clean_zillow_2017')
    print('* train_val_test_split')
    print('* split')
    print('* Xy_split')
    print('* scale_data')
    print('* visualize_scaler')

    

######################### ACQUIRE DATA #########################

def get_db_url(db):

    '''
    This function calls the username, password, and host from env file and provides database argument for SQL
    '''

    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
#------------------------- ZILLOW DATA FROM SQL -------------------------

def new_wrangle_zillow_2017():

    '''
    This function reads the zillow (2017) data from the Codeup database based on defined query argument and returns a DataFrame.
    '''

    # Create SQL query.
    query = """SELECT * FROM predictions_2017 LEFT JOIN unique_properties USING (parcelid) LEFT JOIN properties_2017 USING (parcelid) LEFT JOIN airconditioningtype USING (airconditioningtypeid) LEFT JOIN architecturalstyletype USING (architecturalstyletypeid) LEFT JOIN buildingclasstype USING (buildingclasstypeid) LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid) LEFT JOIN propertylandusetype USING (propertylandusetypeid) LEFT JOIN storytype USING (storytypeid) LEFT JOIN typeconstructiontype USING (typeconstructiontypeid) WHERE propertylandusetypeid = 261 and transactiondate LIKE '2017%%'"""
    
    # Read in DataFrame from Codeup db using defined arguments.
    df = pd.read_sql(query, get_db_url('zillow'))

    return df

def get_wrangle_zillow_2017():

    '''
    This function checks for a local file and reads it in as a Datafile.  If the csv file does not exist, it calls the new_wrangle function then writes the data to a csv file.
    '''

    # Checks for csv file existence
    if os.path.isfile('zillow_2017.csv'):
        
        # If csv file exists, reads in data from the csv file.
        df = pd.read_csv('zillow_2017.csv', index_col=0)
        
    else:
        
        # If csv file does not exist, uses new_wrangle_zillow_2017 function to read fresh data from telco db into a DataFrame
        df = new_wrangle_zillow_2017()
        
        # Cache data into a new csv file
        df.to_csv('zillow_2017.csv')
        
    return pd.read_csv('zillow_2017.csv', index_col=0)

#------------------------ ONE WRANGLE FILE TO RUN THEM ALL ------------------------

def wrangle_zillow():
    """
    This function is used to run all Acquire and Prepare functions.
    """
    df = get_wrangle_zillow_2017()
    df = clean_zillow_2017(df)
    return df



######################### PREPARE DATA #########################

def clean_zillow_2017(df):

    """
    This function is used to clean the zillow_2017 data as needed 
    ensuring not to introduce any new data but only remove irrelevant data 
    or reshape existing data to useable formats.
    """

    # Clean all Whitespace by converting to NaN using R
    df = df.replace(r'^\s*$', np.NaN, regex=True)

    # Replace on multiple columns
    convert_columns_df = ['basementsqft', 
                          'decktypeid', 
                          'pooltypeid10', 
                          'poolsizesum', 
                          'pooltypeid2', 
                          'pooltypeid7', 
                          'poolcnt', 
                          'hashottuborspa', 
                          'taxdelinquencyyear', 
                          'fireplacecnt', 
                          'numberofstories', 
                          'garagecarcnt', 
                          'garagetotalsqft']

    df[convert_columns_df] = df[convert_columns_df].fillna(0)

    # Drop all columns with more than 19,000 NULL/NaN
    df = df.dropna(axis='columns', thresh=19_000)
    
    # Drop Column regionidneighborhood with 33,408 Null/NaN 
    df = df.drop(columns=['regionidneighborhood'])
    
    # Drop remaining Columns with more than 18,000 Null/NaN 
    df = df.drop(columns=['buildingqualitytypeid',
                          'unitcnt',
                          'propertyzoningdesc',
                          'heatingorsystemdesc',
                          'heatingorsystemtypeid'])
    
    # Drop rows with NULL/NaN since it is only 3% of DataFrame 
    df = df.dropna()
    
    # Convert dtypes
    df = df.convert_dtypes(infer_objects=False)
    
    # filter down outliers to more accurately align with realistic expectations of a Single Family Residence
    # Set no_outliers equal to df
    no_outliers = df

    # Keep all homes that have > 0 and <= 8 Beds and Baths
    no_outliers = no_outliers[no_outliers.bedroomcnt > 0]
    no_outliers = no_outliers[no_outliers.bathroomcnt > 0]
    no_outliers = no_outliers[no_outliers.bedroomcnt <= 8]
    no_outliers = no_outliers[no_outliers.bathroomcnt <= 8]
    
    # Keep all homes that have tax value > 30 thousand and <= 2 million
    no_outliers = no_outliers[no_outliers.taxvaluedollarcnt >= 40_000]
    no_outliers = no_outliers[no_outliers.taxvaluedollarcnt <= 2_000_000]
    
    # Keep all homes that have sqft > 4 hundred and < 10 thousand
    no_outliers = no_outliers[no_outliers.calculatedfinishedsquarefeet > 800]
    no_outliers = no_outliers[no_outliers.calculatedfinishedsquarefeet < 10_000]

    # Assign no_outliers back to the DataFrame
    df = no_outliers
    
    # Create a feature to replace yearbuilt that shows the age of the home in 2017 when data was collected
    df['age'] = 2017 - df.yearbuilt
    
    # Create a feature to show ration of Bathrooms to Bedrooms
    df['bed_bath_ratio'] = round((df.bedroomcnt / df.bathroomcnt), 2)
    
    # fips Conversion
    # This is technically a backwards engineered feature
    # fips is already an engineered feature of combining county and state into one code
    # This feature was just a rabit hole for exercise and experience it also provides Human Readable reference

    # Found a csv fips master list on github
    # Read it in as a DataFrame using raw url
    url = 'https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv'
    fips_df = pd.read_csv(url)
    
    # Cache data into a new csv file
    fips_df.to_csv('state_and_county_fips_master.csv')
    
    # left merge to join the name and state to the original df
    left_merged_fips_df = pd.merge(df, fips_df, how="left", on=["fips"])
    
    # Rewrite the df
    df = left_merged_fips_df

    # Assign unecessary Columns to Drop
    drop_columns = ['propertylandusetypeid',
                    'parcelid',
                    'id',
                    'id.1',
                    'decktypeid',
                    'finishedsquarefeet12',
                    'pooltypeid10',
                    'pooltypeid2',
                    'pooltypeid7',
                    'propertycountylandusecode',
                    'roomcnt',
                    'numberofstories',
                    'propertylandusedesc',
                    'yearbuilt',
                    'rawcensustractandblock',
                    'transactiondate', 
                    'assessmentyear',
                    'garagetotalsqft', 
                    'garagecarcnt',
                    'calculatedbathnbr', 
                    'poolsizesum',
                    'state',
                    'regionidcounty',
                    'taxamount', 
                    'structuretaxvaluedollarcnt', 
                    'landtaxvaluedollarcnt']

    # Drop unecessary Columns 
    df = df.drop(columns=drop_columns)

    # Replace Conditional values
    df["taxdelinquencyyear"] = np.where(df["taxdelinquencyyear"] > 0, 1, 0)
    df["basementsqft"] = np.where(df["basementsqft"] > 0, 1, 0)
    
    # Rename categorical columns
    df.rename(columns = {'hashottuborspa': 'has_hottuborspa',
                         'taxdelinquencyyear': 'has_taxdelinquency', 
                         'basementsqft': 'has_basement', 
                         'poolcnt': 'has_pool', 
                         'name': 'county'}
              , inplace = True)
    
    # Use pandas dummies to pivot features with more than two string values
    # into multiple columns with binary int values that can be read as boolean
    # drop_first = False in draft for human readability; Final will have it set to True.
    dummy_df = pd.get_dummies(data=df[['county']], drop_first=False)
    
    # Assign dummies to DataFrame
    df = pd.concat([df, dummy_df], axis=1)
    
    # Drop dummy Columns 
    df = df.drop(columns='county')

    # Rearange Columns for Human Readability
    df = df[['bedroomcnt', 
             'bathroomcnt',
             'fullbathcnt', 
             'bed_bath_ratio',
             'fireplacecnt', 
             'age', 
             'calculatedfinishedsquarefeet',
             'lotsizesquarefeet', 
             'has_taxdelinquency',
             'has_hottuborspa', 
             'has_basement', 
             'has_pool', 
             'fips',
             'longitude',
             'latitude', 
             'regionidcity', 
             'regionidzip', 
             'censustractandblock', 
             'logerror', 
             'taxvaluedollarcnt',
             'county_Los Angeles County',
             'county_Orange County',
             'county_Ventura County']]
    
    # Drop Location Reference Columns unsuitable for use with ML without categorical translation
    df = df.drop(columns=['longitude',
                 'latitude', 
                 'regionidcity', 
                 'regionidzip', 
                 'censustractandblock'])
    
    # Remove and Archive the logerror results for future comparison 
    logerror = df.logerror

    # Cache DataFrame into a new csv file
    df.to_csv('zillow_2017_cleaned.csv')
    logerror.to_csv('zillow_2017_logerror.csv')
    
    return df

        

######################### SPLIT DATA #########################

def split(df, stratify=False, target=None):
    """
    This Function splits the DataFrame into train, validate, and test
    then prints a graphic representation and a mini report showing the shape of the original DataFrame
    compared to the shape of the train, validate, and test DataFrames.
    """
    
    # Do NOT stratify on continuous data
    if stratify:
        # Split df into train and test using sklearn
        train, test = train_test_split(df, test_size=.2, random_state=1992, stratify=df[target])
        # Split train_df into train and validate using sklearn
        train, validate = train_test_split(train, test_size=.25, random_state=1992, stratify=df[target])
        
    else:
        train, test = train_test_split(df, test_size=.2, random_state=1992)
        train, validate = train_test_split(train, test_size=.25, random_state=1992)
    
    # reset index for train validate and test
    train.reset_index(drop=True, inplace=True)
    validate.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    train_prcnt = round((train.shape[0] / df.shape[0]), 2)*100
    validate_prcnt = round((validate.shape[0] / df.shape[0]), 2)*100
    test_prcnt = round((test.shape[0] / df.shape[0]), 2)*100
    
    print('________________________________________________________________')
    print('|                              DF                              |')
    print('|--------------------:--------------------:--------------------|')
    print('|        Train       |      Validate      |        Test        |')
    print(':--------------------------------------------------------------:')
    print()
    print()
    print(f'Prepared df: {df.shape}')
    print()
    print(f'      Train: {train.shape} - {train_prcnt}%')
    print(f'   Validate: {validate.shape} - {validate_prcnt}%')
    print(f'       Test: {test.shape} - {test_prcnt}%')
 
    
    return train, validate, test


def Xy_split(feature_cols, target, train, validate, test):
    """
    This function will split the train, validate, and test data by the Feature Columns selected and the Target.
    
    Imports Needed:
    from sklearn.model_selection import train_test_split
    
    Arguments Taken:
       feature_cols: list['1','2','3'] the feature columns you want to run your model against.
             target: list the target feature that you will try to predict
              train: Assign the name of your train DataFrame
           validate: Assign the name of your validate DataFrame
               test: Assign the name of your test DataFrame
    """
    
    print('_______________________________________________________________')
    print('|                              DF                             |')
    print('|-------------------:-------------------:---------------------|')
    print('|       Train       |       Validate    |          Test       |')
    print('|-------------------:-------------------:---------------------|')
    print('| x_train | y_train |   x_val  |  y_val |   x_test  |  y_test |')
    print(':-------------------------------------------------------------:')
    
    X_train, y_train = train[feature_cols], train[target]
    X_validate, y_validate = validate[feature_cols], validate[target]
    X_test, y_test = test[feature_cols], test[target]

    print()
    print()
    print(f'   X_train: {X_train.shape}   {X_train.columns}')
    print(f'   y_train: {y_train.shape}     Index({target})')
    print()
    print(f'X_validate: {X_validate.shape}   {X_validate.columns}')
    print(f'y_validate: {y_validate.shape}     Index({target})')
    print()
    print(f'    X_test: {X_test.shape}   {X_test.columns}')
    print(f'    y_test: {y_test.shape}     Index({target})')
    
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test
    # When I run this it returns the OUT but  gives me a name not defined error when I try to call Variables.
    # NameError: name 'X_train' is not defined
    # NameError: name 'y_train' is not defined



######################### SCALE SPLIT #########################

def scale_data(train, 
               validate, 
               test, 
               columns_to_scale,
               scaler,
               return_scaler = False):
    
    """
    Scales the 3 data splits. 
    Takes in train, validate, and test data 
    splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    
    Imports Needed:
    from sklearn.preprocessing import MinMaxScaler 
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import QuantileTransformer
    
    Arguments Taken:
               train = Assign the train DataFrame
            validate = Assign the validate DataFrame 
                test = Assign the test DataFrame
    columns_to_scale = Assign the Columns that you want to scale
              scaler = Assign the scaler to use MinMaxScaler(),
                                                StandardScaler(), 
                                                RobustScaler(), or 
                                                QuantileTransformer()
       return_scaler = False by default and will not return scaler data
                       True will return the scaler data before displaying the _scaled data
    """
    
    # make copies of our original data so we dont corrupt original split
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    # fit the scaled data
    scaler.fit(train[columns_to_scale])
    
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    
    else:
        return train_scaled, validate_scaled, test_scaled


    
######################### DATA SCALE VISUALIZATION #########################

# Function Stolen from Codeup Instructor Andrew King
def visualize_scaler(scaler, df, columns_to_scale, bins=10):
    """
    This Function takes input arguments, 
    creates a copy of the df argument, 
    scales it according to the scaler argument, 
    then displays subplots of the columns_to_scale argument 
    before and after scaling.
    """    

    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(16,9))
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()
    #return df_scaled.head().T
    #return fig, axs
    
    
    
############### Display Visualization figures and run Hypothesis Testing for Final Project Presentation ###############
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



def fig_00(df):
    """
    Display Figure with minimal code in the notebook.
    
    Imports Needed:
    import matplotlib.pyplot as plt
    import seaborn as sns
    """
    
    plt.figure(figsize=(8, 12))
    heatmap = sns.heatmap(df.corr(method='spearman')[['taxvaluedollarcnt']].sort_values(by='taxvaluedollarcnt', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Features Correlating with Tax Assessment Value', fontdict={'fontsize':18}, pad=16)
    
def fig_Q1(df):
    """
    Display Figure with minimal code in the notebook.
    
    Imports Needed:
    import matplotlib.pyplot as plt
    import seaborn as sns
    """
    sns.lmplot(x='calculatedfinishedsquarefeet', y='taxvaluedollarcnt', data=df.astype('float64'), line_kws={'color': 'red'})
    plt.show()
    
def test_Q1(df):
    """
    Display Hypothesis Statistical Testing with minimal code in the notebook.
    
    Imports Needed:
    from scipy import stats
    """
    α = 0.05   
    r, p_val = stats.pearsonr(df.calculatedfinishedsquarefeet,
                              df.taxvaluedollarcnt)
    if p_val < α:
        print('Decision: Reject the null hypothesis')
    else:
        print('Decision: Fail to reject the null hypothesis')
    print()    
    print(f'       r: {r}')
    print(f'   p_val: {p_val}')

def fig_Q2(df):
    """
    Display Figure with minimal code in the notebook.
    
    Imports Needed:
    import matplotlib.pyplot as plt
    import seaborn as sns
    """
    sns.catplot(data=df, y='taxvaluedollarcnt', x='bedroomcnt')
    plt.show()
    
def test_Q2(df):
    """
    Display Hypothesis Statistical Testing with minimal code in the notebook.
    
    Imports Needed:
    from scipy import stats
    """
    α = 0.05   
    r, p_val = stats.pearsonr(df.bedroomcnt,
                              df.taxvaluedollarcnt)
    if p_val < α:
        print('Decision: Reject the null hypothesis')
    else:
        print('Decision: Fail to reject the null hypothesis')
    print()    
    print(f'       r: {r}')
    print(f'   p_val: {p_val}')
    
def fig_Q3(df):
    """
    Display Figure with minimal code in the notebook.
    
    Imports Needed:
    import matplotlib.pyplot as plt
    import seaborn as sns
    """
    sns.catplot(data=df.astype('float64'), x="bathroomcnt", y="taxvaluedollarcnt", kind="box")
    plt.show()
    
def test_Q3(df):
    """
    Display Hypothesis Statistical Testing with minimal code in the notebook.
    
    Imports Needed:
    from scipy import stats
    """
    α = 0.05   
    r, p_val = stats.pearsonr(df.bathroomcnt,
                              df.taxvaluedollarcnt)
    if p_val < α:
        print('Decision: Reject the null hypothesis')
    else:
        print('Decision: Fail to reject the null hypothesis')
    print()    
    print(f'       r: {r}')
    print(f'   p_val: {p_val}')
    
def fig_Q4(df):
    """
    Display Figure with minimal code in the notebook.
    
    Imports Needed:
    import matplotlib.pyplot as plt
    import seaborn as sns
    """
    sns.lmplot(x='bedroomcnt', y='bathroomcnt', data=df.astype('float64'), line_kws={'color': 'red'})
    plt.show()
    
def test_Q4(df):
    """
    Display Hypothesis Statistical Testing with minimal code in the notebook.
    
    Imports Needed:
    from scipy import stats
    """
    α = 0.05   
    r, p_val = stats.pearsonr(df.bedroomcnt,
                              df.bathroomcnt)
    if p_val < α:
        print('Decision: Reject the null hypothesis')
    else:
        print('Decision: Fail to reject the null hypothesis')
    print()    
    print(f'       r: {r}')
    print(f'   p_val: {p_val}')    
    

    
######################### MODELING #########################

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoLars
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error








    
    
def create_predictions_df(validate):
    """
    Create pedictions DataFrame to hold predictions from all models in order to evaluate and compare.
    """
    
    predictions = pd.DataFrame({'actual': validate.taxvaluedollarcnt,
                                'logerror': validate.logerror,
                                'baseline': validate.taxvaluedollarcnt.mean()})
        
    return predictions
    
def simple_model(train, validate, predictions):
    """
    Run Regression Model with minimal code in the notebook
    
    Imports Needed:
    from sklearn.linear_model import LinearRegression
    """
       
    # X must be 2-d
    X_train = train[['calculatedfinishedsquarefeet']]
    # y can be 1-d
    y_train = train.taxvaluedollarcnt

    # 1. create the model
    lm = LinearRegression()
    # 2. fit the model
    lm.fit(X_train, y_train)
    # 3. use the model (make predictions)
    X_validate = validate[['calculatedfinishedsquarefeet']]
    
    predictions['simple_lm'] = lm.predict(X_validate)

def mr_rfe(train, validate):
    """
    Run Regression Model with minimal code in the notebook
    
    Imports Needed:
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import RFE
    """
    
    X_train = train.drop(columns='taxvaluedollarcnt')
    y_train = train.taxvaluedollarcnt
    X_validate = validate.drop(columns='taxvaluedollarcnt')

    lm = LinearRegression()
    k = 2

    # 1. Transform our X
    rfe = RFE(lm, n_features_to_select=2)
    rfe.fit(X_train, y_train)
    print('selected top 2 features:', X_train.columns[rfe.support_])
    X_train_rfe = rfe.transform(X_train)
    
def polynomial_degree(train, validate, predictions):
    """
    Run Regression Model with minimal code in the notebook
    
    Imports Needed:
    from sklearn.preprocessing import PolynomialFeatures
    """
    
    X_train = train[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet']]
    y_train = train.taxvaluedollarcnt
    X_validate = validate[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet']]
    y_validate = validate.taxvaluedollarcnt
    
    # 1. Generate Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly.fit(X_train)
    X_train_poly = pd.DataFrame(poly.transform(X_train),
                                columns=poly.get_feature_names(X_train.columns),
                                index=train.index)
    
    # 2. Use the features
    lm = LinearRegression()
    lm.fit(X_train_poly, y_train)

    X_validate_poly = poly.transform(X_validate)
    predictions['2nd degree polynomial'] = lm.predict(X_validate_poly)
        
    # 3. Examine the coefficients of the resulting model 
    feature_names = poly.get_feature_names(X_train.columns)
    
    return pd.Series(lm.coef_, index=feature_names).sort_values()

def polynomial_interaction(train, validate, predictions):
    """
    Run Regression Model with minimal code in the notebook
    
    Imports Needed:
    from sklearn.preprocessing import PolynomialFeatures    
    """
    
    X_train = train[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet']]
    y_train = train.taxvaluedollarcnt
    X_validate = validate[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet']]
    y_validate = validate.taxvaluedollarcnt
    
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly.fit(X_train)
    
    # 1. Generate Polynomial Features
    X_train_poly = pd.DataFrame(poly.transform(X_train), 
                                columns=poly.get_feature_names(X_train.columns), 
                                index=train.index)
    
    # 2. Use the features
    lm = LinearRegression()
    lm.fit(X_train_poly, y_train)

    # 3. Examine the coefficients of the resulting model 
    X_validate_poly = poly.transform(X_validate)
    predictions['polynomial only interaction'] = lm.predict(X_validate_poly)

    return pd.Series(lm.coef_, index=poly.get_feature_names(X_train.columns)).sort_values()

def lasso_lars(train, validate, predictions):
    """
    Run Regression Model with minimal code in the notebook
    
    Imports Needed:
    from sklearn.linear_model import LassoLars
    """
    
    X_train = train[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet']]
    y_train = train.taxvaluedollarcnt
    X_validate = validate[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet']]
    y_validate = validate.taxvaluedollarcnt
    
    # create the model object
    lars = LassoLars(alpha=1)

    # fit the model to our training data
    lars.fit(X_train, y_train)

    # predict validate
    X_validate_pred_lars = lars.predict(X_validate)

    # Add lassolars predictions to our predictions DataFrame
    predictions['lasso_lars'] = X_validate_pred_lars
    
    return pd.Series(lars.coef_, index=X_train.columns).sort_values()

def gen_lin_mdl(train, validate, predictions):
    """
    Run Regression Model with minimal code in the notebook
    
    Imports Needed:
    from sklearn.linear_model import TweedieRegressor
    """
    
    X_train = train[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet']]
    y_train = train.taxvaluedollarcnt
    X_validate = validate[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet']]
    y_validate = validate.taxvaluedollarcnt
    
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)

    # fit the model to our training data
    glm.fit(X_train, y_train)

    # predict validate
    X_validate_predict_glm = glm.predict(X_validate)

    # Add general linear model predictions to our predictions DataFrame
    predictions['glm'] = X_validate_predict_glm
    
    return pd.Series(glm.coef_, index=X_train.columns).sort_values()

def calculate_mse(y_predicted, predictions):
    """
    Run Regression Model with minimal code in the notebook
    
    Imports Needed:
    from sklearn.metrics import mean_squared_error
    """
    
    return mean_squared_error(predictions.actual, y_predicted)
    predictions.apply(calculate_mse).sort_values()
    
def test_polynomial_degree(test):
    """
    Run Regression Model with minimal code in the notebook
    
    Imports Needed:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error
    """
    
    # any transformations applied to your training data must be applied to the test as well
    X_test = test[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet']]
    y_test = test.taxvaluedollarcnt
    
    # 1. Generate Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly.fit(X_test)
    X_test_poly = pd.DataFrame(poly.transform(X_test),
                                columns = poly.get_feature_names(X_test.columns),
                                index = test.index)

    # 2. Use the features
    lm = LinearRegression()
    lm.fit(X_test_poly, y_test)

    test_predictions = lm.predict(X_test_poly)
    test_actual = test.taxvaluedollarcnt
    
    return mean_squared_error(test_actual, test_predictions)
