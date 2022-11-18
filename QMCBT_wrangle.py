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
    print('* null_stats')
    print('* clean_zillow_2017')
    print('* train_val_test_split')
    print('* split')
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
    query = """SELECT *
    FROM predictions_2017
    LEFT JOIN unique_properties USING (parcelid)
    LEFT JOIN properties_2017 USING (parcelid)
    LEFT JOIN airconditioningtype USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
    LEFT JOIN propertylandusetype USING (propertylandusetypeid)
    LEFT JOIN storytype USING (storytypeid)
    LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
    WHERE propertylandusetypeid = 261 and transactiondate LIKE '2017%'"""
    
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
                    'propertylandusedesc']

    # Drop unecessary Columns 
    df = df.drop(columns=drop_columns)

    
    
    
    
    
    
    
    
    
    
    
    
    
    # MAINTAIN COLUMNS
    
    # Rearange Columns
    df = df[['propertylandusetypeid',  
    'propertylandusedesc',  
    'bedroomcnt',  
    'bathroomcnt',  
    'bed_bath_ratio',  
    'calculatedfinishedsquarefeet',  
    'yearbuilt',  
    'age',  
    'taxvaluedollarcnt',  
    'taxamount',  
    'taxpercent',  
    'fips',  
    'name',  
    'state']]
    
    # Drop index and description columns used only for initial filter and verification of data pulled in from SQL.
    df = df.drop(columns=['propertylandusetypeid', 'propertylandusedesc']) 
    
    # Rename Columns
    df = df.rename(columns={'bedroomcnt': 'bedrooms',  
                        'bathroomcnt': 'bathrooms',  
                        'bed_bath_ratio': 'bath_to_bed_ratio',  
                        'calculatedfinishedsquarefeet': 'sqft',  
                        'taxvaluedollarcnt': 'tax_appraisal',  
                        'taxamount': 'tax_bill',  
                        'taxpercent': 'tax_percentage',  
                        'name': 'county'})
    
    return df



def null_stats(df):
    """
    This Function will display the DataFrame row count, 
    the NULL/NaN row count, and the 
    percent of rows that would be dropped.
    """

    print('COUNT OF NULL/NaN PER COLUMN:')
    # set temporary conditions for this instance of code
    with pd.option_context('display.max_rows', None):
        # print count of nulls by column
        print (df.isnull().sum().sort_values(ascending=False))
    print('')
    print(f'     DataFrame Row Count: {df.shape[0]}')
    print(f'      NULL/NaN Row Count: {df.dropna().shape[0]}')
    
    if df.shape[0] == df.dropna().shape[0]:
        print()
        print('Row Counts are the same')
        print('Drop NULL/NaN cannot be run')
    
    elif df.dropna().shape[0] == 0:
        print()
        print('This will remove all records from your DataFrame')
        print('Drop NULL/NaN cannot be run')
    
    else:
        print(f'  DataFrame Percent kept: {round((df.dropna().shape[0] / df.shape[0]), 4)}')
        print(f'NULL/NaN Percent dropped: {round(1 - (df.dropna().shape[0] / df.shape[0]), 4)}')

        

######################### SPLIT DATA #########################

def split(df, stratify=False):
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


def Xy_split(feature_cols, target):
    """
    
    """
    
    print('_______________________________________________________________')
    print('|                              DF                             |')
    print('|-------------------:-------------------:---------------------|')
    print('|       Train       |       Validate    |          Test       |')
    print('|-------------------:-------------------:---------------------|')
    print('| x_train | y_train |   x_val  |  y_val |   x_test  |  y_test |')
    print(':-------------------------------------------------------------:')
    print()
    print('* 1. tree_1 = DecisionTreeClassifier(max_depth = 5)')
    print('* 2. tree_1.fit(x_train, y_train)')
    print('* 3. predictions = tree_1.predict(x_train)')
    print('* 4. pd.crosstab(y_train, y_preds)')
    print('* 5. val_predictions = tree_1.predict(x_val)')
    print('* 6. pd.crosstab(y_val, y_preds)')
    print()
    print()
    print(':------------------------------:')
    print('|Copy, Paste, and Run this code|')
    print(':------------------------------:')
    print()
    print('X_train, y_train = train[feature_cols], train[target]')
    print('X_validate, y_validate = validate[feature_cols], validate[target]')
    print('X_test, y_test = test[feature_cols], test[target]')
    print('X_train.head().T')



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
