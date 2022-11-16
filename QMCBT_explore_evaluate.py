
# ######################### EXPLORE #########################

# IMPORTS NEEDED FOR EXPLORATION

import pandas as pd
import numpy as np



def explore_toc ():
    """
    PRINT TABLE OF CONTENTS FOR CUSTOM EXPLORE FUNCTIONS
    """
    print("                            ** CUSTOM EXPLORATION FUNCTIONS **")
    print("              explore_tips: Print a list of useful FUNCTIONS, METHODS, AND ATTRIBUTES used for EXPLORATION")
    print("    nunique_column_all(df): Print NUNIQUE of all Columns")
    print("nunique_column_objects(df): Print NUNIQUE of Columns that are OBJECTS")
    print("    nunique_column_qty(df): Print NUNIQUE of Columns that are *NOT* OBJECTS")
    print("         numeric_range(df): Compute RANGE for all NUMERIC Variables")
    print("            null_stats(df): Checks for NULLS in DataFrame, returns report with Null count and percentage of drop")
    print("      check_whitespace(df): Checks for WHITESPACE in DataFrame, Replaces with NaN, returns report")

    
    
def explore_tips():
    """
    PRINT A LIST OF USEFUL FUNCTIONS, METHODS, AND ATTRIBUTES USED FOR EXPLORATION
    """
    print("** USEFUL EXPLORATORY CODE**")
    print("DFNAME.head()")
    print("DFNAME.shape")
    print("DFNAME.shape[0] #read row count")
    print("DFNAME.describe().T")
    print("DFNAME.columns.to_list()")
    print("DFNAME.COLUMNNAME.value_counts(dropna=False)")
    print("DFNAME.dtypes")
    print("DFNAME.select_dtypes(include='object').columns")
    print("DFNAME.select_dtypes(include='float').columns")
    print("pd.crosstab(DFNAME.COLUMN-1, DFNAME.COLUMN-2)")

    
    
def nunique_column_all(df):
    """
    This Function prints the nunique of all columns
    """
    for col in df.columns:
        print(df[col].value_counts())
        print()

        
        
def nunique_column_objects(df): 
    """
    This Function prints the nunique of all columns with dtype: object
    """
    for col in df.columns:
        if df[col].dtypes == 'object':
            print(f'{col} has {df[col].nunique()} unique values.')

            
            
def nunique_column_qty(df): 
    """
    This Function prints the nunique of all columns that are NOT dtype: object
    """
    for col in df.columns:
        if df[col].dtypes != 'object':
            print(f'{col} has {df[col].nunique()} unique values.')

            
            
def numeric_range(df):
    """
    This Function computes the range for all numeric variables
    """
    numeric_list = df.select_dtypes(include = 'float').columns.tolist()
    numeric_range = df[numeric_list].describe().T
    numeric_range['range'] = numeric_range['max'] - numeric_range['min']
    return numeric_range



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
        print(df.isnull().sum().sort_values(ascending=False))
    print()
    print(f'     DataFrame Row Count: {df.shape[0]}')
    print(f'      NULL/NaN Row Count: {df.shape[0] - df.dropna().shape[0]}')
    
    if df.shape[0] == df.dropna().shape[0]:
        print()
        print('Row Counts are the same')
        print('Drop NULL/NaN cannot be run')
    
    elif df.dropna().shape[0] == 0:
        print()
        print('This will remove all records from your DataFrame')
        print('Drop NULL/NaN cannot be run')
    
    else:
        print()
        print(f'  DataFrame Percent kept: {round((df.dropna().shape[0] / df.shape[0]), 4)}')
        print(f'NULL/NaN Percent dropped: {round(1 - (df.dropna().shape[0] / df.shape[0]), 4)}')



def check_whitespace(df):
    """
    This Function checks the DataFrame argument for whitespace,
    replaces any that exist with NaN, then returns report.
    
    Imports Needed:
    import numpy as np
    """
    
    row_count = df.shape[0]
    column_list = df.columns
    row_value_count = df[column_list].value_counts().sum()
    whitespace_count = row_count - row_value_count
    
    if whitespace_count > 0:
        df = df.replace(r'^\s*$', np.NaN, regex=True)
        print(f'There were {whitespace_count} Whitespace characters found and replaced with NULL/NaN.')
        print()
        print()
        
    else:
        print(f'There were {whitespace_count} Whitespace characters found.')
        print()
        print()

    # set temporary conditions for this instance of code    
    with pd.option_context('display.max_rows', None):
        # print count of nulls by column
        print('COUNT OF NULL/NaN PER COLUMN:')
        print (df.isnull().sum())


########################## EVALUATE #########################

# IMPORTS NEEDED FOR EVALUATION

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector



def eval_toc():
    """
    PRINT TABLE OF CONTENTS FOR CUSTOM EVALUATION FUNCTIONS
    """
    print("                        ** CUSTOM EVALUATION FUNCTIONS **")
    print("             eval_tips: PRINT A LIST OF USEFUL FUNCTIONS, METHODS, AND ATTRIBUTES USED FOR EXPLORATION")
    print("   print_class_metrics: PRINT CLASSIFICATION METRICS FROM CONFUSION MATRIX")
    print("print_confusion_matrix: PRINT CONFUSION MATRIX WITH HELPFUL VISUAL THEN PRINTS CLASSIFICATION REPORT")
    print()
    print("              ** TEST & FIT **")
    print('select_kbest: Words go here to describe the function')
    print('         rfe: Words go here to describe the function')
    print('         sfs: Words go here to describe the function')    
    print()    
    print("                  ** PLOT & GRAPH **")
    print('    cs_vis_types: Words go here to describe the function')    
    print('        sunburst: Words go here to describe the function')    
    print('visualize_scaler: Words go here to describe the function')
    print()
    print()

    
def eval_tips():
    """
    PRINT A LIST OF USEFUL FUNCTIONS, METHODS, AND ATTRIBUTES USED FOR EVALUATION
    """
    print("** USEFUL EVALUATION CODE**")
#    print ("DFNAME.head()")
#    print ("DFNAME.shape")
#    print ("DFNAME.shape[0] #read row count")
#    print ("DFNAME.describe().T")
#    print ("DFNAME.columns.to_list()")
#    print("DFNAME.COLUMNNAME.value_counts(dropna=False)")
#    print ("DFNAME.dtypes")
#    print("DFNAME.select_dtypes(include='object').columns")
#    print("DFNAME.select_dtypes(include='float').columns")
#    print("pd.crosstab(DFNAME.COLUMN-1, DFNAME.COLUMN-2)")



def print_class_metrics(actuals, predictions):
    """
    This Function was adapted and slightly altered 
    from original code provided by Codeup instructor Ryan McCall.
    It provides classification metrics using confusion matrix data.
    """
    TN, FP, FN, TP = confusion_matrix(actuals, predictions).ravel()
    
    ALL = TP+TN+FP+FN
    negative_cases = TN+FP
    positive_cases = TP+FN
    
    accuracy = (TP+TN)/ALL
    print(f"Accuracy: {accuracy}")

    true_positive_rate = TP/(TP+FN)
    print(f"True Positive Rate: {true_positive_rate}")

    false_positive_rate = FP/(FP+TN)
    print(f"False Positive Rate: {false_positive_rate}")

    true_negative_rate = TN/(TN+FP)
    print(f"True Negative Rate: {true_negative_rate}")

    false_negative_rate = FN/(FN+TP)
    print(f"False Negative Rate: {false_negative_rate}")

    precision = TP/(TP+FP)
    print(f"Precision: {precision}")

    recall = TP/(TP+FN)
    print(f"Recall: {recall}")

    f1_score = 2*(precision*recall)/(precision+recall)
    print(f"F1 Score: {f1_score}")

    support_pos = TP+FN
    print(f"Support (0): {support_pos}")

    support_neg = FP+TN
    print(f"Support (1): {support_neg}")
    
    # this will return the Series header for y if defined by target= 
        # when conducting split but throws an error if not defined.
    #print(f"Target Feature: {target}, is set for Positive")
    # y_validate.name


    
def print_confusion_matrix(actuals, predictions):
    """
    This function returns the sklearn confusion matrix with a helpful visual
    and then returns the classification report.
    """
    print('sklearn Confusion Matrix: (prediction_col, actual_row)')
    print('                          (Negative_first, Positive_second)')
    print(confusion_matrix(actuals, predictions))
    print('                       :--------------------------------------:')
    print('                       | pred Negative(-) | pred Positive (+) |')
    print(' :---------------------:------------------:-------------------:')
    print(' | actual Negative (-) |        TN        |    FP (Type I)    |')
    print(' :---------------------:------------------:-------------------:')
    print(' | actual Positive (+) |   FN (Type II)   |         TP        |')
    print(' :---------------------:------------------:-------------------:')
    print()
    print(classification_report(actuals, predictions))


    
######################### TEST & FIT #########################

def select_kbest(X, y, k=2):
    """
    Select K Best
    
    - looks at each feature in isolation against the target based on correlation
    - fastest of all approaches covered in this lesson
    - doesn't consider feature interactions
    - After fitting: `.scores_`, `.pvalues_`, `.get_support()`, and `.transform`
    
    Imports needed:
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    
    Arguments taken:
    X = predictors
    y = target
    k = number of features to select
    """
    
    kbest = SelectKBest(f_regression, k=k)
    _ = kbest.fit(X, y)
    
    X_transformed = pd.DataFrame(kbest.transform(X),
                                   columns = X.columns[kbest.get_support()],
                                   index = X.index)
    
    return X_transformed.head()

def rfe(X, y, k=2):
    """
    Recursive Feature Elimination (RFE)
    
    - Progressively eliminate features based on importance to the model
    - Requires a model with either a `.coef_` or `.feature_importances_` property
    - After fitting: `.ranking_`, `.get_support()`, and `.transform()`
    
    Imports Needed:
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    
    Arguments taken:
    X = predictors
    y = target
    k = number of features to select
    """
    
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(X, y)
    
    X_transformed = pd.DataFrame(rfe.transform(X),
                                 index = X.index,
                                 columns = X.columns[rfe.support_])
    
    return X_transformed.head()

def sfs(X, y, k=2):
    """
    Sequential Feature Selector (SFS)
    
    - progressively adds features based on cross validated model performance
    - forwards: start with 0, add the best additional feature until you have the desired number
    - backwards: start with all features, remove the worst performing until you have the desired number
    - After fitting: `.support_`, `.transform`
    
    Imports Needed:
    from sklearn.feature_selection import SequentialFeatureSelector
    
    Arguments taken:
    X = predictors
    y = target
    k = number of features to select
    """
    
    model = LinearRegression()
    sfs = SequentialFeatureSelector(model, n_features_to_select=k)
    sfs.fit(X, y)
    
    X_transformed = pd.DataFrame(sfs.transform(X),
                                 index = X.index,
                                 columns = X.columns[sfs.support_])
    
    return X_transformed.head()






######################### PLOT & GRAPH #########################

# IMPORTS NEEDED FOR PLOTS AND GRAPHS

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# import preprocessing
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer



def cs_vis_types():
    print('Types of Visualization')
    print()
    print()
    print('- **Univariate Distributions**')
    print()
    print('    - Continuous variable distributions')
    print('        - histogram')
    print('        - boxplot')
    print('        - displot')
    print()      
    print('    - Discrete variable distributions')
    print('        - countplot')
    print()
    print()
    print('- **Bi- and multi-variate relationships**')
    print()
    print('    - Continuous with Continuous')
    print('        - scatter')
    print('        - line')
    print('        - pairplot')
    print('        - heatmap')
    print('        - relplot')
    print()          
    print('    - Discrete with Continuous')
    print('        - violin')
    print('        - catplot')
    print('        - sunburst')
    print('        - boxplot')
    print('        - swarmplot')
    print('        - striplot')
    print()
    print('    - Discrete with Discrete')
    print('        - heatmap')

            
            
def sunburst(df, cols, target):
          """
          This function will map out a plotly sunburst which is a form of correlated heat map.
          It does not work well on continuous values and is more suited for distinct values.
          """
          
          fig = px.sunburst(df, path = cols, values = target)
          return fig.show()

        
        
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

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
######################### WORKING #########################

# BUILD A FUNCTION THAT DOES THIS FOR ALL "FLOAT" COLUMNS
# float_cols = train_iris.select_dtypes(include='float').columns

# Plot numeric columns
#plot_float_cols = float_cols 
#for col in plot_float_cols:
#    plt.hist(train_iris[col])
#    plt.title(col)
#    plt.show()
#    plt.boxplot(train_iris[col])
#    plt.title(col)
#    plt.show()

# BUILD A FUNCTION THAT DOES THIS FOR ALL "OBJECT" COLUMNS
# train.species.value_counts()
# plt.hist(train_iris.species_name)

# BUILD A FUNCTION THAT DOES THIS
#test_var = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#for var in test_var:
#    t_stat, p_val = t_stat, p_val = stats.mannwhitneyu(virginica[var], versicolor[var], alternative="two-sided")
#    print(f'Comparing {var} between Virginica and Versicolor')
#    print(t_stat, p_val)
#    print('')
#    print('---------------------------------------------------------------------')
#    print('')

# sns.pairplot(DF, hue='TARGET_COLUMN', corner=True)
# plt.show()

# BUILD A FUNCTION; This will list out Accuracies for each model
# accuracy_dictionary = {'Baseline': (petpics_df.actual == petpics_df.baseline).mean(), 
#                   'Model_1 accuracy': (petpics_df.actual == petpics_df.model1).mean(),
#                   'Model_2 accuracy': (petpics_df.actual == petpics_df.model2).mean(),
#                   'Model_3 accuracy': (petpics_df.actual == petpics_df.model3).mean(),
#                   'Model_4 accuracy': (petpics_df.actual == petpics_df.model4).mean()}
# accuracy_dictionary


# ```
# {'Baseline': 0.6508,
#  'Model_1 accuracy': 0.8074,
#  'Model_2 accuracy': 0.6304,
#  'Model_3 accuracy': 0.5096,
#  'Model_4 accuracy': 0.7426}
#  ```


# Wraps Codeup Instructor Ryan McCall's Classification Matrix Function into a single print event with Named Headers.

# Actual = 

#print('** Baseline:')
#print(print_classification_metrics(petpics_df.actual, petpics_df.baseline))

#print('')
#print('** Model_1:')
#print(print_classification_metrics(petpics_df.actual, petpics_df.model1))

#print('')
#print('** Model_2:')
#print(print_classification_metrics(petpics_df.actual, petpics_df.model2))

#print('')
#print('** Model_3:')
#print(print_classification_metrics(petpics_df.actual, petpics_df.model3))

#print('')
#print('** Model_4:')
#print(print_classification_metrics(petpics_df.actual, petpics_df.model4))

