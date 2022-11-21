# 2017 Tax Assesed Home Value Prediction Models


# Project Overview:
This project has been tasked with collecting, cleaning and analyzing Zillow data from 2017 in order to improve a previous prediction model that was designed to predict the Tax Assessed Home Value for Single Family Properties based on available realestate data.


# Goals: 
* Predict property tax assessed values of Single Family Properties
* Outperform existing logerror model
* Recommend improvements for a more accurate model
* Define relevant fips codes for our data


# Reproduction of this Data:
* Can be accomplished using a local ```env.py``` containing ```user, password, host``` information for access to the Codeup SQL database server.
* All other step by step instructions can be found by reading the below Jupyter Notebook files located in my [proj-2_zillow](https://github.com/QMCBT-JustinEvans/project-2_zillow.git) github repository.
    * 01_wrangle
    * 02_explore
    * 03_model


# Acquire

* ```zillow``` data from Codeup SQL database was used for this project.
* The data was initially pulled on 15-NOV-2022.
* The initial DataFrame contained 52,441 records with 69 features  
    (69 columns and 52,441 rows) before cleaning & preparation.
* Each row represents a Single Family Property record with a Tax Asessment date within 2017.
* Each column represents a feature provided by Zillow or an informational element about the Property.


# Prepare

**Prepare Actions:**

* **Whitespace:** Removed 52,441 Whitespace characters.
* **REFORMAT:** Reformatted 13 columns containing 596,382 NaN entries to 0.
* **CONVERT dtypes:** Convert dtypes to accurately reflect data contained within Feature.
* **FEATURE ENGINEER:** Use Yearbuilt to create Age Feature, Drop yearbuilt for redundancy; create Feature to show ratio of Bathrooms to Bedrooms.
* **fips CONVERSION:** Use fips master list to convert fips to county and state, Drop state for redundancy.
* **PIVOT:** Pivot the resulting county column from fips conversion to 3 catagorical features. 
* **DROP:** Dropped 27 Columns unecessary to data prediction (ie.. index and redundant features).
* **REPLACE:** Replaced conditional values in 2 columns to transform into categorical features.
* **RENAME:** Columns for Human readability.    
* **REORDER:** Rearange order of columns for human readability.   
* **DROP 2:** Drop Location Reference Columns unsuitable for use with ML without categorical translation.
* **CACHE:** Write cleaned DataFrame into a new csv file ('zillow_2017_cleaned.csv').  
* **ENCODED:** No encoding required.
* **MELT:** No melts needed.


# Summary of Data Cleansing
* Cleaning the data resulted in less than 10% overall record loss

* DROP NaN COLUMNS: 39 features each containing over 30% NaN were dropped; resulting in no record loss.
*    DROP NaN ROWS: 1,768 records containing NaN across 13 features were dropped; resulting in only 3% record loss.
*         OUTLIERS: Aproximately 3,000 outliers were filtered out in an attempt to more accurately align with realistic
                    expectations of a Single Family Residence; resulting in less than a 6% decrease in overall records.
*           IMPUTE: No data was imputed

* logerror: The original logerreror prediction data was pulled over and prepared with this DataFrame for later comparison in order to meet the requirement of improving the original model.  
    
* Note: Special care was taken to ensure that there was no leakage of this data.


# Split

* **SPLIT:** train, validate and test (approx. 60/20/20), stratifying on target of 'churn'
* **SCALED:** no scaling was conducted
* **Xy SPLIT:** split each DataFrame (train, validate, test) into X (selected features) and y (target) 


## A Summary of the data

### There are 28,561 records (rows) in our training data consisting of 18 features (columns).
* There are 7 categorical features made up of only 2 unique vales indicating True/False.
* There are 5 categorical features made up of multiple numeric count values.
* There are 6 continuous features that represent measurements of value, size, time, or ratio.


# Explore

* Exploration of the data was conducted using various Correlation Heat Maps, Plot Variable Pairs, Categorical Plots, and many other graph and chart displays to visualize Relationships between independent features and the target as well as their relationships to eachother. 

    
* Each of the three selected features were tested for a relationship with our target of Tax Assesed Value.
    1. Bedrooms
    2. Bathrooms
    3. Property Squarefeet  
    
    
* All three independent features showed a significant relationship with the target feature.

* Three statistical tests were used to test these questions.
    1. T-Test
    2. Pearson's R
    3. $Chi^2$


# Exploration Summary

## Generally speaking the majority of the features in our DataFrame have a linear Relationship with our taeget but the Features with the most significant relationship were:

  1. Property Squarefeet ```('calculatedfinishedsquarefeet')```
  2. Bathrooms ```('bathroomcny')```
  3. Bedrooms ```('bedroomcnt')```     
    
* We found that Property SQFT has the highest correlation with Tax Assessed Property Value at 48%
* The number of Bathrooms comes in at a close second at 44%
* However, the number of Bedrooms (still the third highest correlation) only scores a 24%

## Takeaways:
* Features available are sufficient to conduct predictions
* We could benefit greatly from additional data and Feature Engineering of Location Data
    
## Features that will be selected for Modeling:
* Our target feature is Tax Assessed Property Value ```('taxvaluedollarcnt')```
* Our selected features are:
    1. Property Squarefeet ```('calculatedfinishedsquarefeet')```
    2. Bathrooms ```('bathroomcny')```
    3. Bedrooms ```('bedroomcnt')```


# Modeling

* Baseline and logerror are our evaluation metrics  
* Our Target feature is the Tax Assessed Property Value (taxvaluedollarcnt) 
    
* **Baseline** This would be simply guessing the Average Tax Assessment Property Value as our prediction every time.

* **Logerror** These are the results of the previous team's predictive model.

* Models will be developed and evaluated using six different regression model types: 
    1. Simple Model
    2. Polynomial Degree 2
    3. Polynomial Only Interaction
    4. Lasso-Lars
    5. Generalized Linear Model
    6. Baseline  
    
    
* Models will be evaluated on train and validate data
* The model that performs the best will ultimately be the one and only model evaluated on our test data 


## Comparing Models

* None of the models are anywhere close to being in danger of overfit
* Both of the polynomial models performed at the top
* The Lasso Lars model was not that far behind the polynomials
* Simple LM was a little farther behind but still fairly close
* Baselin and GLM were nearly identical and both performed with a significantly higher rate of error
* logerror did not even beat baseline
    
    
## ```2nd Degree Polynomial``` is the best model and will likely continue to perform well above Baseline and logerror on the Test data.

# Evaluate on Test: Best Model (```2nd Degree Polynomial```)

# Conclusions

* We were asked to find a model that performs better than the logerror model currently on file
    * Every model we ran would have outperformed the current logerror model including baseline
    * The ```2nd Degree Polynomial``` model is the best performing model,  
  it has outperformed all other models that we evaluated,  
  including both the current logerror model and the Baseline.
    
* We were asked to include a crossreference of the fips to the corresponding location  
    * That information was pulled in using a master .csv file
    * It is now hard coded into the DataFrame used for this project
    * It is also listed in the Dictionary and below
    
|fips|State|County|
|:--:|:---:|:-----|
|6037|CA   |Los Angeles County|    
|6059|CA   |Orange County| 
|6111|CA   |Ventura County|  


## Recommendations

* Use the 2nd Degree Polynomial model
* Continue to collect data and feature engineer


## Next Steps

## Next Steps
* During our Modeling, a run of ```Multiple Regression + RFE``` revealed that the two top features were the Orange County and Ventura County categorical features.
* Although this did not align with our correlation tests and we did not pursue it
* It proves that these features have significance
    
### Given more time...
* We could spend some time converting Location Data into something useful for Machine Learning
* We could scrape Zip Code Income, Population and Demographics to include in the DataFrame
* [Name Census](https://namecensus.com/zip-codes/california/orange-county/#:~:text=Orange%20County%20makes%20up%20approximately,information%20for%20each%20zip%20code) keeps all of this data
* Here is an example: <a href="https://namecensus.com/demographics/california/90620/">90620 Zip Code Income, Population and Demographics</a></div>

REF: 90620 Zip Code Income, Population and Demographics. NameCensus.com. Retrieved from https://namecensus.com/demographics/california/90620/.
