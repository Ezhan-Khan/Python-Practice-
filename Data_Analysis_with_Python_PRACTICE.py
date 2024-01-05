# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:45:18 2023

@author: Ezhan Khan
"""
                     #  DATA ANALYSIS with PYTHON


#Will cover IMPORTANT 3 LIBRARIES - 'NumPy', 'Pandas' and 'Klearn' 
# (used in MANY Industries - Healthcare, Finance, Insurance, Internet...)

# ALSO will learn to BUILD 'Data PIPELINES'
# MACHINE LEARNING MODELS for PREDICTIONS on REAL-WORLD Datasets


#%%          CREATING 'PLOTS' in PYTHON (FUNDAMENTALS)
from matplotlib import pyplot as plt
#'%matplotlib inline' needed for JUPYTER NOTEBOOKS, since want to DISPLAY Graphs IN the NOTEBOOK INTERFACE ITSELF
import seaborn as sns

#             MATPLOTLIB FUNCTIONS:

#Line Plot  -  'plt.plot(x,y)'

#Scatter Plot  -  'plt.scatter(x,y)'

#Histogram Plot  -  'plt.hist(x, bins)'    ('y-axis' = COUNT/NUMBER of VARIABLES in EACH BIN)
#note: add ' edgecolor="color" ' argument for More Clarity = BINS 'COLOURED OUTLINE'

#Bar Plot  -  'plt.bar(x, height)'
# x-axis = Categorical Variable,  'height' = NUMBER of VALUES for EACH CATEGORY
#Adjust the 'WIDTH' of 'EACH BAR' by adding 'width' argument

#'PSEUDO COLOR' Plot (HEATMAP)  -  'plt.pcolor(C, cmap = "Color Scheme")'
# - display 'MATRIX DATA/DataFrame' as ARRAY of COLOURED CELLS (FACES).
# COLOURS of Vertices are SPECIFIED by MATRIX Argument 'C'


#             SEABORN FUNCTIONS:

# 'REGRESSION' Plot  -  'sns.regplot(x='header_1', y = 'header_2', data=df)'
#Draws SCATTER PLOT for 'x' and 'y' Variables, 'FITTING REGRESSION MODEL' and PLOTS the RESULTING 'REGRESSION LINE' with '95% Confidence interval'.
# 'x' and 'y' parameters are SHARED as DataFrame HEADERS, with DataFrame as Argument too. 
#OPTIONAL - could add 'line_kws = {"color":""}' - adds 'SPECIFIC COLOUR' for the 'REGRESSION LINE'


# 'BOX and WHISKER' (Box Plot)  -  sns.
#Useful for 'COMPARISONS' of Quantitative Data Distributions BETWEEN Variables OR across 'LEVELS' of 'CATEGORICAL VARIABLES' 
#Box shows 'QUARTILES' of Dataset, 'Whiskers' show REST of Distribution, OUTLIERS are OUTSIDE Plot
#'WHISKERS' show LOWER 25% (from MIN Value to Q1) and UPPER 25% (from MAX Value down to Q3)
#'IQR' = Q3 - Q1


# 'RESIDUAL' Plot  -  sns.residplot(data=df, x=df['header_1'], y=df['header_2'])
#displays 'QUALITY' of 'POLYNOMIAL REGRESSION'
#regresses on 'y on x', then Draws SCATTER PLOT of RESIDUALS
#'RESIDUALS' = 'DIFFERENCE' between 'OBSERVED and PREDICTED' Values of 'DEPENDENT VARIABLE' (y-axis variable)
#(i.e. HOW MUCH does a Regression Line VERTICALLY MISS a Data Point?)
#   -  HOW FAR OFF are 'PREDICTIONS' from 'ACTUAL' (observed) Data Points?


# 'KDE' Plot  -  sns.kdeplot(X)
# (Kernel Density Estimate) - creates 'PROBABILITY DISTRIBUTION CURVE' of Data
#Based on LIKELHOOD of OCCURENCE on a SPECIFIC VALUE
#Created for a SINGLE VECTOR of INFORMATION
#Compares 'LIKELY CURVES' of ACTUAL DATA with PREDICTED DATA


# 'DISTRIBUTION' Plot  -  sns.distplot(X, hist=True or False)
#CAN be used to COMBINE 'Histogram' WITH 'KDE'
# = DISTRIBUTION CURVE, which USES 'BINS of HISTOGRAM' as 'REFERENCE for ESTIMATION'
#Can USE this INTERCHANGEBLY with 'KDE' (above) -  SAME EACT THING!!!
#(note: if 'hist=False', is JUST the KDE, but if 'hist=True' it PLOTS the Histogram WITH the DISTRIBUTION Plot)




#%%                      MODULE 1  - IMPORTING DATASETS


#Python Libraries contain many 'built-in' modules
#EACH can have DIFFERENT APPLICATIONS:

# 'Scientific Computing' Libraries -  Pandas (Data Structures), NumPy (arrays and matrices), SciPy (Integrals, Solving DIFFERENTIAL EQUATIONS, OPTIMIZATION)

# 'Visualization' Libraries - 'Matplotlib' (MOST POPULAR), 'Seaborn' (based ON Matplotlib - great for Heat Maps, Violin Plots, Time Series like 'Histograms')

# 'ALGORITHMIC' Libraries (BUILD MODEL using Data Set, obtain PREDICTIONS) 
#  -  'Skikit-learn' Library  (ML, Regression, CLassification...), 'Statsmodels' (also for Statistical Models and Statistical TESTS)

#                IMPORTING DATA:
#Depends on 'FORMAT' (.csv, .json, .xlsx...)  and 'FILE PATH' ('/Desktop/mydata.csv' or INTERNET 'https://archive.ics.uci.edu/autos/imports-85.data')
#Initially, Data wont make sense - but ONCE we READ IT IN, should MAKE SENSE!
#       KNOW THIS! Just RECAPPING:
#Import CSV (as Pandas DataFrame) - 'pd.read_csv(url, header=None)'     - CSV has NO Header Row, so have SPECIFIED this!
#PRINT the DataFrame with 'df.head()) or 'df.tail()'
#    df.columns = ["list of HEADERS"]  # - ADDS 'Headers' TO the File!

#SAVE/'EXPORT' Pandas DataFrame TO 'CSV file' using 'df.to_csv(path.csv)' - Now have STORED as CSV FILE!


# MAIN 'DATA TYPES' in PYTHON  -  'object' (string), 'int64' (int), 'float64' (float), 'Datetime64' (see 'datetime Module' for more)
#MAY need to 'MANUALLY CORRECT' any 'WRONG Data Types'
df.dtypes  #VIEW 'DATA TYPES' for 'EACH COLUMN' of a DataFrame
#e.g. 'bore' column is 'object/STRINGS' (but SHOULD be CONVERTED to 'numeric', since represents 'DIMENSIONS of CAR ENGINE')
             #'SUMMARY STATISTICS':
df.describe()   # count, mean, std, min, max, quartiles as '25%, 50% and 75%'
s# '.describe()' will IGNORE COLUMNS with 'NON-NUMERICAL' Values.
#IF we want to ALSO SEE 'NON-Numeric' Columns IN SUMMARY, add ' include = "all" ' argument. 
df.describe(include = "all")
#Will get 'NaN' ('Not a Number's) in the Summary Table, for 'NUMERIC Functions' performed on OBJECT/STRINGS (since is NOT APPROPRIATE for them)

#For 'STRINGS', also get STATS on 'unique' (number of unique column values), 'top' (MOST COMMON String Value), 'freq' ('frequency' of TIMES that 'top' object/string APPEARS)


import pandas as pd
import numpy as np

# 'IMPORTING CSV' EXAMPLE  -  will use Dataset on 'Used Automobiles' CSV File
filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df = pd.read_csv(filepath, header = None)   #LOADING the DF, using a FILEPATH/URL here!
#note: here, do NOT have a 'Header Row', so may have to ADD one!
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
#ADD these 'HEADERS' TO the DataFrame:
df.columns = headers
df.columns.values   #see that HEADERS have been ADDED!
#'Attribute/Column Variables' - 'symboling', 'normalized-losses', 'make'...

#'REPLACE' EACH Occurance of '?' with 'np.NaN' (NaN) in the ENTIRE DataFrame:
df1 = df.replace("?", np.NaN)
#Now can use 'DROP NA' to DROP all 'MISSING VALUES (NaN)':
df1 = df1.dropna(subset=["price"], axis=0)   #ALONG the 'price' Column, want to DROP all 'NaN'/MISSING VALUES!
# use 'subset = ["price"]' so JUST REMOVES 'NaN' from 'price' Column SPECIFICALLY!

df1.dtypes   #view Data Types for EACH Column/Attribute
#As we see, May NEED to CHANGE 'DATA TYPES' for a few columns...
summary_stats = df1.describe(include = "all")
#(statitsical summary, for ALL 26 Columns)

# '.describe()' for 'SPECIFIC COLUMNS ONLY' (selecting):
df1[['length', 'compression-ratio']].describe()

df1.info()  #looking at SPECIFIC 'INFO' about the Columns!

#Finally SAVING this DataFrame TO a 'CSV' (EXPORT):
df1.to_csv("automobile.csv", index=False)  



#' EXAMPLE 2' - 'Laptop Pricing' Dataset:
url=  "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_base.csv"
laptop_df = pd.read_csv(url, header = None)
print(laptop_df.head())    
len(laptop_df)   #'238 Rows'
#ADDING HEADERS (as usual!):
laptop_df.columns = ["Manufacturer", "Category", "Screen", "GPU", "OS", "CPU_core", "Screen_Size_inch", "CPU_frequency", "RAM_GB", "Storage_GB_SSD", "Weight_kg", "Price"]
laptop_df   #Adding HEADER ROW for EACH Column
#REPLACE '?' with 'NaN':
laptop_df = laptop_df.replace("?", np.NaN)
print(laptop_df.dtypes)  #yypes for EACH COLUMN
#Getting SUMMARY STATS for ALL COLUMNS (including 'strings/Object' Type)
print(laptop_df.describe(include = "all"))
print(laptop_df.info())  #print EXTRA INFO on 'Non Null Count' and 'Dtype', as well as 'Memory Usage'. 






#%%                      MODULE 2  -  Data PRE-PROCESSING
#   (DATA 'WRANGLING', 'PRE'-Processing, 'MISSING VALUES', 'FORMATTING/NORMALIZATION' of Data, Data BINNING, DUMMY Variables)

#Note: Data 'Pre-Processing' is just ANOTHER WORD for 'Data-Cleaning/Wrangling'
# = CONVERT/MAP Data from RAW FORM to ANOTHER Format so it is PREPARED for ANALYSIS!
#Values in DataFrame are Changed/Modified BY COLUMN:
#   Here will use SAME 'df' - AUTOMOBILE DataSet ('COPIED FROM ABOVE'!)
filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df = pd.read_csv(filepath, header = None)   
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers
df.columns.values   #see that HEADERS have been ADDED!

df[["symboling"]] = df[["symboling"]] + 1  #KEEPS ADDING '1' to the 'symboling' COLUMN VALUES, in the DataFrame!



#                     DEALING with 'MISSING VALUES':
#Usually given a "?", 'NaN' "N/A", "0" or BLANKS
df.info()   #lets us IDENTIFY COLUMNS with MISSING VALUES (Less 'non-null' that Expected!)

#                          SEVERAL OPTIONS:

# 1. Check 'Data COLLECTION SOURCE' - WHY is Data missing? Maybe fill it in IF we FIND OUT WHY?

# 2. 'DROP' the MISSING Values - best to DROP that (ROW) VALUE (NOT the ENTIRE Column/Variable!)
df.dropna(subset=["specific column"], axis=0, inplace = True)

# 3.'REPLACE Data' (could replace with 'AVERAGE' of 'SIMILAR DATAPOINTS')
#e.g. get AVERAGE Value of the Column to REPLACE the MISSING Value
mean = df[['normalized-losses']].mean()   
df["normalized-losses"].replace(np.nan, mean)  #e.g. Replaces All MISSING VALUES from 'Normalized-Losses' column with the 'COLUMN's MEAN'
#ALTERNATIVE WAY to DO this is with 'df.fillna(value)', which FILLS 'NA' Values with a SPECIFIC VALUE!


#4. What about CATEGORICAL Variables? - Replace by 'MODE' (MOST COMMON/'FREQUENCY')

#5. Replace BASED on 'OTHER FUNCTIONS (using either 'replace' or 'fillna' again)

#6. LEAVE Missing Data AS IS - 'PANDAS' usually DOES NOT INCLUDE NA Values during DATA ANALYSIS Calculations or VISUALIZATIONS. So DONT NEED TO WORRY!

#e.g.2 - 'price' column has 4 Rows of MISSING Values '?'
print(df[df['price'] == '?']['price']) #obtains the 'price' Column SPECIFICALLY with ['price'] AFTER FIRST fitlering DataFrame for df['price']=='?'  
#Could REPLACE with 'np.NaN' Values:
df['price'].replace('?', np.nan, inplace=True)
#'inplace = True' - OVERWRITES the ORIGINAL DataFrame Values ("?" WITH 'NaN' instead!)

#Note - MISSING DATA is MORE IMPORTANT to DEAL WITH if planning to do 'MACHINE LEARNING'!
#IF a 'COLUMN' has 'MORE THAN 50% MISSING' Data = DROP ENTIRE COLUMN! WONT be Useful! So REMOVE!

#7. 'INTERPOLATE' - FILL IN Values BASED on VALUES SURROUNDING it  (using Special Equation!)






#               Data 'FORMATTING/STANDARDIZATION' (different units, conventions...)    
#Different places, different formats - NEED a CONSISTENT FORMAT!
#e.g. 'N.Y.', 'New York', 'NY' - DIFFERENT way to WRITE the SAME THING!
#But - BETTER to have a CONSISTENT Format!

#Converting 'mpg' TO 'L/100km':
df['city-mpg'] = 235/df["city-mpg"]  #SIMPLE CALCULATION, OVERWRITES DF COLUMN Values!
df.rename(columns={"city-mpg":"city-L/100km"}, inplace = True)
df['city-L/100km']

#                 WHAT IF 'WRONG DATA TYPE' for certain columns?
df["price"]   #has 'object'/STRING' Datatype! Should CHANGE THIS!
#SHOULD be 'float' or 'integer'!
df.dtypes   
#CONVERT DataType of a COLUMN using 'df[column].astype("type")'
df["price"] = df["price"].astype("float64")



#                Data 'NORMALIZATION' (='SCALING/CENTRING' Data for BETTER COMPARISONS) 
#NOTE - IMPORTANT for 'MODEL ANALYSIS' LATER ON!!!
#NORMALIZE Variables so 'RANGE of Values' is 'CONSISTENT' and EASIER to USE for ANALYSIS!
#(makes some Statistical Analysis EASIER in future!)
#If 2 Columns Values are NOT NORMALIZED, can be HARD to COMPARE during LATER Analysis!
#SO? - can PUT the VALUES of the Columns into SIMILAR RANGES - so Analytical Models are more BALANCED/LESS BIAS!

#                    '3 NORMALIZATION METHODS':
#1.  'Simple Feature Scaling' -   'x_new = x_old / x_max'  -  NEW Values range between '0-1'  -  DIVIDES EACH Value BY the 'MAX VALUE' in that feature.  
#2.  'Min-Max'  -  x_new = (x_old - x_min) / (x_max - x_min)   -  new values range between '0-1'
#3. 'Z-score'  -  x_new = (x_old - MEAN) / SD  -  NEW Values typically range around '0' (-3 to +3 usually)

#e.g. 'length' Column values are MUCH LARGER than 'width' or 'height' - so can NORMALIZE them:
df[['length', 'width', 'height']]  
#1. using 'Simple Feature Scaling':
df['length1'] = df['length']/df['length'].max()
#2. using 'Min-Max' Method:
df['length2'] = (df['length'] - np.min(df['length'])) / (np.max(df['length']) - np.min(df['length']))
#3. using 'Z-Score' method:
df['length3'] = (df['length'] - np.mean(df['length']))/(df['length'].std())
#VIEWING the CHANGES to 'length':
print(df[['length', 'length1', 'length2', 'length3']].head())



#               Data 'BINNING' = Create 'BIGGER CATEGORIES' from NUMERICAL Values (USEUFL for COMPARING 'GROUPS')
# i.e. CONVERT 'NUMERIC' INTO CATEGORICAL Variables (GROUP 'NUMERICAL' Values into 'BINS)
# Helps to 'PLOT HISTOGRAM FIRST', to 'DECIDE' on 'APPROPRIATE NUMBER OF BINS'
#  e.g. 'price' could be GROUPED by BINS into RANGE, from '5000 TO 45,500':
#  CATEGORIZE by 'low', 'mid' and 'high' price ranges:
#First GROUP them INTO 'Bins'
bins = np.linspace(min(df['price']), max(df['price']), 4)
#provide GROUP NAMES:
group_names = ["Low", "Medium", "High"]
#Then use df['new_binned_column'] = 'pd.cut(column, bins, labels, include_lowest = True )'
df['price_binned'] = pd.cut(df['price'], bins, labels=group_names, include_lowest=True)
df[['price', 'price_binned']]  #See that 'prices' have been GROUPED INTO 'BINS' (Low, Medium and 'High')
#Now can PLOT 'HISTOGRAM' using 'price_binned' AS Histogram 'BINS'...  -  COOL!



#              Creating 'INDICATOR/DUMMY VARIABLES' - 'CATEGORICAL' Variables INTO 'NUMERIC' (QUANTITATIVE) Variables:
#MOST 'Statistical Models' CANNOT take in 'OBJECTS/STRINGS' TYPES as input 
#(Most can ONLY take 'NUMBERS')

#e.g. Converting 'Fuel Type' (Categorical) INTO 'NUMERIC':

#1. Add 'DUMMY VARIABLES/SEPARATE COLUMNS' for EACH UNIQUE Category. 
#i.e. 'gas' column, 'diesel' column...
#2. Assign '0 or 1' for EACH CATEGORY (1=Yes, is that Fuel Type, 0 = No, Not that Fuel Type)
# (this is often called 'One Hot Encoding')

#Use PANDAS FUNCTION to DO this!  'pd.get_dummies()'
pd.get_dummies(df['fuel-type'])  
#converts 'fuel-type' into SEPARATE 'DUMMY VARIABLE Columns' for 'diesel' and 'gas' (0 or 1...)
#               - COOL and Very SIMPLE!




#                       PRACTICE 1 ( = OVERVIEW of ALL ABOVE!):

  #using usual 'Automobile Dataset'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df = pd.read_csv(filepath, header = None)   
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers
df.columns.values   #Viewing the Added 'Column Headers'

#Replacing ALL "?" values with 'NaN':
df.replace("?", np.nan, inplace = True)
df.head()  #see, that now have 'NaN' Values instead of '?'
#IDENTIFY any MISSING VALUES:
missing_data = df.isnull()
missing_data.head()  #becomes 'TABLE of BOOLEAN VALUES', where 'True = Missing/NULL', 'False = NOT NULL'
#COUNT the Number of NULL Values for EACH COLUMN:
for column in missing_data.columns.values:
    print(column)
    print(missing_data[column].value_counts())
    print("")   #NICELY Lists the COUNT of Values ('True' and 'False') for EACH Column 

#Either  1. DROP the MISSING Values (whole Column or Row)
#Or      2. REPLACE Data by - Column MEAN. FREQUENCY (MODE) or OTHER FUNCTIONS

mean_norm_loss = np.mean(df['normalized-losses'].astype("float64"))
print("Average of normalized-losses:", mean_norm_loss)     # = '122.0'
#REPLACE 'NaN' values in 'normalized-losses' column WITH 'mean':
df["normalized-losses"].replace(np.nan, mean_norm_loss, inplace = True)
df["normalized-losses"].head()  #can see that NAN values have been REPLACED with '122'
#REPEAT the SAME PROCCESS for 'bore' column:
mean_bore = np.mean(df['bore'].astype('float64'))
mean_bore   #3.330
df['bore'].replace(np.nan, mean_bore, inplace=True)
#REPEAT Same Process for 'stroke' column:
mean_stroke = np.mean(df['stroke'].astype('float64'))
mean_stroke      # '3.255'    
df['stroke'].replace(np.nan, mean_stroke, inplace=True)
#REPEAT for 'horsepower' Column:
mean_horsepower = np.mean(df['horsepower'].astype('float64'))
print(mean_horsepower)    # '104.256' 
df['horsepower'].replace(np.nan, mean_horsepower, inplace=True)
#REPEAT for 'peak-rpm' Column:
mean_peakrpm = np.mean(df['peak-rpm'].astype('float64'))
print(mean_peakrpm)    #'5125.37'
df['peak-rpm'].replace(np.nan, mean_peakrpm, inplace=True)

# REPLACING with 'MOST FREQUENT' CATEGORY (MODE):
#Use '.value_counts()' WHENEVER we want COUNT for DIFFERENT VALUES found IN a COLUMN:
print(df['num-of-doors'].value_counts())   
# 114 for 'four' doors, 89 for 'two' doors  - MOST have '4' doors!
#'OR' can FIND the MOST COMMON Category with '.idmax()'
print((df['num-of-doors'].value_counts())   

#SAME THING! Just use '.replace()':
df['num-of-doors'].replace(np.nan, "four", inplace=True)
#so all MISSING Values are REPLACED with "four" (most common)

#Or, if we DROPPED a 'WHOLE ROW' with NA, in "price" Column:
df.dropna(subset = ["price"], axis=0, inplace = True)
df.reset_index(drop=True, inplace=True)
df.head()   #just RESETS the INDEX (since we DROPPED 2 Rows)
#ONLY DROP ROWS as a 'LAST RESORT'

#LASTLY, can CHECK the Different 'DATA TYPES':
df.dtypes  
#See that 'bore', 'price', 'peak-rpm' and 'stroke' SHOULD be 'FLOAT'
df[['bore', 'stroke', 'price', 'peak-rpm']] = df[['bore', 'stroke', 'price', 'peak-rpm']].astype("float64") 
#'normalized-losses' SHOULD be 'INTEGER'
df[['normalized-losses']] = df[["normalized-losses"]].astype("int")
df.dtypes   #should have CHANGED now!

#CHANGING Data to a COMMON FORMAT ('STANDARD')- 'mpg' to 'L/100km'
df['city-L/100km'] = 235/df['city-mpg']
df['city-L/100km']   
df['highway-L/100km'] = 235 / df['highway-mpg']
df['highway-L/100km']   #doing SAME for 'highway-mpg' to 'L/100km'

#Practicing 'NORMALIZATION' for 'length', 'width' and 'height' columns:
df['length'] = df['length']/np.max(df['length'])
df['width'] = df['width']/np.max(df['width'])  
df['height'] = df['height'] / np.max(df['height'])
#(NORMALIZED using 'Simple Feature Scaling' 
df[['length', 'width', 'height']].head()  
#NICE! Have NORMALIZED so ALL VALUES are WITIHN RANGE of '0-1'

#'BINNING' - could put 'horsepower' INTO BINS/RANGES of Values:
df['horsepower'] = df['horsepower'].astype('int64')
#PLOTTING 'HISTOGRAM' of Horsepower DISTRIBUTION:
plt.hist(df['horsepower'])  
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
#PLOTS a SIMPLE HISTOGRAM of 'Horsepower' DISTRIBUTION!
#Lets do this with '3 EQUAL-SIZED BINS' instead:
bins = np.linspace(np.min(df['horsepower']), np.max(df['horsepower']), 4)
#Note: divide using '4', since for '3 Bins', would need '4 DIVISIONS'
group_names = ['Low', 'Medium', 'High']
#Create New df Column FOR the BINNED CATEGORIES:
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)
df[['horsepower-binned', 'horsepower']].head()   #VIEWING Column Values with ASSOCIATED 'BIN GROUPS' Column

df['horsepower-binned'].value_counts()  # = 'COUNT' for EACH GROUP CATEGORY
# Low = 153, Medium = 43, High = 5  -  so MOST Cars are within the 'LOW' Horsepower Bin

#VISUALIZING using the BINS - FIRST as 'BAR CHART', then as 'HISTOGRAM'
plt.bar(group_names, df['horsepower-binned'].value_counts())
plt.xlabel('horsepower')
plt.ylabel('count')
plt.title('horsepower bins')     #MUCH CLEARER VISUALIZATION, when using 'BINS'!

plt.hist(df['horsepower'], bins=3)
plt.xlabel('horsepower')
plt.ylabel('count')
plt.title('horsepower bins')     #MUCH CLEARER VISUALIZATION, when using 'BINS'!


# DUMMY VARIABLES Practice - 'fuel-type' CATEGORIES into NUMERICAL Variables:
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()   #EASILY Created DUMMY VARIABLES for 'fuel-type' column
#Could then just RENAME the COLUMN Headers:
dummy_variable_1.rename(columns = {'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()   

#'CONCATENATE/ADDING ON 'dummy_variable_1' TO the DATAFRAME:
df = pd.concat([df, dummy_variable_1], axis=1)
df.columns.values  #using 'pd.concat([df, column(s)], axis=1)   'axis=1' ADDS ON 'as COLUMNS'
#DROPPING 'fuel-type' Column (no longer needed):
df.drop("fuel-type", axis=1, inplace=True)
df.head()

#More Practice - DUMMY VARIABLE for 'aspiration' Column:
df['aspiration'].value_counts()  #see that MOST are 'std' (just for fun!)
dummy_variable_2 = pd.get_dummies(df['aspiration'])
dummy_variable_2     #Columns for 'std' and 'turbo' are made
df = pd.concat([df, dummy_variable_2], axis=1)
df.drop("aspiration", axis=1, inplace=True)
df.head()  #removed 'aspiration' column - Since is 'NO LONGER NEEDED'!

#FINALLY? Just SAVE this CLEANED DataFrame to a CSV:
df.to_csv("Cleaned_Car_Data.csv")



#                      PRACTICE 2 - LAPTOP Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

laptop_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv"
df_laptop = pd.read_csv(laptop_url)
df_laptop.info()  #tells us COUNT of 'NON-NULL' Values IN the Column and DataType
df_laptop.head()

#Lets Convert 'Screen_Size_cm' Column Values to 'float' and ROUND Values to 2 d.p.
df_laptop["Screen_Size_cm"].astype("float64")
df_laptop[['Screen_Size_cm']] = np.round(df_laptop[['Screen_Size_cm']], 2)


#1. Evaluate any MISSING DATA:
missing_data = df_laptop.isnull()
for column in missing_data.columns.values:
    print(column)
    print(missing_data[column].value_counts())
    print("")
#See that 'Screen_Size_cm' AND 'Weight_kg' have MISSING DATA (NaN)

#Could REPLACE 'Weight_kg' NaN values with 'Column MEANS':
weight_mean = np.mean(df_laptop['Weight_kg'])     #'1.8622'
df_laptop['Weight_kg'].replace(np.nan, weight_mean, inplace = True)
df_laptop['Weight_kg']   #BEST for CONTINUOUS DATA to REPLACE with 'MEAN'
#Since 'Screen_Size_cm' is more 'CATEGORICAL'- REPLACE with 'MODE' (MOST FREQUENT):
df_laptop['Screen_Size_cm'].value_counts()  #'39.62' cm Screen is MOST COMMON!
most_frequent_screen = 39.62
df_laptop['Screen_Size_cm'].replace(np.nan, most_frequent_screen, inplace = True)


#2.  FIX DATA TYPES:
#Ensure 'Weight_kg' and 'Screen_Size_cm' are 'FLOAT' Data Types:
df_laptop[['Weight_kg', 'Screen_Size_cm']] = df_laptop[['Weight_kg', 'Screen_Size_cm']].astype("float") 
df_laptop[['Weight_kg', 'Screen_Size_cm']].dtypes  #Both FLOATS now - SIMPLE!


#3.   Data STANDARDIZATION (i.e. Unit Conversions):
#Convert 'Screen Size' from 'cm' TO 'inches' 
#AND Convert 'Weight' from 'kg' to 'pounds'
# (1 inch = 2.54 cm ,  1 kg = 2.205 pounds)
df_laptop['Screen_Size_cm'] = df_laptop['Screen_Size_cm'] / 2.52
df_laptop.rename(columns = {'Screen_Size_cm': 'Screen_Size_inches'}, inplace = True)
df_laptop['Weight_kg'] = df_laptop['Weight_kg'] * 2.205
df_laptop.rename(columns = {'Weight_kg':"Weight_pounds"}, inplace = True)
df_laptop.columns.values   #All 'Standardized' - Cool!


#4.   Data NORMALIZATION 
#Normalize 'CPU_frequency' column using MAXIMUM Column Value:
df_laptop['CPU_frequency'] = df_laptop['CPU_frequency'] / np.max(df_laptop['CPU_frequency'] )


#5.       BINNING:
#The 'Price' could be 'BINNED' into 'Low', 'Medium' and 'High'
bins = np.linspace(min(df_laptop['Price']), max(df_laptop['Price']), 4)
#provide GROUP NAMES:
group_names = ["Low", "Medium", "High"]
#Then use df['new_binned_column'] = 'pd.cut(column, bins, labels, include_lowest = True )'
df_laptop['price_binned'] = pd.cut(df_laptop['Price'], bins, labels=group_names, include_lowest=True)
df_laptop[['Price', 'price_binned']]  #See that 'prices' have been GROUPED INTO 'BINS' (Low, Medium and 'High')
#PLOTTING 'BAR CHART' for this:
plt.bar(group_names, df_laptop['price_binned'].value_counts())
plt.xlabel('Price')
plt.ylabel('Count')
plt.title('Price Bins')     


#6.   INDICATOR VARIABLES (DUMMY VARIABLES):
#'Screen' Attribute/Column has 2 Categories - 'Screen-IPS_panel' and 'Screen-Full_HD'
Screen_dummy = pd.get_dummies(df_laptop['Screen'])
Screen_dummy    #Columns for 'Full HD' and 'IPS Panel' are made
df_laptop = pd.concat([df_laptop, Screen_dummy], axis=1)
df_laptop.drop("Screen", axis=1, inplace=True)
#(removed original 'Screen' Column, since No Longer Needed!)

df_laptop.head()  #lets view our Progress - NICE!

df_laptop.to_csv("Cleaned_Laptop_Data.csv")






#%%                      MODULE 3  -  'EXPLORATORY Data Analysis' (also called 'EDA')
# (Descriptive Statistics, GROUPBY, CORRELATION and OTHER Stats)

#'EPA' is an ESSENTIAL 'PRELIMINARY STEP' of DATA ANALYSIS!:
# - SUMMARIZE 'MAIN CHARACTERISTICS' of Data
# - Gain 'BETTER UNDERSTANDING' of Data Set
# -  Uncover 'RELATIONSHIPS' 
# - EXTRACT 'IMPORTANT VARIABLES'

#QUESTION - " WHAT 'CHARACTERISTICS' have 'MOST IMPACT' on 'Car Price'? "


#                    'DESCRIPTIVE STATISTICS'
#'EXPLORE the DATA', BEFORE doing any Complex Model-Building
df = pd.read_csv("Cleaned_Car_Data.csv")
df.describe2()   #SUMMARY STATISTICS (automatically ignored 'NaN' values too - Good!)
#View COUNTS of CATEGORICAL Variables with 'value_counts()':
drive_wheel_counts = df['drive-wheels'].value_counts().to_frame()   #'.to_frame()' just CONVERTS Output into 'DataFrame'!
drive_wheel_counts   # 'fwd = 120', 'rwd = 76', '4wd = 9'
#Just RENAMING 'drive-wheels' column so EASIER to READ:
drive_wheel_counts.rename({'drive-wheels':'value_counts'}, inplace=True)
drive_wheel_counts.index.name = 'drive-wheels'
drive_wheel_counts   #Adding NAME Header TO the 'INDEX COLUMN' of the Counts Table

# 'BOXPLOTS' - also GREAT way to 'VISUALIZE' Numerical DATA
#can view 'Median', 'Q1' and 'Q3' (IQR = Q3 - Q1)
#Upper and Lower 'Extremes' (whiskers) = ' 1.5 * IQR' ABOVE Q3 and BELOW Q2
#OUTLIERS = Dots found on OUTSIDE Box or Whiskers

#ALSO Allow 'EASY COMPARISONS' between 'CATEGORIES':
import seaborn as sns
df['price'].dtypes   # Both OBJECTS so FIRST must CONVERT to 'float64'
df['price'] = df['price'].astype("float64")

sns.boxplot(x='drive-wheels', y='price', data=df)

#CONTINOUS VARIABLES in a RANGE - can find RELATIONSHIPS
#Use SCATTER PLOT - e.g. 'x=Engine Size', 'y-price'
y = df['price']
x = df['engine-size']
plt.scatter(x, y)
plt.title("Scatterplot of Engine Size vs. Price")
plt.xlabel("Engine Size")
plt.ylabel("Price")   #Now we KNOW what are LOOKING AT!
#Can see a POSITIVE, LINEAR RELATIONSHIP between the 2 Variables!
# (i.e. AS 'Engine Size' increases, 'PRICE' will TOO!)


#                           'GROUPBY'
#"is there a RELATIONSHIP between 'drive system' TYPES and 'price' of vehicles?
# - can GROUP (AGGREGATE) by a 'drive system' Variable to SHOW this!
#Use 'df.groupby()' method (applied to CATEGORICAL Variables, SINGLE or MULTIPLE Variables...)
# Example - GROUPING by 'drive-wheels' and 'body-style'
#Then, use '.mean() method' to get AVERAGE 'price' FOR EACH 'drive-wheels' and EACH 'body-style' COMBINATIONS!
df_test = df[['drive-wheels', 'body-style', 'price']]    #choose our specific rows
df_group = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
df_group    #'as_index=False' argument just means we 'DO NOT' WANT Grouped Columns AS 'Index Columns' in the Output DataFrame          
#Now can see WHICH GROUPS have HIGHEST or LOWEST 'price'!
#PROBLEM? This Table is DIFFICULT TO READ and HARD to VISUALIZE!!!

#          'GROUPED Table' INTO 'PIVOT TABLE':
#So? CONVERT into 'PIVOT TABLE' (using '.pivot()' Method):
df_pivot = df_group.pivot(index='drive-wheels', columns = 'body-style')
df_pivot   #'drive_wheels' Variable = ROW 'INDEX' of Pivot Table, 'body-style' is in the COLUMNS
# = MORE CONDENSED, RECTANGULAR GRID - EASIER to VISUALIZE!
#   (JUST like in Excel!)

#   'HEATMAP' Plot can ALSO REPRESENT a PIVOT TABLE:
#Plot 'TARGET Variable' vs. MULTIPLE Variables
#get 'COLOUR INTENSITY' which indicates 'Data Value'
#Use 'pcolor(data, cmap = "color scheme")' function:
plt.pcolor(df_pivot, cmap = 'RdBu')
plt.colorbar()
plt.show()   #given LEGEND to see WHAT each 'COLOR + INTENSITY' represents!


#GROUPING BY for 'MODE' (Most Common/Frequent):
source = pd.DataFrame({
    'Country': ['USA', 'USA', 'Russia', 'USA'], 
    'City': ['New-York', 'New-York', 'Sankt-Petersburg', 'New-York'],
    'Short Name': ['NY', 'New', 'Spb', 'NY']})
#GROUPING by 'MOST COMMON' value of 'Short name':
source.groupby(['Country','City'])['Short Name'].agg(pd.Series.mode).to_frame()
#used '['Short Name'].agg(pd.Series.mode).to_frame()'
#AGGREGATES by 'MODE' for 'Short Name', then CONVERTS BACK to 'DATAFRAME' Element (from 'Series')    
#So MOST COMMON 'Short Name' for EACH 'Country and City' Combination is GIVEN! 

#Can 'GROUP BY' many 'STATISTICS' (just like in SQL!)
#by '.max()', '.min()', '.median()', '.sum()'

#What about Grouping By 'ROW COUNT'? (like .value_counts())?
#Can use '.size()' OR '.count()'


#                          CORRELATION
# = Measures EXTENT which 2 Variables are 'INTERDEPENDENT'
#Important Reminder! -  'CORRELATION' does NOT IMPLY 'Causation'!!
# (exploring 'Correlation' is MORE COMMON THAN 'Causation' in Data Science though)

# Examples - CORRELATION between 'Engine-Size' and 'Price'
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)  #used 'REGRESSION PLOT' to show this!
#same as Scatter Plot, but with 'REGRESSION LINE' through Data Points
#Have 'POSITIVE-LINEAR' CORRELATION here, Very Steep Correlation between Engine Size and Price!   

# Correlation between 'highway-mpg' and 'price':
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)   #Here, have 'NEGATIVE-LINEAR' CORRELATION
#as 'highway-mpg' increases, 'price' DECREASES

# Correlation between 'peak-rpm' and 'price':
sns.regplot(x = "peak-rpm", y="price", data=df)
plt.ylim(0,)   # = WEAK CORRELATION! So CANT use RPM for Predictions!


#            CORRELATION - 'STATISTICAL METHODS':
#Want to 'MEASURE STRENGTH' of Correlation between 2 Variables


#        'PEARSON CORRELATION'  
# Gives 'Correlation COEFFICIENT':
#  - Value CLOSE to '+1' = LARGE POSITIVE Relationship
#  - Value CLOSE to '-1' = LARGE NEGATIVE Relationship
# ALSO Gives 'P-Value':
#  - P-Value '< 0.001'  =  'STRONG CERTAINTY' in Result
#  - P-Value '< 0.05'   =  'MODERATE CERTAINTY' in Result
#  - P-Value '< 0.1'    = 'WEAK CERTAINTY' in Result
#  - P-Value '>0.1'     = 'NO CERTAINTY' in Result

#OVERALL? - STRONG Correlation= 'Coefficeint of +1 or -1' and 'P-Value LESS than 0.001'
#Note: 'CORRELATION' is NOT RELATED to SLOPE of the LINE!!! - 'Correlation' = STRENGTH of Relationship, 'Slope' does not indicate 'Correlation'

#EXAMPLE - Correlation between 'horsepower' and 'price'
from scipy import stats  #REALLY SIMPLE with 'Scipy Stats' Package!!!
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
pearson_coef    #0.8096  -  CLOSE to '+1', so POSITIVE Correlation
p_value    # 6.2735e-48   - VERY SMALL, so STRONG CONDIDENCE/CERTAINTY in this RESULT!!!
#Could then VISUALIZE this Relationship with HEATMAP - see how Variables RELATE to each other, how they relate to PRICE...




#             PRACTICE 1  - 'What are MAIN Characteristics which IMPACT 'Car PRICE' the MOST?
#Using 'Cleaned Car Dataset' (from MODULE 2):
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
    
filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
car_df = pd.read_csv(filepath)
car_df.head()
car_df['peak-rpm'].dtypes  #see that 'peak-rpm' is a 'FLOAT' 

# 'df.corr()' - Calculates 'CORRELATION' between 2 Variables for 'ENTIRE DataFrame (ALL Variables with Each Other'!
correlations = car_df.corr()   #pairs EACH Variable with EACH OTHER, to SEE CORRELATIONS BETWEEN THEM!
#These Correlations are same as 'PEARSON CORRELATIONS' (above!)
#Can Select just a SPECIFIC COLUMN and SORT it 'DESCENDING' (from HIGHEST CORRELATION Value to Lowest)
correlations['price'].sort_values(ascending=False)


car_df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()
#here calculated CORRELATION for JUST a Few Columns. Given as Table.

#Can VISUALIZE any 'CONTINOUS NUMERICAL' Variables with 'SCATTER PLOTS' (with Regression Line too!)
#'POSITIVE-LINEAR' Relationship - 'engine-size' and 'price'
#already VISUALIZED ABOVE - here is CORRELATION:
car_df[['engine-size', 'price']].corr()  #Gives '0.8723' Correlation Coefficient!
#'NEGATIVE-Linear' RELATIONSHIP' - 'highway-mpg' and 'price'
car_df[['highway-mpg', 'price']].corr()   #approximately '-0.704' Correlaiton Coefficient
#'WEAK-Linear' RELATIONSHIP - 'peak-rpm' and 'price'
car_df[['peak-rpm', 'price']].corr()      # very WEAK correlation '-0.1016'
car_df[['stroke', 'price']].corr()       # '0.08231' - Very WEAK Correltaion between 'stroke' and 'price'
sns.regplot(x='stroke', y='price', data=car_df)
plt.ylim(0,)       #is WEAK RELATIONSIHP, not very Linear!

#   For 'CATEGORICAL VARIABLE'? - given as 'object' or 'int64' Data Types
#Can use 'BOXPLOTS' to visualize RELATIONSHIPS with 'CATEGORIES':
sns.boxplot(x="body-style", y="price", data=car_df)
# (plots a BOXPLOT for EACH 'body-style' CATEGORY 'x-axis' Variable)
# - DISTRIBUTIONS of PRICE have 'OVERLAPS' between 'CATEGORIES' ('Differences' are 'NOT DISTINCT ENOUGH'!) 
# - so 'body-style' is NOT a GOOD PREDICTOR of 'price'!

#Lets try 'engine-location' and 'price':
sns.boxplot(x="engine-location", y="price", data=car_df)
# - Now can CLEARLY SEE that have NO OVERLAP between 'front' and 'rear' Category Boxes 
# - 'engine-location' is GOOD PREDICTOR of 'price' - 'rear' Engine Location = MORE EXPENSIVE OVERALL!

#'drive-wheels' and 'price':
sns.boxplot(x="drive-wheels", y="price", data=car_df)
# - have Somewhat CLEAR Diffferences between 'drive-wheel' PRICE DISTRIBUTIONS!
# - so COULD say that 'drive-wheels' is reasonable predictor of 'price' 

#   Lets do some BASIC 'DESCRIPTIVE STATISTICS':
car_df.describe(include = "all")    #including 'object/string' data types too!
#VALUE COUNTS:
drive_wheels_counts = car_df['drive-wheels'].value_counts().to_frame()   #Gives COUNT for EACH 'drive-wheels'
#(CONVERTED from 'SERIES' into 'DATAFRAME' using '.to_frame()':
drive_wheels_counts.rename(columns = {'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts   #named the 'Index Column'!

#REPEAT this for 'engine-location' (SIMPLE!):
engine_loc_counts = car_df['engine-location'].value_counts().to_frame()   #Gives COUNT for EACH 'drive-wheels'
engine_loc_counts.rename(columns = {'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts     
# - '198' engines at FRONT, ONLY '3' at 'rear'  
# = SKEWED Results! 'engine-location' would be POOR PREDICTOR for 'price'
    

#    Practicing 'GROUPING' ('groupby'):
list(car_df['drive-wheels'].unique())   #returns 'DISTINCT Values' FROM a 'DataFrame Variable' (and just CONVERTED to 'LIST Object' from an Array).
#     Which 'drive-wheel' is 'most valuable' ('price')? 
#select JUST the Columns we WANT:
df_group1 = car_df[['drive-wheels', 'body-style', 'price']]
df_grouped = df_group1.groupby(['drive-wheels'], as_index=False).mean()
df_grouped  #Just found AVERAGE 'price' for EACH 'drive-wheels' Category!
#See that 'rwd' (rear-wheel drive) cars are MOST Expensive ON AVERAGE!
#NOW GROUPING by 'body-style':
df_grouped2 = df_group1.groupby(['body-style'], as_index=False).mean()
df_grouped2     #see that AVERAGE PRICE of 'hardtop' cars is MOST

#GROUPING by MULTIPLE VARIABLES -  'drive-wheels' and 'body-style'
#(COVERED ABOVE in Notes)
#ALSO covered how we 'PIVOTTED' the Table (created PIVOT TABLE, like in Excel!)
df_pivot   #could FILL IN the 'na' values with '0'?
df_pivot.replace(np.nan, 0, inplace=True)
df_pivot

#VISUALIZING Relationship of 'Body-Style' vs. 'Price'
#Did ABOVE in NOTES - using SIMPLE 'HEAT MAP'

#HERE will create 'MORE PRESENTABLE/CLEAR' HEAT MAP with BETTER LABELS:
fig, ax = plt.subplots()
im = ax.pcolor(df_pivot, cmap = 'RdBu')
#Now adding ROW and COLUMN Labels and FORMATTING:
row_labels = df_pivot.columns.levels[1]
col_labels = df_pivot.index     #GET the LABELS from Pivot Table
ax.set_xticks(np.arange(df_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(df_pivot.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(row_labels)    #ADD the LABELS for x and y axis
plt.xticks(rotation=45)   
fig.colorbar(im)
plt.show()          #ADDS the LABELS to the Axes and CENTRES the TICKS so it LOOKS NICER!

#      More Practice with 'PEASRON CORRELATION'
#LOOPING THROUGH to find it for EACH 'Variable' (Looping is REAL TIME-SAVER!)
for variable in ['wheel-base', 'horsepower', 'length', 'width', 'curb-weight', 'engine-size', 'bore']:
    pearson_coeff, p_value = stats.pearsonr(car_df[variable], car_df['price'])
    print(f"Pearson Correlation Coefficient is {pearson_coeff}, with a P-Value of '{p_value}' \n")   

#   So WHICH VARIABLES have SIGNIFICANT EFFECT on 'price'?
#Continuous NUMERICAL Variables - Length, Width, Curb-Weight, Engine-Size, Horsepower, City-mpg, Highway-mpg, Wheel-base, Bore
#CATEGORICAL Variables - Drive-wheels (found by Boxplot!)



#                    PRACTICE 2 - 'Laptops Pricing' Dataset
#use Cleaned DataFrame for 'Laptops'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

laptop_filepath="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
laptop_df = pd.read_csv(laptop_filepath)
laptop_df.columns.values

sns.regplot(x='CPU_frequency', y='Price', data=laptop_df)
plt.ylim(0,)       

sns.regplot(x='Screen_Size_inch', y='Price', data=laptop_df)
plt.ylim(0,)       

sns.regplot(x='Weight_pounds', y='Price', data=laptop_df)
plt.ylim(0,)       
#Finding '.corr()' CORRELATIONS for EACH Variable with 'Price':
for variable in ['CPU_frequency', 'Screen_Size_inch', "Weight_pounds"]:
    print(f"Correlation of Price and {variable} is: {laptop_df[[variable, 'Price']].corr()}")
#ALL have pretty WEAK CORRELATIONS with 'Price'!

#Categorical Variables - BOX PLOTS:
    
sns.boxplot(x="Category", y="Price", data=laptop_df)

sns.boxplot(x="GPU", y="Price", data=laptop_df)

sns.boxplot(x="OS", y="Price", data=laptop_df)

sns.boxplot(x="CPU_core", y="Price", data=laptop_df)

sns.boxplot(x="RAM_GB", y="Price", data=laptop_df)

sns.boxplot(x="Storage_GB_SSD", y="Price", data=laptop_df)

#Descriptive Statistics - with '.describe()'
laptop_df.describe(include = "all")

#GROUPBY and PIVOT TABLE for 'GPU', 'CPU_core' and 'Price':
variables_group = laptop_df[['GPU', 'CPU_core', 'Price']]
grouped = variables_group.groupby(['GPU', 'CPU_core'], as_index = False).mean()
print(grouped)  #For EACH 'GPU' with EACH 'CPU_core', get the 'Average PRICE'
#Grouping as PIVOT TABLE:
grouped_pivot = grouped.pivot(index = 'GPU', columns = 'CPU_core')
grouped_pivot   #now much more CONDENSED and CLEAR VIEW!
#DISPLAY Pivot Table as HEAT MAP:
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap = 'RdBu')
#Now adding ROW and COLUMN Labels and FORMATTING:
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index     #GET the LABELS from Pivot Table
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(row_labels)    #ADD the LABELS for x and y axis
fig.colorbar(im)
plt.show()          #ADDS the LABELS to the Axes and CENTRES the TICKS so it LOOKS NICER!

#   PEARSON CORRELATION and P-Values:
#Evaluate for EACH VARIABLE we tested ABOVE:
for variable in ['CPU_frequency', 'Screen_Size_inch', "Weight_pounds", 'RAM_GB', 'Storage_GB_SSD', 'Weight_pounds', 'OS', 'GPU', 'Category']:
    pearson_coeff, p_value = stats.pearsonr(laptop_df[variable], laptop_df['Price'])
    print(f"For {variable} vs. Price  -  Pearson Correlation Coefficient is {pearson_coeff}, with a P-Value of '{p_value}' \n")     #added '\n' so leaves SPACE between EACH - Looks NICER!
#LOOPING makes this SOOOO MUCH EASIER!
    
    






