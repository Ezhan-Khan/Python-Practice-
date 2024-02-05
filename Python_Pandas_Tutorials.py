#%%
#                                  FULL Pandas Tutorial/Practice


#                         GET DATA INTO PANDAS DATAFRAME:

import pandas as pd
     
csv_df = pd.read_csv(r"C:\Users\Ezhan Khan\Documents\PYTHON PRACTICE\PYTHON for DATA ANALYTICS\Pandas_File_Formats\countries_of_the_world.csv")
csv_df
#can specify many DIFFERENT ARGUMENTS - most are not used very often though!
#e.g. could specify 'header = None' (have dataframe with NO headers)
#OR can SPECIFY 'names = [new_header1, new_header2,...]'

#Importing Text (.txt) Files:
#1. CAN be done with 'read_csv', SPECIFYING SEPARATOR as 'sep = "\t"  argument')
txt_df = pd.read_csv(r"C:\Users\Ezhan Khan\Documents\PYTHON PRACTICE\PYTHON for DATA ANALYTICS\Pandas_File_Formats\countries_of_the_world.txt", sep = '\t')
txt_df.head()
#OR, just use '.read_table()' on the .txt file:
txt_df2  = pd.read_table(r"C:\Users\Ezhan Khan\Documents\PYTHON PRACTICE\PYTHON for DATA ANALYTICS\Pandas_File_Formats\countries_of_the_world.txt")
txt_df2
#SIMPLE!

#Importing JSON Files using 'read_json'
json_df =  pd.read_json(r"C:\Users\Ezhan Khan\Documents\PYTHON PRACTICE\PYTHON for DATA ANALYTICS\Pandas_File_Formats\json_sample.json")
json_df

#Importing Excel Files
excel_df =  pd.read_excel("world_population_excel_workbook.xlsx")
#NOTE - might not work, since xlsx files NOT SUPPORTED


#What if we want Pandas OUTPUT to SHOW MORE ROWS/COLUMNS?
#Use 'pd.set_option('display.max.rows', 235)'
pd.set_option('display.max.rows', 100)
#Can also use 'display.max.columns' to specify COLUMNS SHOWN
pd.set_option('display.max.columns', 20)
#Handy to let us get a BETTER LOOK at the data!

world_df = pd.read_csv("world_population_excel_workbook.csv")
#get some INFO on the DataFrame (memory usage, columns, datatypes, non-null values)
world_df.info()
#Dataframe Shape (number of Rows and Columns)
world_df.shape

#'LOCATE' SPECIFIC ROWS using either '.iloc' OR '.loc'
#'loc' selects rows BY Row LOCATION and NAMES - 'df.loc[Row(s), [Columns_Names]]'
#'iloc' selects rows BY ROW INDEX - 'df.iloc[row_index]'


#%%                   FILTERING and ORDERING DataFrame

world_df

#FILTERING to ONLY view 'Rank' values BELOW 10:
world_df[world_df['Rank']<=10]  

#                  '.isin(["value1", "value2"])' 
#Filter rows to find for 'SPECIFIC COLUMN CATEGORY VALUES':
#Use '.isin(["value1", "value2"])' SEVERAL POTENTIAL VALUES here
world_df[world_df['Country'].isin(['Bangladesh', 'Brazil'])]    
#Recall - same as 'IN' in SQL - e.g.  WHERE Country IN ("country1", "country2")

#                   '.str.contains('string')':
#SIMILARLY can Filter for Columns which CONTAIN SPECIFIC STRINGS:
world_df[world_df['Country'].str.contains('United')]
#Recall - same as 'LIKE' in SQL!

#BETTER WAY to FILTER? - use '.filter(items = [col1, col2...], axis= 0 or 1)'
#axis =1 for COLUMNs, =0 for ROWS
world_df.filter(items = ['Country', 'Continent'], axis=1)    #obtains JUST 'Country' and 'Continent' COLUMNS
#Can ALSO use with ' like="string" ' (FILTER ROW/COLUMN where it is LIKE Sometihng)
#First, can Set 'Country' Column as INDEX:
df = world_df.set_index('Country') 
#Now can FILTER WHERE 'Index' Column is 'LIKE' that:   
df.filter(like = 'United', axis=0)   #(ALTERNATIVE to '.contains' above!)


#To FILTER BY the ROW, use '.iloc' or 'loc' (as mentioned above)
#  USEFUL WAY to FIND 'All Relevant DATA' ON a 'PARTICULAR Row':
#e.g. LOCATION where 'INDEX' is 'United States'
loc_df = df.loc['United States']  #find data for row_index called 'United States'
#gives COLUMNS with ALL VALUES 
#'iloc' requires 'INTEGER-LOCATION' (ROW-NUMBERING)
iloc_df = df.iloc[3]  #find data for row_index of '3'
iloc_df   #All Columns and Values for THAT LOCATION 
# (NOTE - think of this like 'XLOOKUP' in EXCEL!)


#ORDER BY - using '.sort_values(by=  , ascending=)' 
df_below_10 = world_df[world_df['Rank'] < 10]
df_below_10 = df_below_10.sort_values(by = "Rank", ascending=False)
#Can also SORT by 'MULTIPLE VALUES':
df_below_10.sort_values(by = ["Continent","Country"], ascending=False)
#can even SPECIFY whether Ascending or Descending for EACH VALUE SORTED BY 
#e.g. ascending = [False, True]  (descending for 'Continent', ascending for 'Country')
#REALLY SIMPLE!


#%%                   (ROW) 'INDEXING' in Pandas

df = world_df
df.head()

#Index is just a NUMBER/LABEL for a ROW
#Can SET 'Country' as the Index:
df.set_index('Country', inplace=True)
#(include 'inplace=True' so current df is UPDATED)

#If we want to REVERSE THIS, can just RESET INDEX:
df.reset_index(inplace=True)

#Now, can use 'loc' and 'iloc' ON this df:
df.loc['Albania']  #retrieve all data for 'Albania' Row_Index
#OR, can use 'iloc', using the INDEX integer position:
df.iloc[1]


#   'MULTI-INDEXING' = using not just one, but several indexes!
df.reset_index(inplace=True)
#Now, SET BOTH 'Continent' AND 'Country' as Indexes
df.set_index(['Continent', 'Country'], inplace=True)

#Can SORT the INDEXES so looks neater, using '.sort_index()'
df.sort_index(inplace=True)
#Now is NICELY SORTED by 'Continents' first, THEN by 'Countries' ALPHABETICALLY!
#Or, can sort DESCENDING:
df.sort_index(ascending=False, inplace=True)

#NOW, for 'loc' need to SPECIFY BOTH 'Continent' AND 'Country':
df.loc['Africa', 'Angola']
#But for, 'iloc', just the 'row_index' NUMBER is needed (AS USUAL)
df.iloc[1]


#%%                'GROUP BY (AGGREGATION)' in PANDAS:
    
flavours_df = pd.read_csv(r"C:\Users\Ezhan Khan\Documents\PYTHON PRACTICE\PYTHON for DATA ANALYTICS\Pandas_File_Formats\Flavors.csv")
flavours_df
#table of different Ice Cream Flavours, with RATINGS on whether liked, flavour, texture and TOTAL Rating.

#Can GROUP BY 'Base Flavour' (find Aggregate stats for EACH Base Flavours)
grouped_flavours = flavours_df.groupby(['Base Flavor'], axis=0, as_index = False)
#creates a 'groupby' OBJECT, which can be saved as variable
#then, can find '.mean()' ('AGGREGATED AVERAGE' BY the GROUPED 'Base Flavor' Column)
average_base_flavours = grouped_flavours.mean()
#can COMPARE the AVERAGE RATINGS for 'chocolate' vs 'vanilla' 

#OTHER AGGREGATE FUNCTIONS - .count(), .min(), .max(), .sum()
grouped_flavours.count()  #shows, results MIGHT be SKEWED in favour of 'chocolate'
grouped_flavours.min()  #take lowest values for 'chocolate' and 'vanilla' respectively (alphabetically for string data type columns, like 'Flavour' or 'Liked')
grouped_flavours.max()  #takes highest values
grouped_flavours.sum()  #(only totals for NUMERICAL Columns, since cannot do this for Strings!)


#Alternatively, have 'AGGREGATION FUNCTION':   (BETTER if wanting to use SEVERAL Aggregate Functions SIMULTANEOUSLY!)
#'.agg({'column':['mean', 'max', 'count', 'sum'],...})'
#-takes 'dictionary' argument
#-'key-value' pairs, where 'key' = Column which is to be aggregated, Value = LIST of AGGREGATE FUNCTION(s) to APPLY to it!
#REPEAT 'key-value' pair for EACH COLUMN which we want to INCLUDE IN the Aggregation Output!
flavours_df.groupby('Base Flavor').agg({'Flavor Rating':['mean', 'max', 'count', 'sum']})
#Could pass in MULTIPLE Aggregated COLUMNS too:
flavours_df.groupby('Base Flavor').agg({'Flavor Rating':['mean', 'max', 'count', 'sum'],'Texture Rating':['mean', 'max', 'count', 'sum']})    

#all this, just returns AGGREGATE STATISTICS for EACH Specified Column, BY the GROUPED Column ('Base Flavor')

#'GROUPING' BY 'MULTIPLE COLUMNS' 
multiple_grouped = flavours_df.groupby(['Base Flavor', 'Liked'], axis=0, as_index = False)
multiple_grouped.mean()
#see that for 'Vanilla', the 'No's were really HIGH, whereas 'Yes's were really high 

multiple_aggregated = flavours_df.groupby(['Base Flavor', 'Liked']).agg({'Flavor Rating':['mean', 'max', 'count', 'sum']})
multiple_aggregated


#If we only want a HIGH-LEVEL OVERVIEW of the Data?
#Just use '.describe()'
flavours_df.groupby('Base Flavor').describe()
#EACH COLUMN is GROUPED BY 'Base Flavor' and given an OVERVIEW of ALL Ratings.




#%%         'MERGE, JOIN and CONCATENATE' DataFrames in Pandas


#combining 2 separate dataframes INTO ONE!

#LIKE SQL JOINS (usual):
#have 'INNER' Join, 'OUTER' Join, 'LEFT' Join, 'RIGHT' Joins

df1 = pd.read_csv(r"C:\Users\Ezhan Khan\Documents\PYTHON PRACTICE\PYTHON for DATA ANALYTICS\Pandas_File_Formats\Joins_LOTR.csv")
df2 = pd.read_csv(r"C:\Users\Ezhan Khan\Documents\PYTHON PRACTICE\PYTHON for DATA ANALYTICS\Pandas_File_Formats\Joins_LOTR2.csv")
df1
df2

#                  'MERGE' (most important method!)

#this is just like SQL Joins!  
df1.merge(df2)   #DEFAULT = INNER JOIN (ONLY MATCHING IDs are MERGED!)
df1.merge(df2, how='inner')  #does SAME IF JOIN-METHOD is SPECIFIED in ARGUMENT (how = "inner")
#BETTER to Specify COLUMN(s) to JOIN 'ON':
df1.merge(df2, how='inner', on=['FellowshipID', 'FirstName'])  
#OPTIONAL - can specify SUFFIXES 'suffixes = ('_x','_y')' - e.g. when joining by 'FellowshipID' COLUMN ONLY, which results in FirstName_x and FirstName_y for EACH respective TABLE!

# 'OUTER JOIN'
df1.merge(df2, how='outer')  
#returns ALL ROWS from BOTH TABLES, EVEN IF they DONT MATCH
#ANY 'UNMATCHED' Data is given 'NaN' (NULL) values!

# 'LEFT JOIN'
df1.merge(df2, how='left')  
#pulls ALL from LEFT DataFrame, EVEN IF NOT IN 'Right' DataFrame!
#i.e. EVEN IF FellowshipID is MISSING from Table 2, is STILL INCLUDED, but is JUST given 'NaN' for MISSING Data!

# 'RIGHT JOIN' (just opposite of LEFT Join)
df1.merge(df2, how='right') 

# 'CROSS JOIN' (NOT Used AS Much!)
#just gets MATCHING PAIRS of EVERYTHING from BOTH TABLES
#     df1.merge(df2, how ='cross')


#                   'JOIN' FUNCTION

#SIMILAR to 'MERGE' (does same stuff)
#But, is a LITTLE HARDER to UNDERSTAND!
#Need to SPECIFY 'SUFFIXES' sometimes:
df1.join(df2, on='FellowshipID', how='outer', lsuffix='_left', rsuffix ='_right')
#get SEPARATE COLUMNS for EACH DATAFRAME with SUFFIXES PROVIDED!
#LOOKS DIFFERENT!!!
#Note - .join() method is BEST WHEN JOINING ON 'INDEXES' 
#Just SET 'FellowshipID' AS the INDEX:
df4 = df1.set_index('FellowshipID').join(df2.set_index('FellowshipID'), how='outer', lsuffix='_left', rsuffix ='_right')    
df4
#Now looks MUCH NICER (like MERGE Method Output!)


#                  'CONCATENATE' (like UNION in SQL)
#used to STACK DataFrames ON TOP of EACH OTHER
pd.concat([df1,df2])
#Can even JOIN HERE:
pd.concat([df1,df2], join='inner')
#but, here ONLY takes 'COLUMNS' which are the SAME!
#Can specify 'axis' (=1 for columns, =0 for Rows)
pd.concat([df1,df2], join='inner', axis=1)

#        APPEND (another way to jon dataframes)
df1.append(df2)   #DEPRECATED - STICK to 'pd.conact' in FUTURE!!!


#%%                         PANDAS 'VISUALISATION'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Ezhan Khan\Documents\PYTHON PRACTICE\PYTHON for DATA ANALYTICS\Pandas_File_Formats\Ice_Cream_Ratings_Visualization_Data.csv")
df

#Side Note - Use 'figzise=(length, width)' argument where needed

#FIRST, can select 'DIFFERENT STYLES' OF VISUALS (using matplotlib.pyplot):
#View 'ALL AVAILABLE STYLES':
plt.style.available    
#'plt.style.use' to APPLY STYLE for ALL VISUALS!
plt.style.use('seaborn-darkgrid')


#PLOT in PANDAS by using '.plot()' method:
#specify x, y and 'kind' of PLOT we want ('line', 'bar', 'barh', 'hist', 'box', 'kde')    
df.plot(x='Date', y=['Flavor Rating', 'Texture Rating', 'Overall Rating'], kind='line')
#can also add title, grid, legend, style, change 'labels'...

#'subplots = True' gives SEPARATE PLOTS for EACH 'y' Variable:
df.plot(x='Date', y=['Flavor Rating', 'Texture Rating', 'Overall Rating'], subplots=True, kind='line')    

#Adding Title, xlabel, ylabel..
df.plot(x='Date', y=['Flavor Rating', 'Texture Rating', 'Overall Rating'], kind='line', title='Ice Cream Ratings', xlabel = 'Daily Ratings', ylabel = 'Scores')


#Looking at kind = "bar" chart:
df.plot(x='Date', y=['Flavor Rating', 'Texture Rating', 'Overall Rating'], kind='bar', title='Ice Cream Ratings', xlabel = 'Daily Ratings', ylabel = 'Scores')
#Could make it 'STACKED' by specifying 'stacked = "True"
df.plot(kind='bar', stacked=True)
#JUST for 'Flavor Rating' Column:
df.plot(x='Date', y='Flavor Rating', kind='bar', stacked=True)
#'HORIZONTAL' BAR Chart - 'barh':
df.plot(x='Date', y=['Flavor Rating', 'Texture Rating', 'Overall Rating'], kind='barh', stacked = True, title='Ice Cream Ratings', xlabel = 'Daily Ratings', ylabel = 'Scores')    
#(LOOKS BETTER when there are MANY 'x' LABELS!!)


#Looking at SCATTER PLOT:
df.plot(x='Texture Rating', y='Overall Rating', kind='scatter', s=200, c="darkblue")
#used 's=200' to make SIZE of points LARGER
#used 'c="color"' to SPECIFY 'COLOUR' of points


#Looking at HISTOGRAM:
df.plot(x='Texture Rating', kind='hist', bins = 4, c="darkblue")
#can specify 'bins' for HOW SPREAD OUT DISTRIBUTION IS!


#Looking at BOXPLOT:
df.boxplot()
#can COMPARE 'Medians' and overall distribution of data for EACH Variable


#AREA Plot (filled in line chart):        (kind="area")
df.plot(x='Date', y=['Flavor Rating', 'Texture Rating', 'Overall Rating'], kind='area', figsize=(10,5))
#note - for ANY CHART, use 'figsize=(length, width)' to MAKE LARGER/SMALLER!


#PIE CHART
df.plot(y='Flavor Rating', kind='pie', figsize=(10,5))






#%%
























