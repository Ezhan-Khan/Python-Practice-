/# -*- coding: utf-8 -*-

#  import sys   then doing 'sys.version'  - tells us VERSION of PYTHON (i.e. looks at SYSTEM settings)

#  'list.extend([list to add])'   - 'extend' is LIKE 'append', but ADDS 'LIST VALUES' TO the END of the EXISTING List 
#                                  (so is MORE like '.update()' but for LISTS, not Dictionaries!)         

#  'del(dictionary[key])'  - can use 'del' to DELETE any key:value pair (just specify 'key')

# isinstance(tuple_example, tuple)  - CHECK for TUPLE (better than doing 'type(p) == tuple')


#%%         Working with DIFFERENT 'FILE FORMATS' (csv, xml, json, xlsx)

#                     'Data Engineering' involves 'ETL':
# 'Data Extraction' - get data from MULTIPLE SOURCES     
# 'Data TRANSFORMATION' - REMOVE UNECESSARY Data, ONLY Keeping what we NEED for ANALYSIS, data from multiple sources converted to SAME FORMAT
# 'Data LOADING' - LOAD data INSIDE a 'Data WAREHOUSE' (=LARGE VOLUMES of Data accessed to GATHER INSIGHTS)

# RARELY get 'NEAT' Tabular Data, so MUST be able to deal with DIFFERENT 'File Formats'.

#Python can make PROCESS of READING Data from DIFFERENT File Formats SIMPLE, using LIBRARIES!

#                  'Pandas' (JUST RECAP):
#   pd.read_csv('file.csv')  - SIMPLE, know this!
# add ARGUMENT 'header = None' - IF we want 'NO HEADER ROW'  -  pd.read_csv(url, header = None)

# 'ADD COLUMN HEADERS' to 'DataFrame' - 'df.columns = ['col_header1', 'col_header2'...] - adds New HEADERS

#  df['First Name'] - SINGLE BRACKETS '[]' selects 'First Name' Column (as 'Set')
#Use 'DOUBLE BRACKETS' - '[[]]' when Selecting 'MULTIPLE Columns' - df = df[['First Name', 'Last Name', 'Location', 'City'...]]
#use 'loc' to select SPECIFIC rows (e.g. 'df.loc[0]' - selects FIRST ROW)
#Selecting specific RANGE of ROWS for a SPECIFIC Column - 'df.loc[0:2, "First Name"]'

#'TRANSFORM' FUNCTION (in Pandas):
import pandas as pd
import numpy as np

df=pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])   # = 'ARRAY' to 'DATAFRAME'
df    
#use 'df.transform' to CHANGE FOMRAT of the Data:
df = df.transform(func = lambda x : x + 10)
df       #using 'LAMBDA' (quick-function) ADDS 10 to EACH VALUE in the DataFrame! 
df= df.transform(func = ['sqrt'])
df #doing 'SQUARE ROOT' now!




# 'JSON File' (Similar to Python Dictionary):
import json

person = {
    'first_name' : 'Mark',
    'last_name' : 'abc',
    'age' : 27,
    'address': {
        "streetAddress": "21 2nd Street",
        "city": "New York",
        "state": "NY",
        "postalCode": "10021-3100"
    }
}
#WRITE a JSON File with 'json.dump()' method:      (note: 'dumps()' can do this too, SAME THING!)
with open('person.json', 'w') as f:
    json.dump(person, f)   #creates NEW JSON file FROM 'person' Python Dictionary!

#To convert this 'BACK to PYTHON Object' ('deserialization'), can use 'load()' function (shown ABOVE)
with open('person.json', 'r') as openfile:
    json_object = json.load(openfile)   #'json.load(openfile)' lets you READ the FILE!
    print(json_object)    #SEE 'PYTHON MAIN NOTES' File - ALL Covered THERE
type(json_object)  #converted into 'dict' python object!




#         'XML' (Extensible Markup Language)

import xml.etree.ElementTree as etree  #use 'xml' library module 'xml.etree.Element.Tree' to PARSE and CREATE 'XML' Documents. 
#'ElementTree' = XML Document AS a TREE. Move across 'Nodes' (ELEMENTS and SUB-ELEMENTS) of XML File. 
#Can CREATE 'FILE STRUCTURE' with 'etree.Element('employee'), 'etree.SubElement(employee, 'details')... then 'first.text' 'Shiv', 'second.text = 'Mishra' ...
#THEN can CREATE a NEW XML File with THIS STRUCTURE:
mydata1 = etree.ElementTree(employee)
with open("new_sample.xml", "wb") as files:
    mydata1.write(files)    #WRITES into 'NEW XML FILE'

#  READING with 'xml.etree.ElementTree' ('XML' INTO a 'DATAFRAME')
tree = etree.parse("fileExample.xml")  #First, can PARSE the XML File (using 'etree' attribute)
root = tree.getroot()
columns = ["Name", "Phone Numbers", "Birthday"]   #LIST of COLUMNS for the DataFrame
df = pd.DataFrame(columns = columns)     #ASSIGN CREATED Column Headers TO the DataFrame

for node in root:         
    name = node.find("name").text
    phonenumber = node.find("phonenumber").text
    birthday = node.find("birthday").text
#Use LOOP to COLLECT the NECESSARY DATA and APPEND the Data TO the DataFrame:
df = df.append(pd.Series([name, phonenumber, birthday], index=columns)....,ignore_index = True)    

#'READ the XML File' - with 'pd.read_xml(file.xml', xpath = "/employee/details")


#         'BINARY' FILE FOMRAT  = 'non-readable characters' 
#usually 'IMAGES' like 'JPEGs', 'GIFs', 'MP3s', Binary Documents like WORD or PDF 
img = Image.open('dog.jpg')
display(img)




#%%            APIs with 'PYTHON'

#              PANDAS is an API!!!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   

dict_= {'a':[11,21,31], 'b':[12,22,32]}
#Create 'PANDAS OBJECT/INSTANCE':
df = pd.DataFrame(dict_)   #creates 'DataFrame' OBJECT (from 'Pandas' - acts like 'Class' INSTANCE!)
type(df)      
df.head()    #calling first few rows - COMMUNICATES with 'API' to DO this!
df.mean()    #communicates WITH API to find 'mean' of DataFrame Columns!


#           EXAMPLE API Data - Using 'NBA API': 

    #How well have 'Golden State Warriors' performed vs. 'Toronto Raptors'?
#For EACH GAME, determine NUMBER of POINTS 'GSW' WON/LOST againts 'TR' 
# e.g. if won by '3', means WON by 3 points. Lost by 2 points = -2 value

#Import 'teams' module FROM 'nba_api':
!pip install nba_api
from nba_api.stats.static import teams
import matplotlib.pyplot as plt
#Use '.get_teams()' method to GET ALL the 'teams' - SIMPLE!
nba_teams = teams.get_teams()  
# ='LIST of DICTIONARIES' for 'EACH TEAM' (EACH as a List Element!)
#    (this is a Simple 'GET REQUEST', using '.get_teams()')
print(nba_teams[0:3])     #viewing first 3 elements.

#Need to CONVERT to ONE 'DICTIONARY' FORMAT (so can EASILY be CONVERTED to 'pd.DataFrame' !)
# FUNCTION - CONVERTS 'LIST' of 'Dicitonaries' into SINGLE 'DICTIONARY'
def one_dict(list_dict):
   keys = list_dict[0].keys()  #Access JUST the 'KEYS' of FIRST Dictionary of List (into a 'dict object/list' of 'KEYS')
   out_dict = {key:[] for key in keys}  #used 'DICT COMPREHENSION' to CREATE DICTIONARY from 'keys' (created above) and with 'values' as EMPTY LIST
   for dict_ in list_dict:
       for key, value in dict_.items():
           out_dict[key].append(value)    
   return out_dict                      
# (then ADDED the corresponding 'value' for EACH 'key' of list_dict)
# Using '.append(value)' to 'out_dict'
#  ('DEMONSTRATING' HOW the 'FUNCTION' WORKS):
keys = nba_teams[0].keys()
out_dict = {key:[] for key in keys}
out_dict  #is DICTIONARY of 'key':[] 
#Now must ADD TO the EMPTY LISTS with LOOP:
for dict_ in nba_teams:   #for EACH 'DICITONARY' (= ELEMENT in 'List of Dictionaries')
    for key, value in dict_.items():   
        out_dict[key].append(value)  #Now using .items() to ACCESS EACH Key and EACH Value!
#ADDS the related info for EACH 'value' (in nba_teams) to EACH LIST ('value') of 'out_dict'
out_dict    #Now is NEATER and can EASILY CONVERT into DATAFRAME!

#Using 'FUNCTION' (same thing):
nba_team_dictionary = one_dict(nba_teams)  #AWESOME!
nba_team_dictionary
#Now can CONVERT into DATAFRAME!!!
df_teams = pd.DataFrame(nba_team_dictionary)
list(df_teams.columns.values)  #gives LIST of COLUMN NAMES

# FILTERING COLUMNS 'JUST' for 'Warriors' Nickname:
df_warriors = df_teams[df_teams['nickname'] =='Warriors']
df_warriors  #Gives JUST the INFO for 'WARRIORS'

#     Make ANOTHER API CALL using 'LeagueGameFinder' function
# This is from 'stats.endpoints' module of 'nba_api':
from nba_api.stats.endpoints import leaguegamefinder
# Will use PARAMETER 'team_id_nullable' = 'UNIQUE ID' for 'WARRIORS' ('id' found from DATAFRAME ABOVE!):
gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=1610612744 )
gamefinder.get_json()   #use '.get_json()' to GET a JSON FILE for this!
#Can use 'get_data_frames()' to GET 'AS DATAFRAME':
games = gamefinder.get_data_frames()[0]
games.head()   
#'PLUS_MINUS' Column = INFOMRATION about the SCORE (if -ve, Warriors LOST by that many points, If +ve, warriors WON by that many points)
#'MATCHUP' Column = TEAM which Warriors PLAYED ('GSW = Goldern State Warriors', 'TOR = Toronto Raptors') 
#(Note - 'vs' = HOME Game, '@' = AWAY Game)

#   SEPARATE this DataFrame into 2 DATAFRAMES (EASY!)
# 1. for HOME Games (for Warriors 'vs'. Raptors)  2. AWAY Games (for Warriors '@' Raptors)
games_home = games[games['MATCHUP']=='GSW vs. TOR']
games_away = games[games['MATCHUP']=='GSW @ TOR']
#SIMPLE FILTERING! Now can 'Calculate MEAN' for 'PLUS_MINUS' Column for EACH DataFrame:
np.mean(games_home['PLUS_MINUS'])    #'3.96774' - PLAY BETTER when 'Home' on AVERAGE!
np.mean(games_away['PLUS_MINUS'])    #-2.24375'

#FINALLY can PLOT for 'PLUS_MINUS' Column for EACH DATE:
fig, ax = plt.subplots()  #PLOTS for 'home' and 'away' TOGETHER on SAME GRAPH!
games_away.plot(x='GAME_DATE', y='PLUS_MINUS', ax=ax)
games_home.plot(x='GAME_DATE', y='PLUS_MINUS', ax=ax)
ax.legend(['away', 'home'])
plt.show()      #COOL! Can clearly see, They perform BETTER when at 'HOME'!



#          EXAMPLE 2 - 'PyCoinGecko Library' for 'CoinGecko API'
!pip install pycoingecko    #='CRYPTOCURRENCY'
from pycoingecko import CoinGeckoAPI

cg = CoinGeckoAPI()   #Create 'OBJECT/INSTANCE' for 'CoinGeckoAPI' CLASS
#Used a 'Class FUNCTION (METHOD)' to REQUEST Data
#'Arguments' (Parameters) to SPECIFY DATA we WANT:
# = 'bitcoin' data, in 'USD', from PAST '30' Days   
bitcoin_data = cg.get_coin_market_chart_by_id(id = 'bitcoin', vs_currency='usd', days=30)
print(bitcoin_data)
#get JSON FILE = 'DICTIONARY' of 'NESTED LISTS'
#EACH Nested List contains 2 Elements - 'TIMESTAMP' and 'RELATED INFO (prices, market_caps, total_volumes'
#Includes 'Keys' - 'prices', 'market_caps', 'total volumes'
list(bitcoin_data.keys())   #(viewing keys) 

#     HOW can we USE this Raw Data?
       #CONVERT into DATAFRAME:
df_bitcoin= pd.DataFrame(bitcoin_data, columns = ['prices'])
df_bitcoin   #'EACH ROW' = 'LIST' of 2 Elements - 'TImestamp' and 'price'            

#  SPLIT 'prices' COLUMN into 2 NEW COLUMNS - 1 for 'EACH ELEMENT' of List:
#(i.e. Splitting LIST 'BY ELEMENTS'):
df_bitcoin[['Timestamp', 'Prices']] = pd.DataFrame(df_bitcoin['prices'].tolist(), index=df_bitcoin.index)
df_bitcoin  #Looks GREAT Now - Normal Columns!

#   Now can Create 'Date' Column FROM 'TimeStamp' (CONVERTS 'TimeStamp' INTO 'Date')
#Use Pandas ' pd.to_datetime(column, unit='') '
df_bitcoin['Date'] = pd.to_datetime(df_bitcoin['Timestamp'], unit='ms')
df_bitcoin




#%%             'REST APIs' and 'HTTP REQUESTS':

# 'HTTP Method' (MORE COMMON than ABOVE!) = GENERAL way to TRANSFER Information THROUGH the WEB (usually in 'URL'!)
#Done through a 'URL' ('Uniform-Resource-Locator')
#This includes Many TYPES of REST APIs - send request, via HTTP message with JSON File
#Client (You) SEND object TO CLIENT in HTTP response (type and length of resource...)
#              (COPY REST FROM NOTES...)

#   Python EXAMPLE - 'GET REQUEST':
import requests  #'requests' library

url ='https://www.ibm.com/'     
r = requests.get(url)   #'GET' the 'RESPONSE' OBJECT, saved as variable 'r'
#Accessing ATTRIBUTES of the 'Response':
r.status_code  # = '200'  =  indicates 'SUCCESS'!
print("request body: ", r.request.body)  # NONE (nothing in 'request' body for 'get' request)
#Viewing HEADERS:
r.headers  #given as 'Dictionary'
print(r.headers['date'])  #DATE Request was SENT 
print(r.headers['Content-Type'])  #given as 'text/HTML' response type
#Also checking 'encoding' (optional):
r.encoding  # 'utf-8'
r.text[0:100]   #'text' used to DISPLAY 'HTML' in BODY, viewing First 100 Characters
 
# ALSO can load 'NON-TEXT' Requests (e.g. for 'IMAGES'):
import os
from PIL import Image
from IPython.display import IFrame

url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/IDSNlogo.png'
r = requests.get(url)
print(r.headers)   #STILL get HEAEDRS
print(r.headers['Content-Type'])  # 'image/png' type

#         SAVE and VIEW the IMAGE (using 'file object'):
#SPECIFY the 'File Path' and 'Name':
path=os.path.join(os.getcwd(),'image.png')  
path          #Just CALLED it 'image.png'
with open(path, 'wb') as f:
    f.write(r.content)   #WRITE INTO 'image.png' FIle
#VIEW the IMAGE (IN Python IDE CONSOLE) by OPENING 'path':
Image.open(path)   #COOL! Displays Image IN CONSOLE!

url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%205/data/Example1.txt'
path=os.path.join(os.getcwd(),'example1.txt')
r=requests.get(url)
with open(path,'wb') as f:
    f.write(r.content)


#  'PYTHON EXAMPLE 2' - 'INCLUDING URL PARAMETERS' in the 'Get' Request:
url_get='http://httpbin.org/get'
payload={"name":"Joseph","ID":"123"}  #PROVIDING Parameters as 'DICTIONARY'
r=requests.get(url_get,params=payload)
r.url    # 'http://httpbin.org/get?name=Joseph&ID=123'
print("request body:", r.request.body)
print(r.status_code)
print(r.text)
r.headers['Content-Type']  # 'application/json'
r.json()
print(r.json()['args'])


#  POST REQUEST - Example
# (similar to 'get', but POST sends data IN the 'REQUEST BODY')
#Just CHANGE the 'URL Route' to '/post':
url_post='http://httpbin.org/post'
r_post=requests.post(url_post,data=payload)
print("POST request URL:",r_post.url )
print("GET request URL:",r.url)
#  http://httpbin.org/post  and  http://httpbin.org/get?name=Joseph&ID=123
#so 'post' doesn't have 'parameters' IN the URL!
#INSTEAD, 'POST' Request has 'Parameters' IN the REQUEST 'BODY':
print("POST request body:",r_post.request.body)    # 'name=Joseph&ID=123'
print("GET request body:",r.request.body)        # 'None'
#View 'form' to get URL Parameters for POST:
r_post.json()['form']    # {'ID': '123', 'name': 'Joseph'} 


#%% MORE 'API EXAMPLES'

#     1.  'RandomUser' API - (provides RANDOMLY GENERATED USERS for Testing Purposes)
#(can specify 'user details' like 'gender', 'image', 'email', 'address', 'title', 'first and last name'...)
#MANY 'METHODS' can be USED ON 'Get Response' - 'get_cell()', 'get_city()', 'get_dob()', 'get_gender()', 'get_street()', 'get.state()'...

!pip install randomuser
from randomuser import RandomUser
import pandas as pd

r = RandomUser()   #Create OBJECT for 'random user' (INSTANCE of Class!)
#Use 'generate_users()' to get LIST of RANDOM Users:
some_list = r.generate_users(10)  #for '10 random users'
some_list
#Use 'GET METHODS' to GET PARAMETERS we want to CONSTRUCT a DATASET:
name = r.get_full_name()   #get 'FULL NAME' of the 'RandomUser'!     
email = r.get_email()      #'volkan.arican@example.com'
#Have MANY OTHER '.get' Methods - '.get_state()', '.get_gender()', '.get_city()'
#'LOOPING' for the '10 users' in 'some_list':
for user in some_list:
    print(f"{user.get_full_name()} - {user.get_email()}")
#(got 'full name' and 'email' of EACH Random User!)


#Getting their 'PICTURES':
for user in some_list:
    print(f"{user.get_full_name()} - {user.get_picture()}")
#(EACH Picture is a '.jpg' file - COOL!)

#can CREATE a 'DataFrame' FROM 'GETTING' Data:
# FIRST create FUNCTION - use 'GET METHODS' to Create 'DICTIONARY' of Key:Value Pairs, for EACH of '10 RANDOM USERS'
def get_users():
    users = []
    for user in RandomUser.generate_users(10):
        users.append({"Name":user.get_full_name(), "Gender":user.get_gender(), "City":user.get_city(), "State":user.get_state(), "Email":user.get_email(),"DOB":user.get_dob(),"Picture":user.get_picture() })
    return pd.DataFrame(users)  #Then CONVERT 'Dictionary LIST' into DATAFRAME!

df1 = get_users()
df1.head()
print(df1.columns.values)   # List of Columns - 'Name', 'Gender', 'City', 'State', 'Email', 'DOB', 'Picture'
#NICE! Now Can USE this DATAFRAME for TESTING, DATA ANALYSIS- GREAT!


#         2. 'Fruitvice' API  (= WEB Service, providing DATA on ALL KINDS of FRUITS! FREE to USE!)
#NOW will use 'requests' Library (=MORE COMMON WAY to use APIs - for 'WEB' Services!):
import requests
import json
url = "https://fruityvice.com/api/fruit/all"
data = requests.get(url)
data  #is Just a 'response' object!
#RETRIEVE DATA (from 'data' object) using 'json.loads(json_file.text)' function:
results = json.loads(data.text)
results  #gives 'LIST of DICTIONARIES' on Many FRUIT TYPES (with Name, Family, Genus, Nutritions...)
#CONVERT the 'JSON File' to Pandas 'DATAFRAME':
results_df = pd.DataFrame(results)
results_df
# - BUT... 'nutritions' Column is STILL NESTED as DICTIONARY!!
#IF we have NESTED 'Dictionary' COLUMN, must use 'pd.json_normalize(results)' to FLATTEN/NORMALIZE the Data
results_df2 = pd.json_normalize(results)
results_df2    #Now have SPLIT 'nutritions' into INDIVIDUAL COLUMNS for 'nutritions.calories', 'nutritions.fats', 'nutritions.carbohydrates'...

# ALTERNATIVE WAY:   (MUCH COOLER!!!)
# - 1. use 'one_dict' (Our USER-DEFINED FUNCTION from our FIRST 'nba API example'):
results_one_dict = one_dict(results)
one_dict_df = pd.DataFrame(results_one_dict)
#But STILL have 'nutritions' Column of 'DICTIONARY VALUES'
# -  2.  '.apply(pd.Series)' - ONE way to CONVERT DICTIONARY Rows into NEW COLUMNS of a DataFrame:
nutritions_df = one_dict_df['nutritions'].apply(pd.Series)
#'.apply()' - APPLIES a FUNCTION (here 'pd.Series') to CONVERT EACH 'Dictionary' ROW  INTO a 'SERIES' = NEW COLUMNS for EACH 'KEY'
#NOW can use 'pd.concat([df1, df2], axis=1)' to CONCATENATE 'nutritions_df' TO the ORIGINAL 'one_dict_df' : 
final_df = pd.concat([one_dict_df, nutritions_df], axis=1)    #CONCATENATES this 'nutritions_df' TO the ORIGINAL 'one_dict_df'
final_df = final_df.drop('nutritions',axis=1)     #DROPS OLD 'nutritions' Column of dictionaries
#(Other Method is Quicker, but THIS is more LOGICAL/READABLE/MAKES More SENSE!)

# Lets 'SELECT DATA' from this DataFrame - SIMPLE!:
cherry = final_df[final_df["name"] =="Cherry"]
cherry  #'FILTERS' for JUST the 'Cherry' ROW
# EXTRACT Specific Row VALUES (using .iloc()):
print(cherry.iloc[0][2])   #accessing Row 1, Column 3 ('family' = 'Rosaceae')
print(cherry.iloc[0][4])   #accessing Row 1, Column 5 ('genus' = 'Prunus')
#Finding 'Calories' in 'Banana':
banana = final_df[final_df["name"] =='Banana']
print(banana.iloc[0][5])   # 96 calories

#Another SIMPLE EXAMPLE - (using FREE PUBLIC APIs)
get_url = "https://cat-fact.herokuapp.com/facts"
data2 = requests.get(get_url)
results = json.loads(data2.text)
print(results)
cats_df = pd.DataFrame(results)
cats_df.drop(columns = "status", axis = 1, inplace=True)  #'status' column NOT NEEDED, so just DROP it like so!
print(cats_df)





#%%               PANDAS
import pandas as pd
import numpy as np
import seaborn as sns
# df = pd.read_csv('.csv file')  
# df = pd.read_excel('xlxs_path', sheet_name = "sheet_we_want")

#OR, could CREATE a 'DATAFRAME FROM 'DICTIONARY' (if dictionary is like 'Key: [values]'!)
songs = {'Album':['Thriller', 'Back in Black', 'The Dark Side of the Moon', 
                  'The Bodyguard', 'Bat Out of Hell', 'Back in Black'],
         'Released':[1982, 1980, 1973, 1992, 1977, 1980],
         'Length':['00:42:19', '00:42:11', '00:42:49', '00:57:44', '00:46:33', '00:42:11']}
songs_df = pd.DataFrame(songs)   #CONVERTS to pandas 'DataFrame Object'
print(songs_df)
#Accessing ('SELECT') SPECIFIC COLUMN(s) using [['column_name']] DOUBLE-BRACKETS:
columns_from_df = songs_df[['Album', 'Released']]  # = NEW Dataframe of ONLY SELECTED Columns!

#Accessing 'SPECIFIC VALUES'/'SPECIFIC ROWS' from DataFrame: 
# 'BY INDEX' = use '.iloc[row_index, (col_index])'
# 'BY LABEL' - 'loc[row_index,..]'
print(songs_df.iloc[2])   #e.g. accesses 3rd ROW Index
print(songs_df.loc[2])    #SAME HERE - (since LABEL is 'INDEX' Column)

#Can Even do 'SLICING' to select 'SPECIFIC RANGE' of 'ROWS':
print([songs_df[1:3]])


#RENAME COLUMN Header with 'df.rename()':
df.rename(columns = {"old_header":"new_header"}, inplace=True)
#note: 'inplace=True' simply means ORIGINAL DataFrame is CHANGED!

#'SORTING' DataFrame BY a COLUMN:
#Use 'df.sort_values(by=['Column_Name'], ascending = True or False)
songs_df = songs_df.sort_values(by=['Released'], ascending = False)
songs_df.head(10)     #Top 10 - SORTED by YEAR (DESCENDING)    

# To get ONLY 'UNIQUE VALUES' FROM a COLUMN:
unique_albums = songs_df['Album'].unique()  

# 'dataframe.drop(["columns Name" or "Rows Index"], axis ('=1' for 'COLUMNS', =0 for ROWS), inplace = True)
songs_df.drop([5], axis = 0, inplace = True)   #DROPS 'ROW 6' FROM the DataFrame PERMENANTLY!
songs_df

#Can also 'DROP ROWS BASED ON CONDITION' (use INDEX and '.loc' to FIND Rows to Drop):
for x in df.index:
    if df.loc[x, 'Do_Not_Contact'] == 'Y'
    df.drop(x, axis=0, inplace=True)
#e.g.2 - Dropping IF 'Phone Number' is BLANK:
for x in df.index:
    if df.loc[x, 'Phone_Number'] == '':
        df.drop(x, axis=0, inplace=True)



#                     MISSING VALUES:

# 'df.isnull()' - gives MISSING VALUES as 'BOOLEANS' (True = MISSING, False = NOT Missing)
#THEN, could LOOP for EACH COLUMN in the Dataframe, and do 'missing_data[column].value_counts()'
missing_data = songs_df.isnull()
for column in missing_data.columns.values:   #GIVES 'Column Headers' as a LIST, so can ACCESS for EACH COLUMN EASILY!
    print(column)
    print(missing_data[column].value_counts())
#  'df[column].value_counts()' - COUNTS for DIFFERENT 'VALUES' found in 'EACH COLUMN'
# '.value_counts()' is BEST for BINARY COLUMN Values (1 or 0, True or False). 
#Side Note - '.value_counts()' ONLY works on Panads 'SERIES' NOT Dataframe, so use SINGLE BRACKETS [column] to get SERIES (not [[columns]])!
#Just use '.to_frame()' to CONVERT BACK to DATAFRAME!

#OR, view Nulls as PERCENTAGE of TOTAL:
for col in df.columns:
    pct_missing = np.round(np.mean(df[col].isnull()),2)  #find missing data
    print(f'{col} - {pct_missing}%')  #for each column, as f-string

#EASIEST WAY to VIEW NULLS in EACH COLUMN:
isnull_df = df.isnull().sum()    #just do 'sum()' afterwards!!!

#note: OR could use '.notnull()' - just opposite!
notnull_df = df.notnull().sum()

#REPLACE NULLS with BLANKS:
df.replace(np.nan, '', inplace=True)    
    
#OR can DROP all NULLS/Missing Data:
dataframe.dropna(subset=["specific column"], axis=0, inplace = True)
#REMOVES 'NA/MISSING' Values from SPECIFIC COLUMN ('Subset') 
# (Note: 'inplace=True' argument 'DIRECTLY CHANGES' the 'DATAFRAME'. WITHOUT 'inplace' will NOT CHANGE the ORIGINAL DATAFRAME!)
#        'axis=0' modifies ROWS of DataFrame 
# 'np.NaN' = a 'Not a Number' Value. 
#e.g. REPLACE with 'NaN' using 'df.replace("?", np.nan, inplace=True)'  - replaces EACH OCCURENCE of '?' character WITH 'nan'
#WHY? - So, NOW can use 'dropna'! 
    


#'df.nunique()' = NUMBER of UNIQUE VALUES per COLUMN 
#(may be handy when performing some EDA)
songs_df.nunique() 
#viewing just for 'Album' Column: 
songs_df['Album'].nunique()   #5 unique values in 'Album' Column

#CORRECT WAY to find Number of Respondents for EACH Question:
#FIND ALL values in 'Answer' Column which are 'NOT NULL' (NULLS NOT NEEDED in the Count!):
respondents = df_merged[df_merged['Answer'].notnull()]
respondents   #just FILTERED to ONLY get ROWS where 'Answer' is NOT NULL
#Using .groupby WITH '.nunique()'
total_respondents = respondents.groupby('Question')['Respondent ID'].nunique().reset_index()
respondents.groupby('Question')['Respondent ID'].nunique().reset_index()
#So, for EACH Question, get Total Number of (UNIQUE) Respondents!



#can CHANGE SETTINGS so ALL ROWS are displayed when CALLING Dataframe:
pd.set_option('display.max_rows', None)
#Or, can set Number of DECIMAL PLACES for FLOAT Columns (e.g. 2dp):
pd.set_option('display.float_format', lambda x: '%.2f'% x)

    
# 'dataframe.duplicated()' - VIEW 'DUPLICATE ROWS' ONLY  
duplicate_rows = df[df.duplicated()]

# 'df.drop_duplicates()' - REMOVES ALL 'DUPLICATE ROWS':
songs_df = songs_df.drop_duplicates()   #had 'Back in Black' TWICE, so REMOVED this!
#this is usually the FIRST STEP of ANY DATA CLEANING!



#Viewing CORRELATIONS between NUMERIC Columns:
df.corr()  #simple correlation table
#BETTER to use HEAT MAP to achieve this
sns.heatmap(df.corr(), annot=True)
matplotlib.rcParams['figure.figsize'] = (20,8) 
plt.title("Correlation Matrix for Numeric Features")
plt.xlabel("Movie Features")
plt.ylabel("Movie Features")  
plt.show()

#Use 'df.str.contains("string")' like 'LIKE' in SQL
#Use 'df.isin(["value1", "value2"])' like 'WHERE...IN ()' in SQL


# df.'groupby()' - GROUPS DATAFRAME (AGGREGATION!)
#df.groupby([by], axis=0, as_index = False, ...)
df.groupby('city').mean().sort_values(by='temperature', ascending=False)    
#grouped by 'city', then SORTED by 'temperature' DESCENDING (highest to lowest)


# 'JOINS' ('Merging' DataFrames based on Primary-Foreign Key Pairs) in PYTHON (with pandas):
#  'pd.merge(df1, df2, on = ["col1", "col2"])' 

#REORDER Columns from '2022 - 2017' TO '2017-2022':
#using '[::-1]'  -  df2 = df2[::-1]  
#Can TRANSPOSE dataframe using '.transpose()'
df_transposed = df.transpose() #FLIPS around COLUMNS and ROWS
#Can use this to Convert WIDE Data to LONG Data when PLOTTING

#             UNPIVOTTING Data (WIDE DATA into LONG DATA)
#Use '.melt()' method:
#         ' df.melt(id_vars, 
#                   value_vars, 
#                   var_name, 
#                   value_name) '
#  id_vars = identifier variables (these are kept the SAME)
#  value_vars = what variables to UNPIVOT (becomes LONG Data)
#     (great documentation for 'Pandas' on their website)

#e.g. converting Multiple Columns for 'Questions' into One Column for QUESTIONS and Another Column for ANSWERS
#select column headers used in 'id_vars'
id_vars = list(df_modified.columns[:8])
id_vars 
#select column headers used for 'value_vars' (i.e. UNPIVOT All 'QUESTIONS' (1 Column contains QUESTIONS, Another contains ALL ANSWERS)):
value_vars = list(df_modified.columns[8:])
value_vars
#Now, also add NAMES for 'Variable' Column = QUESTIONS (and Subquestions) and 'Value' Column = ANSWERS
#(given by 'var_name' and 'value_name')
df_melted = df_modified.melt(id_vars=id_vars, 
                             value_vars = value_vars,
                             var_name = "Question and Subquestion", 
                             value_name = "Answer")





#            REPLACING COLUMN VALUES: (.replace())
# Can use df["Column"]'.replace()' to REPLACE 'COLUMN VALUES' with NEW VALUES (JUST like using '.replace()' with STRINGS!!!)
songs_df["Album"].replace("The Bodyguard", "The Guard", inplace=True)

# list(df.columns.values) - GIVES the COLUMN NAMES as a LIST! Useful to Look over!

# 'ADD HEADER ROW'  -  'df.columns' = ["Header 1", "Header 2"...] 

# 'df.info()' - provides INFO about DataFrame 
#(including INDEX 'dtype' and 'COLUMNS', 'NON-NULL Values' and MEMORY USAGE)


#Viewing SPECIFIC DATA TYPES in DATAFRAME - '.select_dtypes()'
df.select_dtypes(include = 'number')
#include = 'number' is for NUMERICAL DATA, include = 'object'
#useful to CHECK to ensure they make sense!     

    
# CONCATENATE 2 DataFrames Columns TOGETHER with:
#'pd.concat([df1, df2], axis=1)'
#('axis=1' concatenates AS COLUMNS (ALONG), 'axis=0' concatenates AS ROWS (DOWN) - like 'UNION' in SQL!)


#  'FILTERING ROWS' in Pandas Dataframe - Specify Condition and Put INSIDE SELECTION of DataFrame:
df_new = songs_df[songs_df['Released'] >= 1980]
print(df_new.head())    #Now is JUST for ROWS where 'Released' >= 1980

#'SET' specific 'COLUMN' AS 'ROW-INDEX COLUMN', using '.set_index(" ")'
df_setting = songs_df.set_index("Album")
df_setting
#NOW when we use 'loc', MUST USE 'LABELS' in 'INDEX Column' as 'ROW' Argument TOO!:
df_setting.loc["Back in Black", 'Released']

#OR, can use 'df.index' = new_index (SPECIFY a NEW INDEX COLUMN, using a LIST)

#SAVE 'DataFrame' TO 'CSV File' using '.to_csv('.csv')' method
df_new.to_csv('saving_DataFrame_as_CSV.csv', index = False)
#note: 'index = False' just REMOVES INDEX COLUMN of DataFrame - SIMPLE!



#           'SERIES'   ( = like 'SINGLE COLUMN' 1D ARRAY or a '1D DATAFRAME')
data = [10, 20, 30, 40, 50]
series = pd.Series(data)      #'pd.Series(data)'
series   #essentially a 'single column' array!
# Note: SELECTING DataFrame Column with 'SINGLE BRACKETS' '[]' will CONVERT it to a 'Series' (i.e. Not a DataFrame anymore)

#NOTE: 'pd.Series' can CONVERT 'DICTIONARY' into a DataFrame (Series for EACH KEY = DataFrame) TOO!
# ACCESS ELEMENTS in a 'Series', by INDEX/INTEGER Positions:
#1. BY 'LABEL'
print(series[2])   #accessed element with 'label/index' of '2'
#2. Access by POSITION (using '.iloc[]')
print(series.iloc[2]) 
#Accessing MULTIPLE ELEMENTS:
print(series[1:4])       
                #(ALL VERY SIMPLE - JUST like LIST ELEMENTS!!!)


#           MORE PRACTICE - 'MANIPULATING DataFrames':
x = {'Name':['Rose', 'John', 'Jane','Mary'], 
     'ID':[1,2,3,4], 
     'Department': ['Architect Group', 'Software Group', 'Design Team', 'Infrastructure'], 
      'Salary':[100000, 80000, 50000, 60000]}
df = pd.DataFrame(x)     #first just created the DF (as usual!)
print(df.head())

id = df[['ID']]
id                   #JUST selecting the 'ID' Column
type(id)  #dataframe object
multiple_columns = df[['Department', 'Salary', 'ID']]
multiple_columns

#                 'Another Example' (PRACTICE):
dict = {'Student':['David', 'Samuel', 'Terry', 'Evan'],
        'Age':[27,24,22,32],
        'Country':['UK', 'Canada', 'China', 'USA'],
        'Course':['Python', 'Data Structures', 'Machine Learning', 'Web Development'], 
        'Marks':[85, 72, 89, 76]}
dataframe = pd.DataFrame(dict)
dataframe

b = dataframe[['Marks', 'Course']]   #selecting 2 columns
b

#Practicing 'loc' and 'iloc' to ACCESS 'SPECIFIC VALUES' from DataFrame:
dataframe.iloc[0,2]  #1st Row , 3rd Column , gives 'UK'
dataframe.loc[2, "Course"]  #3rd Row, "Course' Column 'LABEL' - 'Machine Learning'

#'SET' SPECIFIC 'COLUMN' AS 'INDEX COLUMN', using '.set_index("Name")'
df2 = dataframe.set_index("Student")
#NOW when we use 'loc', MUST USE 'LABELS' in 'INDEX Column' as ROW Argument:
df2.loc["Terry", 'Marks']
df2

# 'SLICING' using 'loc' and 'iloc' (Giving SPECIFIC ROW and COLUMN 'RANGES'):
dataframe.loc[1:2, "Country"]  #Accessing 2nd to 3rd Rows, of 'Country' Column
dataframe.iloc[1:3, 0:1]    #accessed '2nd-3rd Rows', '1st Column' 
#Note: '.loc' includes BOTH Values in RANGE of Slice, so 1:2 will include 2nd AND 3rd Columns!
#     (whereas '.iloc' is same as 'LIST or STRING Slicing', where 1:2 would ONLY be for '2nd' column) 



#%%              'SPLITTING DataFrame COLUMNS' in 'PYTHON'
#'SPLITTING' One 'COLUMN' into 'MULTIPLE COLUMNS'

# If Column Contains 'STRINGS':
#i.e. Can SPLIT 'by DELIMITER':
#Use ' column.str.split('delimiter') '
#      df['split_column'] = df['column'].str.split('delimiter') 
#BUT.. gives us Just ONE COLUMN of 'LISTS' as Row Values (since .split outputs a LIST)
#We WANT '2 NEW COLUMNS' - SO?
#INSTEAD, do:    
#    df[['new_col1', 'new_col2']] = df['column'].str.split(' ', n=max_splits(optional), expand=True)
#OPTIONAL - could specify 'number of splits' (n=...)
# NOTE: 'expand=True' (needed if non-uniform number of splits i.e. SPLITS of DIFFERENT LENGTHS, therefore can place 'None' for any MISSING Values, OTHERWISE would give ERROR!)
#NEED 'expand=True' to make it WORK - get 2 Spit COLUMNS now!

#Example of 'STRING' SPLITTING:
split_df = pd.DataFrame({'AB': ['A1-B1', 'A2-B2']})
split_df
split_df['Split'] = split_df['AB'].str.split('-')
split_df   #BUT, get Column of LISTS - NOT 2 SPLIT Columns!
#So? Must instead SPECIFY 'NEW SPLIT Columns' (BELOW):
split_df[['A','B']] = split_df['AB'].str.split('-', expand=True)
split_df      #AWESOME! NOW has SPLIT into 2 SEPARATE COLUMNS, NOT 1 Column of a 2 Element LIST! 
#ALTERNATE WAY - can do 'TUPLE UNPACKING':
split_df['A'], split_df['B'] = split_df['AB'].str.split('-', n=1).str  
split_df   #THIS way WORKS TOO!


#IMPORTANT EXPLANATION - What is '.str' attribute? 
#  = a MAGIC Object to COLLECT 'METHODS', to TREAT each ROW ELEMENT as a STRING, for a COLUMN.
#    Then can APPLY any of these 'Methods' for EACH ELEMENT 'as EFFICIENTLY AS POSSIBLE'
#e.g. Could apply ANY 'STRING MANIPULATION' METHOD - str.lower(), str.upper()...

#'.str' ALSO lets you 'INDEX' (SLICE PART of the STRING/Object) - get 'SPECIFIC INDEX' or 'slice/range' from a 'STRING' Value:
split_df['AB'].str[:2]   #returns 'A1' for Row1 and 'A2' for Row2  - SIMPLE!
#EVEN can 'INDEX' on the 'ELEMENTS' of 'LIST OUTPUT', PRODUCED FROM '.split()'
split_df['AB'].str.split('-')
split_df['AB'].str.split('-').str[1]     #SIMPLE! Gives 2nd Element of EACH 'List Row' - B1, B2 


#              SPLITTING COLUMN which CONTAINS 'LIST' ROWS:
#HOW can we 'SPLIT a Column' IF each 'ROW' is a 'LIST' of ELEMENTS?
# - so want to SPLIT into 'NEW COLUMNS' for 'EACH ELEMENT' of the 'LIST'
#EXAMPLE:
dictionary = {'teams': [['SF', 'NYG'],['SF', 'NYG'],['SF', 'NYG'],
                ['SF', 'NYG'],['SF', 'NYG'],['SF', 'NYG'],['SF', 'NYG']]}
df1 = pd.DataFrame(dictionary) #first creating DataFrame FROM Dictionary (AS USUAL!)

df1  #have 'teams' column, which has LISTS with 2 Elements
#HOW can we SPLIT this DataFrame INTO 2 COLUMNS (for EACH List ELEMENT)?
#Use  'df[['new_col1', 'new_col2]] = pd.DataFrame(df1['teams'].tolist(), index=df1.index)'
df1[['team1', 'team2']] = pd.DataFrame(df1['teams'].tolist(), index=df1.index)
df1   #WORKED! Now Columns SPLIT for EACH 'team'!!!

#LONGER Way? - split on ',' then strip '[' and ']':
df1['teams'] = df1['teams'].astype('str')
df1
df1[['team1', 'team2']] = df1['teams'].str.split(', ', expand=True)
#striping each column:
df1['team1'] = df1['team1'].str.strip('[')
df1['team2'] = df1['team2'].str.strip('] ' ' ')
#ABOVE Method is BETTER!


#Can then 'DROP' the Original Column MANUALLY, with 'df.drop()':
df1.drop('teams', axis=1, inplace=True)

#'ANOTHER WAY' (Creating NEW DATAFRAME, WITHOUT ORIGINAL COLUMN):
df_split = pd.DataFrame(df1['teams'].to_list(), columns=['team1', 'team2'])
df_split                              #just do 'columns = ['col1, 'col2] as an ARGUMENT





#%%              STRIP/REMOVE TEXT/WHITESPACE from 'Data Frame COLUMN' Values:

#AGAIN, the '.str' attribute comes in handy!
#BEST WAY - Use df['column'] = df['column'].str.strip('characters_to_strip')
df['Country'].str.strip('')
#Can specify ONE STRING of EVERYTHING TO STRIP:
#  str.strip('.../_')   - handy!


#OR, can use '.str.replace(" ", "") to REMOVE Whitespace characters, replacing with nothing!
#(add 3rd argument in 'replace' IF you want to specify NUMBER of REPLACEMENTS to make
# (e.g. ',replace("bill", "william", 3))  - done 3 times ONLY))

#OR if ONLY wanting to strip LEADING whitespace, use 'str.lstrip'  -  SIMPLE!

#Simple Example - using 'str.replace' to STANDARDIZE Column Values:
df['Paying Customer'] = df['Paying Customer'].str.replace('Yes', 'Y')
df['Paying Customer'] = df['Paying Customer'].str.replace('No', 'N')



#                  REGULAR EXPRESSIONS for Data Cleaning

#REMOVING an ENTIRE SUBSTRING from a String?
#CANNOT use Wildcard '*' like in Find and Replace in Excel.
#So? must use 're.sub(r'\(.*\)', '', 'text (###)' (REGULAR EXPRESSION)

#Can use REGULAR EXPRESSION inside '.replace' too:

#e.g. want ALL Phone Numbers in column to have '123-435-2341' format
df['Phone Number'].str.replace('[^a-zA-Z0-9]', '')
# Use '^' to match ALL EXCEPT LETTERS to 'a-z' OR 'A-Z' OR NUMBERS '0-9'
#then REPLACE these exceptions with '' (Nothing/BLANK!)
#now can REFORMAT to ADD the '-' within:
#-first convert 'Phone Number' Column to 'STRING' (str) 
df['Phone Number'].astype("str")
#-Use FUNCTION to SLICE and ADD '-' in between:
def apply_function(x):
    x = x[0:3] + "-" + x[3:6] + '-' + x[6:]
    return x
#APPLY function TO COLUMN values, using df['Column'].apply(function)
df['Phone Number'].apply(apply_function)
#Notice some NA values, so can REMOVE these:
df['Phone Number'].str.replace('nan--', '')
df['Phone Number'].str.replace('Na--', '')



#%%                   CATEGORICAL DATA
#Different Summay Stats are needed for CATEGORICAL Data 
#(above, for 'Quantitative/Numeric Data, used CENTRALITY/Central Tendency)
import pandas as pd
import numpy as np
#Example 1 - 2015 NCY Tree Census Data (Survey of 50000 Trees in City, collected by park department employees...)
nyc_trees = pd.read_csv(r"C:\Users\Ezhan Khan\Documents\PYTHON PRACTICE\Fundamental_Maths\nyc_tree_census.csv")
print(nyc_trees.head())  
# "status", "health, "spc_common", "neighborhood" are ALL CATEGORICAL VARIABLES!
tree_species = nyc_trees["spc_common"] #Extract Columns (AS USUAL)


#       NOMINAL Categorical Variables - MODE ONLY!
#e.g. neighborhood, spc_common ...These have NO ORDER/Rank NOR Numerical equivalents - CANNOT CALCULATE Mean, Median, Nor ANY Measure of SPREAD!
#    ONLY 'MODE' using '.value_counts()' Method:
tree_counts = nyc_trees['neighborhood'].value_counts() 
#This calculates COUNT of EACH VALUE IN a Variable COLUMN (neighborhood)
#Returns a 'TABLE OF FREQUENCIES', DESCENDING ORDER, MODE = TOP ROW (DEFAULT)
print(tree_counts)  #'Annadale-Huguenot...has MOST TREES - 950
#Can EXTRACT THIS 'Neighborhood' using '.index[0]' - Each Value Count is an Index TOP to BOTTOM (Like list)!
print(tree_counts.index[0])   #JUST Prints Neighborhood Name now - BETTER!


#    ORDINAL Categorical Variables - MODE AND MEDIAN

#   MODE - SAME WAY AS ABOVE!

#   MEDIAN 
#(Need to CONVERT to CATEGORY Data Type, THEN use '.cat.codes' attribute to assign EACH category a numerical value)
# 1. First View LIST of 'UNIQUE' CATEGORIES/VALUES (WIHTIN the Categorical COLUMN/Variable):
health_statuses = list(nyc_trees["health"].unique())
health_statuses      #['Good', 'Poor', 'Fair' nan]
                      
# 2  Must MANUALLY ORDER this - Lowest to Highest (REMOVING 'nan')            
health_ordered = ['Poor', 'Fair', 'Good'] 

# 3. Need to assign a 'LEVEL' (RANK) to EACH Category:
#    CONVERT to 'CATEGORY' Type!
nyc_trees['health'] = Categorical(nyc_trees['health'], health_ordered, ordered=True)
    #(3 Inputs - 'Categorical Column', Ordered Categories, 'ordered=True')
#OR, can use 'df[column] = df[column].astype("category")' to do the SAME 
    
# 4. NOW, CONVERT Categories TO NUMERICAL Values (assign each category a numerical value):
#    Use 'cat.codes' ATTRIBUTES - Calculate MEDIAN! 
median_index = np.median(nyc_trees['health'].cat.codes)
print(median_index) # '2' - tells us Index of CATEGORY IN 'Unique' List
median_category = health_ordered[int(median_index)]
print(median_category)  #'Good' is MEDIAN category!


#      IMPORTANT POINT - 'MEAN' CANNOT be done for ORDINAL! 
#'Ordered List' of Unique Categories is EQUALLY SPACED - Represented as EQUALLY SPACED 'INTEGER' NUMBERS
#BUT,'Categories' as NOT ALWAYS EQUALLY SPACED THEMSELVES!
     #(Differences BETWEEN Individual Categories MAY VARY)
#For MEAN - SPACING 'MUST' BE EQUAL!
#So, CANNOT Use these 'cat.codes Numbers' for MEAN - Since SPACING MATTERS FOR MEAN!
#e.g. Happiness Score - 1=very unhappy, 5='very happy' - CANT Always Assume that DIFFERENCE between Categories are Equal!
#        Example Why - MEAN 'Trunk Diam 'Category':  
tree_census2 = pd.read_csv(r'C:\Users\Ezhan Khan\Documents\PYTHON PRACTICE\Fundamental_Maths\nyc_tree_census2.csv')    
#EASY AS USUAL for simple 'Quantitative' Variable like 'trunk diameter':
mean_diam = np.mean(tree_census2['trunk_diam'])
print(mean_diam)      #11.27 inches Mean Diameter
#BUT Now ALSO have an ORDINAL 'Trunk Diameter CATEGORIES'
          #JUST FOLLOW USUAL PROCEEDURE:
print(list(tree_census2['tree_diam_category'].unique()))
#    ['Medium-Large (10-18in)', 'Large(18-24in)'....]
tree_size_categories = ['Small (0-3in)', 'Medium (3-10in)', 'Medium-Large (10-18in)', 'Large (18-24in)', 'Very large (>24in)']
tree_census2['tree_diam_category'] = pd.Categorical(tree_census2['tree_diam_category'],tree_size_categories, ordered=True)
mean_diam_index = np.mean(tree_census2['tree_diam_category'].cat.codes)
print(mean_diam_index)  #1.97 
mean_diam_category = tree_size_categories[int(mean_diam_index)]          
print(mean_diam_category)   #IN 'Medium (3-10in)' 
#NOT CORRECT! - Calculated Quantintatively as '11.27 inches' - which is in 'Medium-Large'
#Calculated WRONG CATEGORY - Mean for ORDINAL Data does NOT WORK! 


#  PERCENTILES/IQR - 'SPREAD' for ORDINAL Categories:  (SAME WAY - using cat.codes!)
# SD and Variance are ALSO NOT Interpretable since Depend ON MEAN
#SO? Can use PROPORTIONS of Data IN a RANGE = PERCENTILES/Quantiles
 #e.g. 'RANGE' of 80% of Data' = FROM '10th Percentile' TO '90th Percentile' (90-10=80%!)
 #   This 'Range' Between 10-90th Percentiles = 'IQR' (INTERQUARTILE RANGE)    

#Example - For Our TREE CENSUS 2, have 'tree_diam_category' (Ordinal Variable):
tree_census2['tree_diam_category'] = pd.Categorical(tree_census2['tree_diam_category'], tree_size_categories, ordered=True)
tree_census2['tree_diam_category'] = tree_census2['tree_diam_category'].astype("category")
tree_census2['tree_diam_category'].dtype
#Calculate 25th and 75th Percentiles: JUST use CAT CODES - (SAME WAY AS MEDIAN)
p_25_index = np.percentile(tree_census2['tree_diam_category'].cat.codes, 25)
p_25_category = tree_size_categories[int(p_25_index)]
print(p_25_category)    # 'Medium (3-10in)' 
p_75_index = np.percentile(tree_census2['tree_diam_category'].cat.codes, 75)
p_75_category = tree_size_categories[int(p_75_index)]
print(p_75_category)       # 'Large (18-24in)'
#EASY! So, IQR=75-25='50% of Data' is from 'MEDIUM' TO 'LARGE' Tree Diameter!


#           TABLE of 'PROPORTIONS' (% OF the TOTAL in Data):
#'Mode' for Categorical uses '.value_counts()' (AS COVERED ALREADY)
#This GIVES a 'TABLE OF 'FREQUENCIES'
#BUT, BETTER to CONVERT to 'PROPORTIONS' (%) 
#EXAMPLE - Tree Census 'Status':   
proportions = tree_census2['status'].value_counts()/len(tree_census2['status'])
# 1. So just DIVIDE by LENGTH OF the Tree Census 'Status' Column
print(proportions)    #Alive=0.953, Stump=0.0267,Dead=0.0194
# 2. OR, can Just do 'normalize=True' WITHIN value_counts (Simpler Way!):
proportions = tree_census2['status'].value_counts(normalize=True)

#IMPORTANT- 'MISSING VALUES' are Coded as 'NaN' in Table of Proportions
#BY DEFAULT these are NOT COUNTED!
#INCLUDE MISSING DATA by adding ARGUMENT 'dropna = False' in .value_counts() (i.e. DONT DROP IT!)
#Note: '/len(..)' Method ALREADY DOES THIS! RESULT VALUES MAY VARY Slightly IF 'NaN' Present - DEPENDS ON METHOD USED!!
print(tree_census2['health'].value_counts(dropna = False, normalize=True))  
print(tree_census2['health'].value_counts(normalize=True))  
#As expected, Proportions are LARGER WIHTOUT 'NaN' Values!


#                 BINARY CATEGORICAL VARIABLES 
# ONLY TWO CATEGORIES (y/n, True/False, 1/0...)
# SUM = to Calculate Frequency AT a Certain Value (e.g. for '1' or '0' - 0 NOT Counted, so JUST Adds '1's - NICE!)
# PROPORTIONS = 'MEAN'!    e.g. 'TRUE/FALSE' or 'y/n' 
#'CONDITIONAL STATEMENTS' - Convert NON-BINARY TO BINARY 'TRUE or FALSE'
#NOW just 'True' or 'False' - so USE ABOVE METHODS for Proportions OR Frequencies! - EASY! 

#Example: Frequency and Proportion of Trees 'Alive'
print(tree_census2['status']=='Alive')  #JUST Rows where 'Alive'
alive_frequency = np.sum(tree_census2['status']=='Alive')
print(alive_frequency)    # '47695' are ALIVE
alive_proportion = np.mean(tree_census2['status']=='Alive')
print(alive_proportion)   # '0.9539' are 'Alive'
#Proportion (Mean) = Frequency 'TRUE'/ TOTAL Number of Column Elements
print(alive_frequency / len(tree_census2['status']))
#    (JUST ANOTHER WAY - SAME ANSWER!!!! - EASY!)

#Example 2: For 'trunk_diam > 30'
print(tree_census2['trunk_diam']>30) #tells us WHICH column elements this Condition is TRUE for and which are FALSE
giant_frequency = np.sum(tree_census2['trunk_diam']>30)
giant_proportion = np.mean(tree_census2['trunk_diam']>30)
print(f"Frequency: {giant_frequency}, Proportion: {giant_proportion}")             
   

#%% CATEGORICAL VARIABLES - SUMMARY EXAMPLE: 'Automobile Evaluation Data'
#Cost and Physical Attributes of 1000 Cars
import numpy as np
import pandas as pd
car_eval = pd.read_csv(r'C:\Users\Ezhan Khan\Documents\PYTHON PRACTICE\Fundamental_Maths\car_eval_dataset.csv')

#Want to Know 'FREQUENCIES' for 'Manufacturer Country' Categories:
frequencies_table = car_eval['manufacturer_country'].value_counts()
print(frequencies_table)   #MOST Cars are from JAPAN
#Accessing for United States (4th element):
print(frequencies_table.index[3])
#Now  want Table of Proportions:
proportions_table = car_eval['manufacturer_country'].value_counts(normalize=True)
print(proportions_table)    

#NOW, want 'buying_cost' Categories:
buying_costs = list(car_eval['buying_cost'].unique())
buying_costs_ordered = ['low', 'med', 'high', 'vhigh']
car_eval['buying_cost'] = pd.Categorical(car_eval['buying_cost'], buying_costs_ordered, ordered=True)
cost_median_index = np.median(car_eval['buying_cost'].cat.codes) 
median_cost = buying_costs_ordered[int(cost_median_index)] 
print(median_cost)   #'med' is Median Cost

#Table of Proportions for 'luggage' Category:
luggage_proportions = car_eval['luggage'].value_counts(dropna=True, normalize=True)
print(luggage_proportions)   #No missing Values here..

#Frequency and Proportions for SPECIFIC CATEGORIES IN a 'doors':
frequency_5more = np.sum(car_eval['doors']=='5more')
print(frequency_5more)    #'246' have cars with '5more' Doors
proportion_5more = frequency_5more/len(car_eval['doors'])  
print(proportion_5more) #0.246 proportion 
#(OR could have found this using 'MEAN' - EITHER WAY WORKS!)


#   'EXTRACTING' from ONE Column 'GIVEN CONDITION in ANOTHER COLUMN':
US_buying_costs = car_eval[car_eval['manufacturer_country']=='United States']['buying_cost']
print(US_buying_costs)  #SAME AS WHAT WE DID WITH WEATHER DATA!
#Could use this for Table of Frequencies - if we want Data from ONE COLUMN GIVEN the CONDITION in the OTHER:
print(US_buying_costs.value_counts())  
#Lets us know that is MOSTLY 'LOW' Cost for 'US' Cars - NICE!
print(car_eval[car_eval['manufacturer_country']=='Japan']['buying_cost'].value_counts())
#Whereas, for 'JAPAN', is Mostly 'MEDIUM' Priced Cars!




#%%                               NUMPY
import numpy as np
#1D Numpy - SIMPLE Linear Algebra/VECTORS! Already in Maths Notes!
u = np.array([1,2])  
v = np.array([3,1])

u.dtype   #Integers 'int32' - use 'dtype' to check DATA TYPE of an ARRAY
u.size    # '.size' gives LENGTH/Number of ELEMENTS IN an Array   
u.ndim    #Number of DIMENSIONS (1D array = 1 Dimension)
u.shape   # (2,)
#Can also Access Elements and SLICE Arrays JUST like Lists Slicing, or DataFrame ROWS Slicing:
v[0]    # '3'
#Overwrite/CHANGE Values IN the Array, JUST like Lists:
u[0] = 7

#USING NUMPY, can SIMPLY do 'Addition and Subtraction' of Vectors (1D Arrays)!
z = u + v
print(z)     #  [4, 3] vector AS EXPECTED!
#PRODUCT (Multiplication) of 2 Numpy Arrays:
z = u*v
print(z)       # [3, 2] vector AS EXPECTED!
#'DOT' PRODUCT:
print(np.dot(u,v))   #returns '5'  (= (1*3) + (2*1) = '5')
#ADDING 'CONSTANT' to an array, ADDS it TO EACH Element:

#JUST like LISTS, can ITERATE through EACH ELEMENT in an Array ('for' loop)

#More Practice:
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[1:5:2])    #only accesses between 1:5, going up in 2 Steps.
#Slicing JUST the EVEN Elements:
print(arr[1::2])
#Assign Value of '100' to ALL elements in a Range:
arr[1:4] = 100
print(arr)   #Now, have REPLACED the 2nd to 4th elements with '100'

#SIMPLE STAT FUNCTIONS in NUMPY  -  np.mean(), np.std(), np.max(), np.min()  
# Even get  'np.pi', 'np.sin()'

# Creating EVENLY-SPACED ARRAY -  'np.linspace(start, end, number_values)' 
x = np.linspace(0, 2*np.pi, 100)     #generates an EVENLY-SPACED ARRAY of '100 values'!
plt.plot(x, np.sin(x))        #creates simple SIN PLOT of 'x' Array
#('np.linspace()' can be used to CREATE 'BINS' - see 'Data Analysis with Python')

#                '2D' NUMPY ARRAYS: 
#SAME EXACT THING (covered in Maths Notes too!)
a = [[11,12,13], [21,22,23], [31,32,33], [22,22,22]]
A_array = np.array(a)  
print(A_array)  #NOTE! - EACH 'SUB-LIST' becomes 'ROW' of the 2D ARRAY (MATRIX)
A_array.shape   #4 rows, 3 columns
A_array.ndim    #ndim refers to NUMBER of NESTED LISTS
A_array.size    #= 12 , since has 12 ELEMENTS
#INDEX 2D Arrays JUST like INDEXING '2D Lists':
A_array[0][0]   #FIRST ROW of FIRST Column  - SUPER EASY!     (Notation '[0,0]' could ALSO be used)
A_array[2][1]   #3rd Row, 2nd Column  =  3
#SLICING 2D Arrays (ALSO SIMPLE):
A_array[0][0:2]   #Slices for FIRST ROW, 1st and 2nd COLUMNS
A_array[1:3, 1]  #2nd and 3rd ROWS, 2nd Column =  'array([22,32]) '
#ADDING 2D ARRAYS (EXACT SAME as for Matrix Addition/1D Addition!)
B = np.array([[32,23,43],[14,13,26],[17,29, 30], [13,24,17]])
print(A_array + B)  #SUPER EASY!
#MULTIPLYING 2D Arrays BY CONSTANT/SCALAR= EXACT SAME!
A*10
#SIMPLE MULTIPLYING of 2 '2D Arrays' - EACH 'LIKE' Element is Multiplied with the OTHER:
A*B       # = array([[352, 276, 559],[294,286,598], [527,928,990]]'

#  MATRIX '(DOT) MULTIPLICATION' (with 2D Arrays) - SLIGHTLY DIFFERENT:
#MUST make sure 'COLUMNS in Matrix A' = 'ROWS in Matrix B'
A = np.array([[0,1,1],[1,0,1]])   #'2 Rows', 3 Columns
B = np.array([[1,1],[1,1],[-1,1]])   #'3 Rows', '2 Columns'

print(np.matmul(A,B))   #  '[0,2],[0,2]'  #DOT Multiplication!
# (using 'np.dot()' ALSO works here!)

#TRANSPOSE a MATRIX/2D Array - use '.T' attribute:
print(A)
print(A.T)    #  [[0 1] [1 0] [1 1]]







#%%       CLASSES PRACTICE
class Car:
    max_speed = 120  #Class Variable/Attribute
    def __init__(self, make, model, color, speed=0):   #CONSTRUCTOR Variable
        self.make = make
        self.model = model      #these are the 'INSTANCE Variables'
        self.color = color
        self.speed = speed    #initially, set to 0 
    def accelerate(self, acceleration):                #Normal METHOD!
        if self.speed + acceleration <= self.max_speed:
            self.speed += acceleration
        else:
            self.speed = self.max_speed
    def get_speed(self):                             #Normal METHOD       
        return self.speed

#Make 2 'INSTANCES/OBJECTS' for this 'Car' Class:
car1 = Car("Toyota", "Camry", "Blue")
car2 = Car("Honda", "Civic", "Red")
   
car1.accelerate(30)  # object.variable - ACCESS the METHOD 'accelerate'
car2.accelerate(20)   # Increases SPEED of Car1 by 30km/h, Car2 by 20km/h

#PRINTING the CURRENT SPEEDS:
print(f"The '{car1.make} {car1.model}' has speed of {car1.get_speed()}")
print(f"The '{car2.make} {car2.model}' has speed of {car2.get_speed()}")

dir(Car)   #look at the Attributes/Methods of 'Car' (at END of List, can see our ADDED ATTRIBUTES/'Methods' and 'Variables' for the Class!)

import matplotlib.pyplot as plt
# may need to add '%matplotlib inline' if plotting in JupityrNotebook!  

class Circle(object):
    def __init__(self, radius=3, color='blue'):    #Constructor
        self.radius= radius       #Instance attributes/variables
        self.color = color
    def add_radius(self, r):      #normal method, adding to radius
        self.radius = self.radius + r  #increase radius of circle
        return self.radius
    def drawCircle(self):         #method to PLOT the Circle
        plt.gca().add_patch(plt.Circle((0, 0), radius=self.radius, fc=self.color))
        plt.axis('scaled')
        plt.show() 

redcircle = Circle(10, 'red')   #specifies 'radius' as '10' here
redcircle.radius            #accessing INSTANCE ATTRIBUTE of 'radius'!
redcircle.color          # Instance Attribute is 'red'

redcircle.drawCircle()   #CALLS the METHOD 'drawCircle' and PLOTS the Circle!
redcircle.add_radius(2)
print(redcircle.radius)    #increased to '12' (since ADDED '2')
redcircle.add_radius(5)    #added on '5'
print(redcircle.radius)    #increased radius to '17'

bluecircle = Circle(radius=100, color = "darkblue")
bluecircle.drawCircle()

#SIMILARLY can create 'RECTANGLE' Class:
class Rectangle(object):
    def __init__(self, width=2, height=3, color='r'):
        self.height = height 
        self.width = width
        self.color = color
    def drawRectangle(self):
        plt.gca().add_patch(plt.Rectangle((0, 0), self.width, self.height ,fc=self.color))
        plt.axis('scaled')
        plt.show()
skinnybluerectangle = Rectangle(2,3,'darkblue')
skinnybluerectangle.height
skinnybluerectangle.width
skinnybluerectangle.color
skinnybluerectangle.drawRectangle()

#Creating VEHICLE Class:
class Vehicle:
    color = "White"
    def __init__(self, max_speed, mileage):
        self.max_speed = max_speed
        self.mileage = mileage
    def seating_capacity(self, seating_capacity):
        self.seating_capacity = seating_capacity
    def display_properties(self):
        print("Properties of Vehicle:")
        print("Color:", self.color)
        print("Maximum Speed", self.max_speed)
        print("Mileage", self.mileage)
        print("Seating Capacity:", self.seating_capacity)

vehicle1 = Vehicle(200, 50000)   #instantiate the object
vehicle1.seating_capacity(5)
print(vehicle1.seating_capacity)  #accessing instance variable 'seating_capacity'
vehicle1.display_properties()    #displays the printed properties


#%%        EXPECTATION Handling:  (TRY - EXCEPT)

#                    COMMON Python ERRORS:
# 'ZeroDivisionError' 
# 'ValueError' (e.g. converting String to Integer is NOT POSSIBLE) 
# 'Name Error' - using UNDEFINED VARIABLE
# 'FileNotFoundError' - Trying to access file, but is NOT FOUND
# 'IndexError' - trying to ACCESS an Element in a List, OUTSIDE the INDEX RANGE
# 'KeyError' - Dictionary Key does NOT EXIST
# 'TypeError' - Object is used in INCOMPATIBLE MANNER, like Concatenating String and Integer (e.g. "Hello" + 5)
# 'AttributeError' - METHOD used ON an OBJECT (object.variable) is WRONG! Object does NOT HAVE that 'ATTRIBUTE/Method' 
# 'ImportError' - IMPORT a Module that is UNAVAILABLE

#Can END a 'try-except' with an 'ELSE' statement to SAY 'WHEN it is RUN'
#Can ALSO include 'finally' Statement to 'CLOSE' a FILE (if a file is run) at the END
a = 1
try:
    b = int(input("Please enter a number to divide a"))
    a = a/b
except ZeroDivisionError:
    print("The number you provided cant divide 1 because it is 0")
except ValueError:
    print("You did not provide a number")
except:
    print("Something went wrong")
else:                                #What to Write IF NO EXCEPTIONS 
    print("success a=",a)
finally:                             #'finally' Statement to END 'try-except'
    print("Processing Complete")

def safe_divide(numerator, denominator):
        try:
            numerator / denominator
        except ZeroDivisionError:
            print("Error: Cannot Divide by Zero")            
safe_divide(30,0)

#For any 'GENERIC, UNSPECIFIED EXCEPTION' - use 'except Exception as e:' (say we dont know the Specific Errors)
def complex_calculation(num):
    try:
        result = num / (num - 5)
        print(f"Result: {result}")
    except Exception as e:               
        print("error occured during calculation")
complex_calculation("No More")


#%%        SETS     (EASY! - Similar to Lists and Dictionaries) 
# = Type of COLLECTION, input DIFFERENT 'TYPES'
# UNORDERED (NO Element Position/Order)
# 'ONLY UNIQUE' ELEMENTS are allowed in a Set - VERY IMPORTANT!
# ANY 'DUPLICATE' Elements are 'REMOVED AUTOMATICALLY', so 'ONLY UNIQUE' elements REMAIN!

#are within CURLY BRACKETS  {  }
# 'set(list)' function CONVERTS a 'LIST into a SET' (removes duplicates so ALL UNIQUE elements ONLY)
album_list = ["Michael Jackson", "Thriller", 1982, "00:42:19", "Pop, Rock, R&B", 46.0, 65, "30-Nov-82", None, 10.0]
album_set = set(album_list)
album_set    #converted to a 'set' 
music_genres = set(["pop", "pop", "rock", "folk rock", "hard rock", "soul", 
                    "progressive rock", "soft rock", "R&B", "disco"])
music_genres

#  setname.add("new element")  - ADDS Individual Element TO a set (ONLY if it is UNIQUE Element)
A = set(["Thriller", "Back in Black", "AC/DC"])
A.add("NSYNC")
A
# 'set.update(set2)' ALSO works for SETS - add ANOTHER set of VALUES INTO a SET
album_set.update({"Bad", "Off the Wall"})
album_set
#  setname.remove("element")  - REMOVES Element from set
A.remove("NSYNC")
A                     #Note - Also can use '.discard()' or '.pop()' to do SAME THING!
#  'in' command can be used to CHECK for a VALUE (JUST like with Lists, Strings and Dictionary Keys!)
"AC/DC" in A

# When ADDING 2 Sets - is the 'INTERSECTION' of 2 VENN DIAGRAMS
# (i.e. ONLY Elements which are IN BOTH Sets, REMOVING the REST)
# Done using '&' OR '.intersection()' - 'album_set_3 = album_set_1 & album_set_2'
album_set1 = {"AC/DC", "Back in Black", "Thriller"}
album_set2 = {"AC/DC", "Back in Black", "The Dark Side of the Moon"}
album_set1.intersection(album_set2)
album_set1 & album_set2      # = 2 ways of finding the Interset of 2 Sets!

# 'set1.difference(set2)'- used '.difference()' to get ONLY UNIQUE Element(s) from SET 1 (and NOT in Set 2)
album_set1.difference(album_set2)   #'Thriller' is JUST in SET 1, NOT in Set 2
album_set2.difference(album_set1)   #'Dark Side of the Moon' is JUST in SET 2, NOT in Set1

# 'set1.union(set2)' = UNION used if we want 'ALL ELEMENTS' from 'BOTH SETS' (Even If Not in the Other Set)
album_set1.union(album_set2)    # = ALL Elements from BOTH Sets

# To CHECK IF a SET IS/IS NOT a 'SUBSET' of 'ANOTHER' -  using '.issubset()' method
#   Returns TRUE or FALSE:
album_set1.issuperset(album_set2)  # 'FALSE' for 'SUPERSET' (Superset = OPPOSITE of Subset)
album_set2.issubset(album_set1)   #'FALSE' for 'SUBSET'     ()
{"Back in Black", "AC/DC"}.issubset(album_set1)  # TRUE!
album_set1.issuperset({"Back in Black", "AC/DC"})  # TRUE!

# 'set.copy()' - creates COPY of SET, which can be changed WITHOUT AFFECTING ORIGINAL (JUST like with LISTS and DICTIONARIES!)


#%%   EASY LOOP Practice


#FOR Loops = when 'NUMBER of ITERATIONS' needed is 'KNOWN'
#WHILE Loops = need to PERFORM a TASK REPEATEDLY, AS LONG AS the CONDITION is TRUE.
#              (STOPS when Condition is NO LONGER TRUE)

# Use 'enumerate(list)' - IF we want 'ELEMENT' AND its 'INDEX' TOGETHER in Iteration 
squares = ["red", "yellow", "green"]
print(enumerate(squares))   # is JUST an OBJECT! ONLY USEFEUL in a 'LOOP'!
for i,square in enumerate(squares):    #BETTER THAN doing COUNTER '+=1'...
    print(square)
    print(i)

squares = ["orange", "orange", "purple", "orange", "blue"]
newsquares = []
i = 0
while squares[i] == 'orange' and i < len(squares):
    newsquares.append(squares[i])
    i += 1
print(newsquares)

dates = [1982, 1980, 1954, 1967, 1969, 1973, 2000]
i = 0
year = dates[0]

while(year != 1973):    
    print(year)
    i = i + 1
    year = dates[i]   #moves onto NEXT year in the List above
    
print(f"It took {i} repetitions to get out of loop.")

PlayListRatings = [10, 9.5, 10, 8, 7.5, 5, 10, 10]
i = 0
rating = PlayListRatings[0]
while i < len(PlayListRatings) and rating >= 6:
    print(rating)
    i = i + 1 
    rating = PlayListRatings[i]      

Animals = ["lion", "giraffe", "gorilla", "parrots", "crocodile","deer", "swan"]
i = 0
animals_less_than_7 = []

while i < len(Animals):
    if len(Animals[i]) == 7:
        animals_less_than_7.append(Animals[i])
    i += 1
print(animals_less_than_7)


for date in dates:
    if date != 1973:
        print(date)
    else: 
        break


#        Python 'NESTED' LOOPS Practice:
#LOOP WITHIN ANOTHER LOOP
#Really DEPENDS on SITUATION, when Nested Loops are found!
#While Loop inside While Loop (Just for Visual Demonsration!!)
while x>0:
    while y>0:
        print("do something")
#For Loop inside For Loop
count = 0
for x in range(3):
    for y in range(9):
        print("do something")
        count+=1
#For Loop INSIDE a 'While' Loop
while x>0:
    for y in range(9):
        print("do something")
#'While Loop' INSIDE a 'For' Loop
for x in range(3):
    while y <0:
        print("do something")

#PRACTICE 1:
for y in range(3):         #print INNER LOOP '3 Times'
    for x in range(1,10):
      print(x, end="")   #printing 1 to 9
    print( ) #Prints a NEW LINE AFTER every INNER LOOP is DONE! - so EACH Inner Loop is on a NEW LINE! 
#(end="delimiter" argument ADDED IF we want to PRINT on SAME LINE after EACH LOOP (One After Each Other!))
#Delimiter could be ANYTHING - Hyphen, Dash...     
#Note - 'INNER LOOP' is performed FIRST, THEN the OUTER-LOOP is APPLIED ON the INNER LOOP!

#PRACTICE 2 - Creating RECTANGLE:
rows = int(input("Enter the Number of Rows: "))
columns = int(input("Enter the Number of Columns: "))
symbol = input("Enter a Symbol to Use: ")
for x in range(rows):
    for y in range(columns):
        print(symbol, end="")
    print() #e.g. could create a Rectangle from '$' Symbols! Really Simple!





#%%   EASY FUNCTIONS PRACTICE:
    
#COMMON IN-BUILT Functions  
# len(), sum() (i.e. of LIST Elements), sorted() or .sort() (.sort() will CHANGE the EXISTING List)  

#    User_defined Functions (SIMPLE THEORY Explanations):
#SCOPE of Variable = PART where Variable is ACCESSIBLE
#IF Variable is Defined OUTSIDE of a FUNCTION, it can be ACCESSED ANYWHERE = within 'GLOBAL' Scope.
#(so is called 'GLOBAL' VARIABLE)

#LOCAL Variables = ONLY WITHIN the FUNCTION. Value remains UNCHANGED

Rating = 9   #'Rating' is defined OUTSIDE of Function, so is GLOBAL Variable
def ACDC(y):                            
    '''
    This Simple Function just
    adds on a number variable 'y'
    to the 'Rating' Variable (which is a GLOBAL Variable)
    '''
    print(Rating)    #'Rating' is accessed FROM OUTSIDE the Function, now INSIDE!
    return (Rating+y)
#Note: used DOCSTRINGS (Documentation Strings) with '''  ''' to EXPLAIN INSIDE the FUNCTION
#      (this can be accessed OUTSIDE the function with 'help(function_name)')
help(ACDC)

#USEFUL - 'global' keyword CAN be used BEFORE a LOCAL Variable (in a function), so it is CHANGED to GLOBAL Variable (ACCESSIBLE OUTSIDE FUNCTION, now in GLOBAL SCOPE!)
artist = "Michael Jackson"
def printer(artist):
    global internal_var 
    internal_var = "Whitney Houston"
    print(artist,"is an artist")
printer(artist)  # FIRST must CALL (Instantiate) the FUNCTION, 
internal_var    #NOW can ACCESS this 'INTERNAL' VARIABLE FROM the FUNCTION!

#If NUMBER of ARGUMENTS are UNKNOWN  -  use def function(*args) as 'argument'
#Then can SPECIFY 'ANY NUMBER of ARGUMENTS' when CALLING the function!
def printAll(*args): # All the arguments are 'packed' into args which can be treated like a tuple
    print("No of arguments:", len(args)) 
    for argument in args:
        print(argument)
#printAll with 3 arguments
printAll('Horsefeather','Adonis','Bone')
#printAll with 4 arguments
printAll('Sidecar','Long Island','Mudslide','Carriage')

#Can ALSO PACK these values into a sort of 'DICTIONARY', if we use '**args'
def printDictionary(**args):      
    for key in args:
        print(key + " : " + args[key])
printDictionary(Country='Canada',Province='Ontario',City='Toronto')


#%%    EASY 'WHILE LOOP' Practice (using 'input()' to make it SIMPLER!)
name = input("Enter Name: ")

#Use 'while' if we WANT something to 'RUN INDEFINITELY' 'WHILE' 'CONDITION is MET'.
#When CONDITION is NO LONGER MET, then we CLOSE the Loop.
#Need way to 'ESCAPE the While Loop' (avoid Infinite) - so put in an INPUT Prompt INSIDE the While loop to BREAK the INFINITE Looping! 

while name == '':
    print("You did not enter your name")
    name = input("Enter Name:")           #'input()' helps to BREAK the INFINITE LOOP - GOOD! Is more 'POTENTIALLY' Infinite

print(f"Hello {name}")   #Only AFTER 'while' loop, THIS code is RUN (follows the ORDER!)


#Example 2 - (Same Thing):
age = int(input("Enter Age: "))

while age < 0:
    print("Age can't be negative")
    age = int(input("Enter your age "))

print(f"You are {age} years old")   #ONLY AFTER 'while loop, this code is RUN (as above)

 
#Example 3 ('Logical Operator', STILL SAME!):
food = input("Enter a food you like (q to quit); ")

while food != "q":
    print(f"You like {food}")
    food = input("Enter another food you like (q to quit): ")
    
print("bye")   #ONLY print THIS code when WHILE Loop is FINISHED


#Example 4  -  More Logical Operators (SIMPLE!):
num = int(input("Enter a # between 1-10: "))

while num < 1 or num > 10:
    print(f"{num} is not valid")
    num = int(input("Enter a # between 1-10: ")

print(f"Your number is {num}")   #REALLY SIMPLE! Again, ONLY prints this IF the WHILE Loop is DONE!



#%%   PYTHON LAB - 'TEXT ANALYSIS' (CLASSES, Functions, Strings, Dictionaries - ALL in ONE!)

#Text Analysis ('Text MINING') - extract 'MEANINGFUL INFORMATION/INSIGHTS' FROM 'TEXTUAL' Data 
#  e.g. Converting TEXT to LOWERCASE, FINDING and COUNTING occurances of All UNIQUE WORDS...

#SCENARIO - Analysing Customer Feedback on a Product. 
#           Customer REVIEWS given as STRINGS

#Defining CLASS and ATTRIBUTES to 'Analyze Text':
class TextAnalyzer(object):
    def __init__(self, text):      
        #First, using .replace() to REMOVE PUNCTUATION:   (JUST like 'find-replace' in Excel!)
        formattedText = text.replace('.', '').replace("!",'').replace('?','').replace(',','')
        #Make Text 'LOWERCASE':                               
        formattedText = formattedText.lower()
        self.fmtText = formattedText    #ASSIGNED value as 'INSTANCE Variable'

    #Adding a METHOD to 'SPLIT Text' into INDIVIDUAL WORDS (by ' ')
    def freqAll(self):
        wordList = self.fmtText.split(' ')   #split into INDIVIDUAL WORDS 
        freqMap = {}
        for word in set(wordList):
            freqMap[word] = wordList.count(word)  #Put into DICTIONARY with 'Key = Word', 'Value' = List 'COUNT' for EACH 'word'
        return freqMap
    #Now will create Another METHOD, to PASS IN a SPECIFIC 'WORD' Argument to be FOUND
    #(this is just to TEST the 'freqAll' function ON the WORD)
    def freqOf(self, word):
        freqDict = self.freqAll()   #RETURNS 'freqMap' (= freqDict here)
        if word in freqDict:        #CHECKS for the 'word' IN the 'Dictionary KEYS'
            return freqDict[word]   #RETURN the 'VALUE' (COUNT) for the 'word'
        else:
            return 0

#Now can EXECUTE and TRY OUT these CLASS FUNCTIONS (for a Given String)
given_string="Lorem ipsum dolor! diam amet, consetetur Lorem magna. sed diam nonumy eirmod tempor. diam et labore? et diam magna. et diam amet."
analyzed = TextAnalyzer(given_string)
print(f"Formatting the Text String: \n{analyzed.fmtText}")
#( = INSTANCE VARIABLE - this REMOVES all PUNCTUATION and Converts string to ALL LOWERCASE)
freq_each_word = analyzed.freqAll()
print(freq_each_word)     #applied '.freqAll()' method, to CONVERT into DICTIONARY (KEY = Word, VALUE = Count)
#(Called 'freqAll' Method to Calculate FREQUENCY of EACH Unique Word)  
word = "lorem"    #Now CHECKING for a SPECIFIC WORD's COUNT
frequency = analyzed.freqOf(word)          
print(frequency)
print(f"The word '{word}' appears {frequency} times.")
#         ALL MAKES SENSE! Great!

    
























