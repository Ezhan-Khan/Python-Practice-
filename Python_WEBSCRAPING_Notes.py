
#%%                   INSPECTING Web Pages with HTML

#HTML = 'Hypertext'-Markup-Language
#describes ELEMENTS of a WEBPAGE
#Need to KNOW HTML BASICS so we KNOW WHAT to TAKE FROM the Website

#e.g. (see word file)
#STARTS and ENDS with '<html>'
#can contain a <head>  .... to '</head>'
#can contain a <body> ... to </body>
# (this is the overall HIERARCHY of the entire HTML file!)

#WITHIN HTML, contain CLASSES, TAGS, ATTRIBUTES, VARIABLE STRINGS...more!

#'TAGS' - e.g. <p> tag and '<title>' tag
#within 'title' Tag, have VARIABLE 'STRING' - '<title>My First Web Page</title>'


#Example - 'https://www.scrapethissite.com/pages/forms/'
#(great website to PRACTICE WEBSCRAPING HTML)
#Can VIEW HTML Code for ANY WEBSITE
#just 'Right-Click' webpage and press'Inspect'

#this page has MORE COMPLEX HTML code WITHIN, but has SAME UNDERLYING STRUCTURE
#starts with <html>, ends with '</html>', has '<head>' to '</head>', has '<body>' to '</body>'
#viewing <title> tag (WITHIN <head>), see it is same as title given in the webpage (as specified here!)

#INCREDIBLY USEFUL TIP - CLICK 'ARROW ICON' to USE Cursor to SELECT PARTS of WEBPAGE to VIEW Related Code in the HTML
#e.g. hover and click title of the webpage to FIND <title> in the HTML file!
#this is a GREAT WAY to EASILY FIND something IN our HTML file, FROM Webpage.
#USE this to HELP DECIDE WHAT to WEBSCRAPE in our code!
#e.g.  <tr...> tag (table columns)...

#'href' is a HYPERLINK which takes you to ANOTHER webpage
# <p> tag usually means a PARAGRAPH

#TABLES are given using <table class="table"> ....</table>
#-contains 


#%%                    'BeautifulSoup and Requests':

#These libraries are great for webscraping in python
#First, import packages needed:
from bs4 import BeautifulSoup  #(from 'bs4' module)
import requests
#Now, Specify the URL (i.e. WHERE we are getting HTML from?)
url = "https://www.scrapethissite.com/pages/forms/"
#Send 'GET REQUEST' TO that URL, returns 'RESPONSE OBJECT' (i.e. API request)
page = requests .get(url)   #IF '200' - means SUCCESSFUL! (Note: 204 or 400, 401, 404...means BAD REQUEST/server not found!)
#Viewing the webpage's HTML, this one is STATIC (some webpages, like for Amazon, may UPDATE regularly though)
#So, by PULLING Data INTO Python, get SNAPSHOT of HTML AT THAT TIME!

#Now, pass HTML data INTO a BeautifulSoup INSTANCE 
#(specify request response above AS '.text' AND 'html' to PARSE output INTO 'HTML format')
soup = BeautifulSoup(page.text, 'html')
print(soup)
#Page contains a LOT OF INFORMATION!
#recognize the 'th' tag, 'td' tag and 'tr' tag, links (href)...

#(not necessary) - can MAKE NICER-LOOKING to view
#use '.prettify()' method (just gives the output a CLEARER Hierarchy!):
print(soup.prettify())


#%%                   'Find' and 'Find_all'

#NOW, will query this HTML to retrieve ONLY INFO which we NEED
#HTML has TONS of INFO!!!
#But HOW can we JUST get INFO we NEED? - use 'Find' and 'Find_all' Methods!

#e.g. Say we JUST want the '<div id="page"> tag
#Use '.find' to extract the FIRST 'div' TAG IN the HTML:
soup.find('div')  
#Or, use 'find_all' to extract ALL 'div' TAGS:
soup.find_all('div')

#What about getting a SPECIFIC 'div' TAG?
#-now, need to specify 'CLASS'
#e.g. '<div class="container">, '<div class="col-md-12">
#USE these to FIND what we WANT!
#SIMILARLY, could SPECIFY 'href', 'a' Tag...WHATEVER we WANT!
#ALL are 'ATTRIBUTES' = variables written WITHIN TAGS (e.g. id="nav-homepage", class=..., href="....")

#Calling Attribute INSIDE 'find_all' too:
soup.find_all('div', class_ = 'col-md-12')

#Trying this for '<p ...>' tag (=PARAGRAPHS of TEXT):
soup.find_all('p')    #gives MULTIPLE 'p' tags 
#'inspect' the HTML file, clicking on Paragraphs ON the Webpage, seeing IF they MATCH OUR result!
#see that top paragraph is given as '<p class="lead">
#next paragraph is given as '<p>  <i class="glyphicon glyphicon-education"> .....

#IF we JUST WANTED the 'p' tag where class="lead"
soup.find_all('p', class_='lead')

#HOW can we PULL JUST the 'TEXT ITSELF'? (i.e. whatever is WITHIN the TAG)
#Here, MUST USE 'find()', with '.text' Attribute:
soup.find('p', class_='lead').text
#TEXT output given inside quotes (so is STRING!)


#What about pulling in 'team_name' (table header)
#TABLE contains '<tr>' tag (=a TABLE ROWS, INCLUDING HEADER as first row!)
#INSIDE <tr> tags, get '<th>' (HEADER Values)
soup.find_all('th') #extracts ALL COLUMN HEADER Names
#DATA WITHIN TABLES (ROWS) is given in <td' tags (INSIDE the 'tr' tags for the row )
#Getting just the FIRST HEADER 'Team Name' (use '.find' instead)
soup.find('th').text


#%%                      Project - WEBSCRAPING ALL DATA in a WEB TABLE 

#NOW will get an ENTIRE WEB TABLE INTO a PANDAS DATAFRAME
#this way can start manipulating the data fully, analyse it...do whatever we want with it!

#Here, will use a DIFFERENT TABLE, from WIKIPEDIA
# 'List of largest companies in the United States by revenue'
#'https://en.wikipedia.org/wiki/List_of_largest_companies_in_the_United_States_by_revenue'

#As usual, import in libraries, get HTML using BeautifulSoup and Requests
url = 'https://en.wikipedia.org/wiki/List_of_largest_companies_in_the_United_States_by_revenue'
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html')
print(soup)
#Pulls in HUGE amount of HTML data
#Need to VIEW the HTML on the Webpage, find WHAT we NEED
#Only want the 'TABLE' of DATA on Largest Companies

#Viewing the CLASS for 1st Table on Webpage:
  #        <table class="wikitable sortable jquery-tablesorter">
#2nd Table Webpage:
   #       <table class="wikitable sortable jquery-tablesorter">
#all tables appear to have SAME 'CLASS'
#since multiple tables, may need to use 'INDEXING' with 'find_all' to get SPECIFIC Table 
soup.find_all('table')
#this gives ALL TABLES here ("wikitable sortable" class for both)

#Can use 'INDEXING' (since is a LIST of sorts!):
table = soup.find_all('table')[1]   #first actual table ('1' gives a citations card, which is a TABLE, but NOT IMPORTANT!)
print(table)  #(just SAVE the table we WANT to a Variable!)
#Repeat this, JUST for CLASS 'wikitable sortable' 
soup.find('table', class_='wikitable sortable')

#Just extracting the 'th' tags (HEADERS) on 'table' (first table):
world_titles = table.find_all('th')
world_titles
#EACH 'header' is WITHIN its OWN 'th' tag

#HOW can we 'EXTRACT EACH INDIVIDUAL HEADER' (as TEXT)? 
# - Use 'LIST COMPERHENSION' to LOOP THROUGH and CLEAN EACH ELEMENT (convert to TEXT, STRIP any unecessary characters...):
world_table_titles = [title.text.strip() for title in world_titles ]    
world_table_titles  #given as LIST of HEADERS!
#CLEAN EACH element, REMOVING '\n' within each
#(used '.strip()' to achieve this - removes unecessary characters!)
#NOW have a CLEAN LIST of HEADERS for 1st Table!

#PUT this List of HEADERS INTO a PANDAS DATAFRAME:
import pandas as pd
#(as usual, in notes)
company_df  = pd.DataFrame(columns=world_table_titles)
company_df.head()     #EMPTY Dataframe currently


#Start PULLING IN 'ROWS (='tr') of HTML Table, to POPULATE DATAFRAME
column_data = table.find_all('tr')
print(column_data[:2])   #can SLICE to JUST view first few elements (otherwise TOO LONG and MESSY!)
#LOOP to get DATA (='td') WITHIN EACH ROW:
for row in column_data:
    row_data = row.find_all('td')
    individual_row_data = [data.text.strip() for data in row_data]  
    print(individual_row_data)
#'EACH ROW' is GIVEN as 'INDIVIDUAL LISTS' (100 Rows)!

#Good, but EACH TIME, data is NOT SAVED
#So? Must SAVE EACH ROW created in Loop TO the DATAFRAME
#ADD ROWS to DataFrame USING 'loc'
for row in column_data[1:]:
    row_data = row.find_all('td')
    individual_row_data = [data.text.strip() for data in row_data]  
    #check CURRENT LENGTH of DataFrame
    length = len(company_df)  
    company_df.loc[length] = individual_row_data
#IMPORTANT POINT - MIGHT get 'ValueError' - "cannot set a row with MISMATCHED COLUMNS" 
#i.e. Numnber of 'ELEMENTS' being INSERTED INTO Data MUST 'MATCH Number of Columns' IN that DataFrame

#ISSUE? - FIRST ROW of 'individual_row_data' is EMPTY LIST '[]'
#So? SLICED THIS OUT - 'column_data[1:]' (above!)
company_df.head()   #AWESOME!!! ADDED ALL ROWS!! Really Cool!

#FINALLY, can EXPORT this DataFrame INTO a CSV file:
company_df.to_csv('python_webscraping_wikitable_as_csv.csv', index=False)



#%%                MORE PRACTICE - 'Hockey Teams' Table

hockey_url = "https://www.scrapethissite.com/pages/forms/?per_page=100"
page = requests.get(hockey_url)
soup = BeautifulSoup(page.text, 'html')

soup.find_all('table')
table = soup.find_all('table')[0]
print(table)
#(Only 1 table on this webpage, so use [0] index)
#this table just has (class = "table")
soup.find('table', class_ ='table')   #SAME THING!

headers = table.find_all('th')
print(headers)
#converting headers to TEXT and STRIPPING empty space
table_headers = [header.text.strip() for header in headers]    
table_headers  #nicely cleaned list of headers!

import pandas as pd
#(as usual, in notes)
hockey_df  = pd.DataFrame(columns=table_headers)
hockey_df.head()     #EMPTY Dataframe currently
len(hockey_df)  #'0' rows currently

column_data = table.find_all('tr')
print(column_data)   #extracts ALL ROWS of Table

#Get 'DATA' ('td') from EACH ROW ('tr') of table:
for row in column_data[1:]:
    row_data = row.find_all('td')
    individual_row_data = [data.text.strip() for data in row_data]      
    #check CURRENT LENGTH of DataFrame
    length = len(hockey_df)  
    hockey_df.loc[length] = individual_row_data

hockey_df.head()
#all rows from HTML Table ADDED to the DataFrame - NICE!

#Save to CSV file:
hockey_df.to_csv("webscraping_hockey_table.csv", index=False)

#Webscraping is a SIMPLE yet VERY USEFUL SKILL to have!
































#%%                 IBM Python Course - 'WEBSCRAPING' Notes

#                 HTML BASICS (See Codecademy Course!)
# Say we want to find info on Basketball Players from a Website 
# Can use 'HTML TAGS' and View HTML COMPOSITION of the Page
# '<body>' of HTML is what we are INTERESTED in

# Get 'Hyperlink Tags' (clicking takes you to a website)

#e.g. WIKIPEDIA - can SELECT HTML Element and INSPECT it (also get CSS and JavaScript)

#EACH HTML Document can be REFERRED to as 'HTML TREE'
# - is a TREE structure, with <html> as the PARENT Tag, then with INDENTED parts for <head>, then Indended further for INNER Layers...

#HTML Tables - given 'tr' TAGS. FIRST Row has <td>.....<td>   ...so on so on...


#     WEBSCRAPING = Automatically 'EXTRACT INFO' from 'WEBPAGES' (in minutes, using PYTHON
#use 'requests' and 'BeautifulSoup' Python Modules

#    EXAMPLE- finding 'Name and Salary' of 'Basketball Players' from WEBPAGE
from bs4 import BeautifulSoup
#Can STORE the 'WEBPAGE HTML' as a 'STRING:
html = "<!DOCTYPE html><html><head><title>Page Title</title></head><body><h3><b id='boldest'>Lebron James</b></h3><p> Salary: $ 92,000,000 </p><h3> Stephen Curry</h3><p> Salary: $85,000, 000 </p><h3> Kevin Durant </h3><p> Salary: $73,200, 000</p></body></html>"
#PASS the HTML into 'BeautifulSoup()' INSTANCE
soup = BeautifulSoup(html, 'html5lib')
#('BeautifulSoup()' represents the Document as a NESTED Data Structure - TREE-LIKE Objects which can be PARSED)
soup   
#Can make this HTML file LOOK NICER to View, with 'prettify()' function (But NOT NECESSARY):
print(soup.prettify())

#'TAG' Object = HTML Tag in ORIGINAL DOCUMENT (e.g. 'title' tag):
tag_object = soup.title  #EXTRACT the <title>....<title> Object FROM the 'HTML file'
print(tag_object)   # '<title>Page Title</title>'
type(tag_object)    #'bs4.element.Tag' Object TYPE

#Note: if MORE THAN ONE TAG with SAME NAME, then FIRST ELEMENT WITH that Tag Name is CALLED. 
tag_object = soup.h3   #have 3 Tags called '<h3>' 
tag_object    #given as '<h3><b id="boldest">Lebron James</b></h3>'

#If we want to NAVIGATE 'DOWN the TREE BRANCH'?
tag_child = tag_object.b  #since '<b ....</b> is a CHILD, WITHIN '<h3>'
tag_child
#Can ACCESS 'parent' FROM this, using '.parent' Attribute:
print(tag_child.parent)   # - BACK to 'tag_object' (EXACT SAME!)
#'.parent' for 'tag_object' is 'body' Element:
tag_object.parent   #Goes FURTHER OUT to '<body>'!!

#'SIBLING' of 'tag_object' is given as 'paragraph':
sibling_1 = tag_object.next_sibling
sibling_1     # <p> Salary: $ 92,000,000 </p>'  -  <p> = 'Paragraph'
#OTHER SIBLING of 'sibling_1' (and therefore of 'tag_object' too):
sibling2 = sibling_1.next_sibling
sibling2      # <h3> Stephen Curry</h3>   -  '<h3>' = 'HEADER' Element
#'next_sibling' for 'sibling_2':
print(sibling2.next_sibling)   #<p> Salary: $85,000, 000 </p>   = SALARY of 'Stephen Curry'

#Access 'TAG ATTRIBUTES' of HTML - treat TAG like a 'DICTIONARY VALUE':
tag_child['id']    # Gives 'boldest'  -  i.e. Accesses VALUES WITHIN 'TAG'   
tag_child.attrs  #ACCESSES Attribute as DICTIONARY  {'id':'boldest'}  (attribute = Dictionary 'value')
#(Alternative to Above, can use '.get()' function):
tag_child.get('id')  # (see Dictionary Notes!)

#'NAVIGABLE STRING' = Can EXTRACT 'TEXT' from Tag 'as STRING':
tag_string = tag_child.string   # returns as 'STRING' (like Python String)
tag_string    # 'Lebron James' 
type(tag_string)  #bs4.element/'NavigableString'
#Can CONVERT this 'Navigable String' to PYTHON STRING:
str(tag_string)   #SAME! (Python String ALSO called 'Unicode String'!)

#                  'FILTERING' TAGS (for TABLES)
#'find_all' function - FILTERS Tags, BASED on Tag Name, Attributes, Text or COMBINATION of these:
table="<table><tr><td id='flight'>Flight No</td><td>Launch site</td> <td>Payload mass</td></tr><tr> <td>1</td><td><a href='https://en.wikipedia.org/wiki/Florida'>Florida<a></td><td>300 kg</td></tr><tr><td>2</td><td><a href='https://en.wikipedia.org/wiki/Texas'>Texas</a></td><td>94 kg</td></tr><tr><td>3</td><td><a href='https://en.wikipedia.org/wiki/Florida'>Florida<a> </td><td>80 kg</td></tr></table>"
#HERE, Create 'HTML TABLE': 
table_bs = BeautifulSoup(table, 'html5lib')
table_bs
#Use 'find_all('Tag Name') to Extract 'ALL TAGS' with THAT 'NAME' (including 'CHILDREN' TOO!)
table_rows = table_bs.find_all(name = 'tr')
table_rows   
#Here, finds ALL Elements which is TAG Object for 'tr'
#is JUST like Python LIST, where we can 'ACCESS ELEMENTS' by 'INDEX':
first_row = table_rows[0]  #gets FIRST ROW (HEADER)
type(first_row)   #given as 'bs4.element.Tag'
 
#Can even 'ITERATE THROUGH' EACH 'Table Cell':
for i, row in enumerate(table_rows):
    print(f"row {i}")           #ITERATE Through EACH ROW (like list elements!)
    cells = row.find_all("td")   #Finds ALL Table CELLS (Row = 'CELL')
    for j, cell in enumerate(cells):
        print(f"Column {j}, Cell {cell}")  #Iterate THROUGH Variable CELLS for EACH ROW
#so, for EACH ROW/Cell (row 0,1,2...), can EXTRACT VALUES for EACH COLUMN (column 0, 1, 2...)

list_input = table_bs.find_all(name = ["tr", "td"])
list_input   

# IF 'Argument' NOT RECOGNIZED, is TURNED into 'FILTER' (on TAG ATTRIBUTES)
#    i.e. SO can ALSO use 'TAG ATTRIBUTE= ' as an ARGUMENT:    
#e.g. - First 'td' elements have Value of 'id' of 'flight', so can FILTER based on THAT 'id' Value:    
print(table_bs.find_all(id = 'flight'))  # [<td id="flight">Flight No</td>]
#e.g.2 - Finding ALL ELEMENTS with LINKS to 'Florida Wikipedia Page'
list_input = table_bs.find_all(href = "https://en.wikipedia.org/wiki/Florida")
list_input    # '[<a href="https://en.wikipedia.org/wiki/Florida">Florida</a>, <a href="https://en.wikipedia.org/wiki/Florida">Florida</a>]' 
#SETTING 'href' to 'True' (finds ALL TAGS with 'href' ATTRIBUTE)
print(table_bs.find_all(href=True))  #ALL TAGS with 'href'
print(table_bs.find_all(href=False))  #All Tags WITHOUT 'href' Value/Attribute

#e.g.3 - Finding element with ' id = "boldest" ' for 'soup':
print(soup.find_all(id = "boldest"))   # [<b id="boldest">Lebron James</b>]

#SIMILARLY, can SEARCH, for 'STRINGS ATTRIBUTE' INSTEAD of 'tags':
table_bs.find_all(string = "Florida")  #Finds ALL ELEMENTS with 'Florida'  ['Florida', 'Florida']

#INSTEAD of 'find_all()' method, can use 'find()' to get 'FIRST ELEMENT' in the DOCUMENT:
two_tables="<h3>Rocket Launch </h3><p><table class='rocket'><tr><td>Flight No</td><td>Launch site</td> <td>Payload mass</td></tr><tr><td>1</td><td>Florida</td><td>300 kg</td></tr><tr><td>2</td><td>Texas</td><td>94 kg</td></tr><tr><td>3</td><td>Florida </td><td>80 kg</td></tr></table></p><p><h3>Pizza Party  </h3><table class='pizza'><tr><td>Pizza Place</td><td>Orders</td> <td>Slices </td></tr><tr><td>Domino's Pizza</td><td>10</td><td>100</td></tr><tr><td>Little Caesars</td><td>12</td><td >144 </td></tr><tr><td>Papa John's </td><td>15 </td><td>165</td></tr>"
two_tables_bs= BeautifulSoup(two_tables, 'html.parser')
two_tables_bs.find("table")  #e.g. Finds FIRST TABLE using "table" NAME TAG
#FILTERING on 'CLASS ATTRIBUTE' to find SECOND Table:
two_tables_bs.find("table", class_='pizza')   #FILTER on 'class_ ATTRIBUTE' TOO to find for 2nd Table, with "pizza" class_.


# 'WEBPAGE EXAMPLE':
import requests
from bs4 import BeautifulSoup
#Use 'get' Method (like APIs) to GET the WEPAGE Data 
url = "http://www.ibm.com"
page = requests.get(url).text   #Using '.text' to get AS 'TEXT'
#Create 'BeautifulSoup OBJECT':
soup = BeautifulSoup(page, "html5lib")
#FILTER for ALL 'a' Tags (using .find_all('a')):
artists = soup.find_all('a')
artists

for link in soup.find_all('a',href=True):  # in html, hyperlinks are represented by '<a>' tag 
    print(link.get('href'))    #SCRAPES All 'LINKS' -  'https://www.ibm.com/cloud?lnk=intro'

#Get ALL ROWS from a Table:
for row in table.find_all('tr'):  #'tr' = Table ROW in HTML
    cols = row.find_all('td')     #'td'= Table COLUMN in HTML
    color_name = cols[2].string
    color_code = cols[3].text
    print("{}--->{}".format(color_name,color_code))
                 