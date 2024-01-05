
"""
Created on Sun Mar 26 22:33:56 2023

@author: Ezhan Khan
"""
#Run ONLY ONE LINE/SELECTIONS - 'F9'
#Unexpected EOF ("Unexpected End of File")

#%%     Statistics Random Practice on Housing Prices
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def distribution_plot(prices, borough, price_mean, price_median, price_mode):
    sns.histplot(prices, stat='density')
    plt.title("F {borough} Appartment Prices Distribution")
    plt.axvline(price_mean, color='orange',linestyle='dashed', linewidth=3, label='mean')
    plt.axvline(price_median, color='r', linestyle='solid', label='median')
    plt.axvline(price_mode[0][0], color = 'y', linewidth=3, label='mode')
    plt.xlabel('')  
    plt.show()
    plt.clf()
    
Brooklyn = "Brooklyn"
brooklyn_prices = [3200, 3400, 3523, 3542, 3609, 4000, 3092, 3200, 3214, 3219, 3200, 4100, 5000]
brooklyn_mean = np.mean(brooklyn_prices)
brooklyn_median = np.median(brooklyn_prices)
brooklyn_mode = stats.mode(np.array(brooklyn_prices))

distribution_plot(brooklyn_prices, Brooklyn , brooklyn_mean, brooklyn_median, brooklyn_mode)



#%%     Lists Practice
#New Section, More Practice:
import random

list_random = ["I", "am", "Now", "Ready", "To", "learn", "Python!"]  
list_random.append("Good Luck!")
print(list_random)

food_items = ["Cake", "Bread", "Cookie", "Pasta"]
new_food_list = food_items + ["Tart", "Rice"]      #ADDING a LIST TO a LIST
print(new_food_list)

calls = ["Juan", "Zofia", "Amare", "Ezio", "Anaya"]    
print(calls[-1])    

pancake_recipe = ["eggs", "flour", "butter", "milk", "sugar", "love"]
pancake_recipe[-2] = "Xylitol"     #overwriting list element
pancake_recipe.append("almond flour")
pancake_recipe.append("flour")    #.append("element")  
pancake_recipe.remove("flour")    #.remove("element") - FIRST OCCURANCE of it
print(pancake_recipe)

heights = [["Noelle", 61], ["Ali", 70], ["Sam", 67], ["Sam", 64]]
noelles_height = heights[0][1]      # 2D Lists - SAME THING, VERY EASY!!
heights[2][0] = "Sam H"             #PYTHON INDEX STARTS at '0' 
heights[0].remove(61)
print(heights)

store_line = ["Karla", "Maxium", "Martim", "Isabella"]
store_line.insert(3, "Samuel")    #specify INDEX WHERE we want to INSERT and the VALUE - SIMPLE!
Isabella = store_line.pop(-1)    #REMOVE using 'Pop' if 'BY INDEX'!     
print(store_line)

# '.index()' finds INDEX of a Specified Element:
print(store_line.index("Maxium")) 
copy = store_line.copy()  #.copy() method makes COPY of Specified List and Saves as ITS OWN!

range_list = list(range(0,10))    #'list()' function CONVERTS 'range' to a LIST
print(range_list)

specific_range = list(range(3, 15, 2))  
print(specific_range[-3])

letters = ["a","b","c","d","e","f","g"]        #NOW practice SLICING LISTS
print(letters[1:6])     #index 1-5
print(letters[:4])      #First 4
print(letters[4:])      #All BUT first 4
print(letters[-4:])     #Last 4
print(letters[:-4])     #All BUT last 4

count_list = ['a','a','a','a','a',4,4,5,5,6,7,8,2,'d','f','t','y','u','u','u','u']
print(count_list.count('u'))
num_collection = [[100,100], [100,200], [100,200], [100,200], [300,200], [400,500]]
print(num_collection.count([100,100]))      #.count() works SAME WAY for 2D lists

store_line.sort()      #SORT Elements in List ALPHABETICALLY
print(store_line)
sorted_line = sorted(store_line)  #BOTH work! Function OR Method!
print(sorted_line)

my_info = ("Daniel Day Lewis", 65, "Retired Oscar Winning Actor/Master of ones craft")
#This is a TUPLE, which is LIKE a LIST but CANNOT be CHANGED once made. Only can access index))
my_info_name = my_info[0]

names = ['Bruce', 'Bob', 'Al', 'Joe']
ages = ['70s', '70s', '80s', '80s']
names_and_ages = zip(names, ages)  #'zip()' creates 'zip object' 
print(list(names_and_ages))   #just convert from 'zip object' to 2D 'list()' of TUPLES 

#   UNPACKING         (Really Cool!)
#Lets us SELECT SPECIFIC ELEMENTS FROM a List EASILY:
inputs = ['David', 'Tennant', 'Scotland', 'Brown', 'Brown', 50]
first,last, *_,hair_color, age = inputs
#   *_  means we DONT WANT THE REST of the List Elements (middle bit) - so, ONLY returns 1st, 2nd and LAST (age) elements in list!
# 1st, 2nd and last elements are all stored as variables 'first, last, hair_colour and age' QUICKLY!
print(f"{first} is {age} years old. He has {hair_color} hair.") 

# '.removeprefix("....")' method can be used to REMOVE SPECIFIED PREFIX FROM List Elements:
links = ["www.b001.io", "www.youtube.com", "www.wikipedia.org"]
for link in links:
    print(link.removeprefix("www."))   #NOTE: ONLY for Python VERSION '3.9.0' (MAY need to UPDATE!)

#%%     Loops Practice

for iteration in range(6):
    print("Iteration Number: " + str(iteration +1))     #Simple CONCATENATION with '+'!
    
ingredients = ['milk', 'Xylitol', 'vanilla extract', 'dough', 'chocolate']
for i in range(len(ingredients)):
    print(f"Ingredient {str(i+1)} - {ingredients[i]}" )       
#(Instead of just printing AS IS for the above, printed a labeled string BY Ingredient NUMBER)

prices = [30,25,40,35]
last_week = [2,3,5,8,4,2]

totals_list = 0
for i in range(len(prices)):
    totals_list += prices[i] * last_week[i]    #CROSS-MULTIPLIES EACH 'price' with 'week amount', into NEW LIST!
print("Total Price: ${0}".format(totals_list))
print(f"Total Price: ${str(totals_list)}")     #using 'f string' is EASIER WAY to write this!

# 'WHILE Loop' can 'ONLY' be used 'WHILE a CONDITION' is 'MET' (STOPS when Condition NO LONGER MET)
count = 0
while count <= 3:
    print(count)
    count += 1      #OR could do '-= 1'

print(ingredients)
index = 0
while index < len(ingredients):   #Could use 'while' loop
    print(ingredients[index])
    index += 1
for element in ingredients:   #OR can just use 'for' loop!
    print(element)


student_period_A = ["Alex", "Briana", "Daniele"]
student_period_B = ["Dora", "Minerva", "Obie"]
for student in student_period_A:
    student_period_B.append(student)
print(student_period_B)

items_on_sale = ['blue shirt', 'striped socks',  'red headband', 'white suit and black tie', 'black suit and white tie', 'jeans and t-shirt', 'Cozy sweater',  'knit dress', 'dinosaur onesie']
for item in items_on_sale:
    if item == "knit dress":
        print("found it!")
        break              #break = STOPS the Loop WHEN 'knit dress' is found!  
                          
for item in items_on_sale:
    if item == 'white suit and black tie' or item == 'black suit and white tie':
        continue                               #Continue = SKIPS Specified Iteration
    print(item)

number_list = [1,2,-1,4,-5,5,2,-9]
for i in number_list:
    if i <= 0:
        continue
    else:                 #Dont REALLY need 'else', but just looks cool!
      print(i)              #just prints negative numbers - SUPER EASY! 

#NESTED LOOPS (most commonly used for 2D Lists):
project_teams = [ ['Ava', 'Samantha'], ['James', 'Johnathan'], ['Lucille', 'Zed'], ['Edgar', 'Gabriel'] ]    
for team in project_teams:
    for person in team:
        print(person)

numbered_2D = [ [2,3,3,4], [3,4,2], [3,4,5], [4,4,5], [3,4,2]]
initial_total = 0
for first_loop in numbered_2D:     #SAME THING can be done for NUMBERED 2D Lists
    for element in first_loop:
        initial_total += element     #adds all up!
print(initial_total)

#  'List COMPREHENSION' (Elegant Loops, - just another way to write it in ONE LINE)
numbers = [31, 42, 55, -92, -15, 17, 23, 34]
doubled = [num*2 for num in numbers]         #just doubles EACH number in list EASILY
print(doubled) 
range_example = [int - 1 for int in range(5)]
print(range_example)

doubled_only_negatives = [num*2 for num in numbers if num < 0]   #ONLY doubles Negatives
print(doubled_only_negatives)

double_triple = [num*2 if num > 0 else num*3 for num in numbers]  #Double positives, triple negatives
print(double_triple)
#ALL VERY EASY and LOGICAL!

#%%     Functions Practice
def directions_to_station():
    print("Exit Ravenswood Park,")
    print('turn right and follow the road down, ')
    print('cross when road is clear and enter Frithwood Avenue, or use zebra crossing further down to your left')
    print('keep following along Frithwood Avenue until the very end of the road. Should last about 5-10 minutes')
    print('At the end of Frithwood Avenue, turn left and follow along for about 3-5 minutes')
    print('Now you have reached the town centre. Just ahead is the station. Simply cross the road at the crossing when free')
    print('You have arrived at your destination!')
directions_to_station()

def welcome(origin, destination):
    print("Welcome to " + destination + "!")
    print("You came from " + origin +"? Wow that's pretty far!")
welcome("New Hampshire", "Albuquerque")

def trip_cost(uber_cost_per_hour, uber_journey_time, hotel_rate_per_day, days_at_hotel):
    uber_cost = uber_cost_per_hour * uber_journey_time       #using 'arguments' IN CALCULATIONS!  
    hotel_cost = hotel_rate_per_day * days_at_hotel
    print(f"Total Cost of Trip: £{str(uber_cost + hotel_cost)}")
trip_cost(65, 3, 100, 2)          #EASY PRACTICE!


def calculate_with_discount(trip_cost, discount=30):
    print(trip_cost - discount)

calculate_with_discount(395)
print(round(365/3, 2))   #IN-BUILT Function to Round to SPECIFIED D.P

#Returns used WHEN CALLING, to STORE Function VALUES 'INTO VARIABLE'.
#This way this VALUE CAN be ACCESSED OUTSIDE of FUNCTION and REUSED
def pounds_exchange_rate(pounds, exchange_rate):
    return pounds * exchange_rate

Turkish_lira_exchange = pounds_exchange_rate(1000, 23.96)
print(f"£1000 = {str(round(Turkish_lira_exchange))} Turkish Liras ")

#For 'MULTIPLE RETURNS', just SEPERATE EACH with 'COMMA':
daysofweek_data = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
def first_three_weekdays(days):
    first_day = f"Hoy es {days[0]}"           
    second_day = f"Manana es {days[1]}" 
    third_day = f"Passado manana es {days[2]}"      #Accessing List elements 
    return first_day, second_day, third_day         #Returning EACH List Variable  

day_1, day_2, day_3 = first_three_weekdays(daysofweek_data)

print(day_1)
print(day_2)
print(day_3)   #EASY! Makes a lot of sense!

# 'return None' can be specified if we dont want to return anything. Usually not necessary though!

#%%   OTHER COOL FUNCTION-RELATED FEATURES (Not Necessary, but COOL!):

    #'LAMBDA FUNCTION' = QUICK, ONE-USE way to write User Defined Function!
print((lambda x,y: x+y)(1,2))
# '(lambda  arguments: function body)(provide x and y values)'
#SUPER EASY!

#    FILTER Function (APPLY a FUNCTION to ALL VALUES in a LIST/RANGE, RETURN AS LIST)
nums = range(1,1000)
def is_prime(num):    #Checks if a Number is a Prime Number
    for value in range(2,num):
        if (num%value)==0:
            return False
        else:
            return True
prime = filter(is_prime, nums) #Applies 'is_prime' function to EVERY number in the List. IF 'True', will STAY IN the LIST, If FALSE = REMOVE from list!
print(list(prime))  #just converts from 'filter object' to actual List (python does this 'filter object' to store memory)

#    Python DECORATORS:  (what is '@' used for?)
#  Example - Want to 'TIME HOW LONG' FUNCTIONS Take to RUN 
#  COULD Time EACH function INDIVIDUALLY with TIMING Functionality:
#BUT...TIME CONSUMING if we have LOTS of Functions
#So better to do this all in ONE SEPERATE Function:
def tictoc(func):
    def wrapper():
        t1=time.time()        
        func()
        t2=time.time() - t1
        print(f'{func.__name__} ran in 'f'{t2} seconds')
    return wrapper    
import time
@tictoc
def do_it():
    time.sleep(1.3)  #accesses 'sleep()' from 'time' module/library. This lets us TIME HOW LONG we WANT the Function to Take WHEN Running!
@tictoc         #SPECIFIED that we want function to take '1.3' Seconds to run - COOL!!
def do_that():
    time.sleep(0.4)    #want function to take 0.4 secs to run!
#'tictoc' simply Calculates TIME TAKEN for a Function (func) ti
#THEN write '@tictoc' BEFORE EACH Function we want to Time
#python reads this and RUNS the 'tictoc' WITH the Function, returning Time Taken! 
do_it()
do_that()
#Note: 'time.time()' Prints the NUMBER OF SECONDS SINCE 'EPOCH' (point where 'time' begins - for UNIX System = 'January 1, 1970, 00:00:00 at UTC')
#Note: 'func.__name__' is just a way to DISPLAY the NAME OF a Function in a stiring!    


# '**kwarg' INSIDE Function Argument:
def func(**kwarg):    #note: can be  **anything - doesnt have to be 'kwarg'
    print(kwarg)  #kwarg refers to 'KEYWORD ARGUMENTS' which we REFER to when Calling the Function!      
    return
#Now can 'Specify ANY' KEYWORD 'ARGUMENTS' WE WANT!!!
print(func(a=1,b=2,c=3))   # {'a':1, 'b':1, 'c':3} 
#CONVERTS these 'keyword arguments' INTO 'DICTIONARY'!
def func2(**k):
    ans = k['a']*k['b']*k['c']   #e.g. just multiplying 'Values' in the Dictionary
    print(ans)    #Access values from DICTIONARY
    return
func2(c=4, a=2, b=2)  #So, ALSO lets us do CALCULATIONS
#As with normal Keyword Arguments - ORDER DOES NOT MATTER!
#BUT....In such a case, would PROBABLY be SIMPLER just doing WITHIN Function!


#%%     Python Code Challenges 1
def divisible_by_ten(num):
    if num % 10 == 0:
        return True
    else:
        return False
  
number = divisible_by_ten(450)
print(number)
list_example = [2,4,7,8,6,4,5,7,6,8,2]
def append_sum(lst):
    for loop in range(3):
        lst.append(lst[-1]+ lst[-2])
    return lst

def more_than_n(lst, item, n):
    if lst.count(item) > n:
        return True
    else:
        return False
print(more_than_n([2,4,6,2,3,2,2,2,1,2], 2, 3))

def remove_middle(lst, start, end):
    return lst[:start] + lst[end+1:]     #Makes sense! EASY!

print(remove_middle(list_example, 2, 7))

def double_index(lst,index):
    if index > len(lst):
        return lst
    else:
        new_lst = []
        new_lst = new_lst + lst[:index]
        new_lst.append(lst[index] * 2)
        new_lst = new_lst + lst[index+1:]
        return new_lst
print(double_index(list_example, 4))

example_list = [2,4,6,4,7,3]
def middle_element(lst):
    lst.sort()
    if len(lst) % 2 == 0:
        return (lst[int(len(lst)/2)] + lst[int(len(lst)/2) - 1])/2                                  
    else:
        return lst[int(len(lst)/2)]
    
print(middle_element(example_list))

def odd_nums(lst):
    new_list = [ ]
    for item in lst:
        if item % 2 != 0:
            new_list.append(item)
    return new_list   

print(odd_nums(example_list))


def delete_starting_evens(lst):     #First elements deleted IF Even
    index = 0                        
    length = len(lst)
    while index<length and lst[0] % 2 == 0:
          lst.pop(0)     #OR, could use .remove(lst[0]) 
          index += 1
    return lst
list_2 = [2,4,6,4,7,3]

def odd_indices(lst):
    new_lst = [ ]
    for index in range(len(lst)):
        if index % 2 != 0:
            new_lst.append(index)
        else:
            new_lst = new_lst
    return new_lst

list_example2 = [4,5,6,7,8,3,4,5,6,7]
print(odd_indices(list_example2))

def exponents(bases, powers):
    new_lst = []
    for index in range(len(bases)):
        new_lst.append(bases[index]**powers[index])
    return new_lst

list_exponents = exponents([2,3,4], [2,2,2])
print(list_exponents)   #Each RESPECTIVE base and power BY INDEX are paired.

def larger_sum(lst1, lst2):
    sum1 = 0
    sum2 = 0                 #VERY EASY
    for num1 in lst1:
        sum1 += num1
    for num2 in lst2:
        sum2 += num2
    if sum1 > sum2:
        return lst1
    else:
        return lst2
print(larger_sum([2,3,4], [2,2,2]))

def reversed_list(lst1, lst2):          #Just need to VISUALISE REVERSE Lists
    for index in range(len(lst1)):
        if lst1[index] != lst2[len(lst2) - 1 - index]:
            return False
        else:
            return True
print(reversed_list([1,2,3,1], [1,1,1,2]))

#%%              STRINGS    
#can access LIKE ELEMENTS in list, individual characters in a string
#AT AN INDEX:
una_fruta = "la sandia"
print(una_fruta[4])

print(una_fruta[3:-2])    #just SLICE 'sand'
print(una_fruta[3:])  
    
def account_generator(first_name, last_name):
    account_name = first_name[:3] + last_name[:3]
    return account_name
print(account_generator("Martin", "Scorsese"))
print(len(una_fruta))

favourite_fruit = "strawberry"           #JUST MORE PRACTICE
last_characters = favourite_fruit[:-5]    #'straw'
print(last_characters)

BCS_quote = "He said, \"Perfection is the ENEMY of Perfectly Adequate\"."
print(BCS_quote)           #Escape Characters (so " " can be added INSIDE STRINGS)
# '\' is ALSO used in LINE CONTINUATION:
beatles = "The long and winding road,\
 That leads to your door,\
 will never disappear,\
 I've seen that road before',\
 It always leads me here,\
 Lead me to your door"
print(beatles)   #Another way to do Multi-Line!


# Use 'in' - CHECK if "sub-string" is 'in' a String Variable
print("sand" in una_fruta)     #returns True or False 

def common_letters(string_one, string_two):
    common = [ ]
    for letter in string_one:
        if letter in string_one and letter in string_two and letter not in common:
            common.append(letter)
    return common
print(common_letters(favourite_fruit, una_fruta)) 
#Note: Can use 'if in' for DICTIONARIES (check for Keys) AND Lists (check for list element IF within List) TOO!


def username(first, second):
    username = first[:3] + second[:4]
    if len(first)<3 or len(second)<4:
        return first + second
    else:
        return username
    
username_1 = username("Martin", "Scorsese")
print(username_1)
def password_generator(username):
    password = " "
    for index in range(len(username)):
        password += username[index]
    return password
        
print(password_generator(username_1))
   
#%%     STRING METHODS

fav_soundtrack = "cinema paradiso"
soundtrack_title = fav_soundtrack.title()   #.title() = TITLE FORMAT (first letters of Each Word are CAPITALIZED)

soundtrack_words = fav_soundtrack.split(" ")    #.split() = SPLIT at specified 'CHARACTER' INTO a 'LIST!

print(soundtrack_title.split('a'))         
                                           #           OR at specified 'DELIMITER' (below)
directors_names = 'Hayao Miyazaki, Isao Takahata, Takeshi Kitano, Yasujiro Ozu, Christopher Nolan, Stanley Kubrick'
directors_list = directors_names.split(", ")
print(directors_list)   # = ['Hayao Miyazaki', 'Isao Takahata', .... ]

director_surnames = [ ]
director_firstnames = [ ]
for name in directors_list:
    first_and_last = name.split(' ')              
    director_surnames.append(first_and_last[-1])
    director_firstnames.append(first_and_last[0])     
print(director_surnames)
print(director_firstnames)

poems_list = ['Afterimages: Audre Lorde: 1997', 'The Shadow: William Carlos Williams: 1915', 'The Road Not Taken: Robert Frost: 1916', 'The Raven: Edgar Allan Poe: 1845']
poems_highlighted = [ ]
for poem in poems_list:
    poems_highlighted.append(poem.split(":"))   #appended Lists into '2D list'!
print(poems_highlighted)   
titles = []
poets = []
dates = []
for section in poems_highlighted:
    titles.append(section[0])
    poets.append(section[1])
    dates.append(section[2])
print(titles)                
print(poets)
print(dates)        #REALLY SIMPLE! Just separated EACH into OWN LIST!


damien_rice = \
'''And so it is just like you said it would be
Life goes easy on me
Most of the time
And so it is the shorter story
No love, no glory
No hero in her sky'''

verse_lines = damien_rice.split('\n')  # '\n' - Splits AT 'LINE BREAKS' into List!
print(verse_lines) 

#OR can do REVERSE - 'JOIN' a List,  'BY Delimiter' INTO a 'STRING':
print("There are countless phenominal and genuinely masterful directors: " + ', '.join(director_surnames)) 
#Note: when JOINING, can join by ANY DELIMITER!
items = ['Iron Sword', 'Stick', 'Bow', 'Torch', 'Axe', 'Pickaxe', 'Mace', 'Dagger', 'Longsword']
print(f"Choose your Weapon: {', '.join(items)}")   #(works in f-string too)



featuring = "        Mark Knopfler       !!!!!!!"
stripped = featuring.strip('!').strip()      #can STRIP SEVERAL things at once!
print(stripped)                              #Just like 'TRIM()' for R and SQL - removes Empty WhiteSpace!
stripped_lines = []
lines = ['Always     ', '          in the middle of our bloodiest battles                  ', 'you lay down your arms','                  like flowering mines', '\n' , '        to conquer me home.                      ']
for line in lines:
    stripped_lines.append(line.strip())    #creates NEW 'STRIPPED' List 

damien_rice = damien_rice.replace("No", "Without any")    #.replace() EACH Occurance of ONE String with ANOTHER String 
damien_rice = damien_rice.replace("no", "without")
damien_rice = damien_rice.replace("sky", "story")
damien_rice = damien_rice.replace("story", "narrative")
print(damien_rice)

#Can 'find FIRST INDEX' VALUE where a String WITHIN in Located!
print(damien_rice.find("Life"))     #.find() method  -  here, "Life" is at Index 44 (FIRST OCCURENCE)

def song_string(soundtrack, artist):
    return F"My Favourite Film Soundtrack is '{soundtrack}' by {artist}"
print(song_string("Cinema Paradiso", "Enio Moricone"))


transport_count = [ ]
transport = ['Bus', 'Train', 'Walk', 'Cycle', 'Car', 'Train&Walk&Cycle', 'Car&Train', 'Walk&Bus', 'Walk', 'Run', 'Scooter', 'Skateboard&Walk' ]
for means in transport:
    if '&' in means:
      diff_means = means.split('&')     #Creates list 'diff_means'of SPLIT means of transport
      for mean in diff_means:
        transport_count.append(mean)       #ADDS each individual element in 'diff_means' TO transport_count
    elif not '&' in means:
      transport_count.append(means)         
print(len(transport_count))         #Just to check ALL 17 have been added!
 

Bus = transport_count.count("Bus")              #COUNT TOTAL Occurances of EACH STRING
Train = transport_count.count("Train")
Cycle = transport_count.count('Cycle')
Car = transport_count.count('Car')
Walk = transport_count.count('Walk')
Skateboard = transport_count.count('Skateboard')
Scooter = transport_count.count('Scooter')
Run = transport_count.count('Run')

list_count = [Bus, Train, Cycle, Car, Walk, Skateboard, Scooter, Run]
transport_each = ['Bus', 'Train', 'Cycle', 'Car', 'Walk', 'Skateboard', 'Scooter', 'Run']

def count_statement(lst1,lst2):
    for index in range(len(lst1)):
       print(f"For {lst2[index]}, there is a count of {lst1[index]}")
  
count_statement(list_count, transport_each)

#Shows that 'f strings' are BEST WAY to write Strings with Variables within:
fstring = f"These are Common Means of Transportation:\n {transport_each} \nThey are all great in their own right!"
print(fstring)    #EASIER than above way IMO!

#%%     MODULES in Python
# Modules = ALSO called 'Libraries or Packages' (means SAME THING!)
from datetime import datetime
import random
from matplotlib import pyplot as plt   #Example of 'ALIASING' - changing Namespace 
from decimal import Decimal 
current_time = datetime.now()    #.now() gives you CURRENT DATE-TIME (Always Changing!)
print(current_time)

Date_of_Birth = datetime(1989,8,23)    #for SPECIFIC Dates
print(Date_of_Birth)           #Will print the date of birth

print(Date_of_Birth.year)    #OR can access INDIVIDUALLY
print(Date_of_Birth.month)
print(Date_of_Birth.weekday())   #EVEN get to access WEEKDAY 
                                #0-6:  0=Monday, 1=Tuesday.....
print(datetime(2018,1,1) - datetime(2017,1,1))  #Even can SUBTRACT Dates!

print(datetime.now() - datetime(2020, 3, 23))


#PARSING/Converting STRING of a Date e.g. 'Jan 15 2018' TO Datetime
#Done using 'datetime.strptime(string, changed format)' Function:
string_to_date = datetime.strptime("Mar 13 2008", "%b %d %Y")  #2nd argument = EMMULATES 'FORMAT' of the String
# %b = ABBREVIATED MONTH (Jan, Feb, Mar.....)
# %d = date of month
# %Y = year
print(string_to_date)   #2008-03-13  00:00:00

#OPPOSITE (Datetime to STRING) '.strftime( )'
date_to_string = datetime.strftime(datetime(2008, 3, 13), "%b/%d/%Y")
print(date_to_string)     #prints string 'Mar/13/2008' - could be ',' or '/' AS LONG as % string EMULATES ITS FORMAT!

             #RANDOM Library/Module:
random_list = [random.randint(100,300) for num in range(101)]
print(random_list)    #Generates a list of 100 RANDOM Integer Numbers
random.randint(1,10)

random_choice = random.choice(random_list)
print(random_choice)   #KEEPS CHANGING RANDOMLY - RANDOMLY PICKED! 

sample = random.sample(random_list, 4)
print(sample)     #Randomly Selects a sample FROM the List of 4 (specified) elements

#USING MATPLOTLIB to PLOT SIMPLY:
nums_a = range(1,21)
nums_b = random.sample(range(1001), 20)  #random sample of 20 from list of 1000
plt.plot(nums_a, nums_b)
plt.show()             
#NOTE: the x-axis RANGE SIZE (num_a) MUST MATCH the SAMPLE SIZE! Otherwise WONT WORK!!!
#Makes sense, because refer to SAME THING - x-axis !!! 
#%%    ACCESSING FILES AS MODULES
#FILES are essentially = MODULES! So can IMPORT as you would Libraries!
# Use Wildcard = * ,just to import EVERYTHING IN that Module!

#Or just 'Import Library' if we want ENTIRE File Contents!
import Module_Import
practice = Successful_import()
print(practice)
print(works())

#Must 'SET' APPROPRIATE 'WORKING DIRECTORY' (i.e. be in 'Correct Folder' in Files Tab, RHS)  
#Otherwise WILL give ERROR - ModuleNotFound
#If making any changes to imported file, must SAVE BEFORE importing again! Otherwise will return 'Error'.


#Search up 'Python Module Index' (for your specific Version of Python) to view a HUGE Collection of Modules/Libraries we can use!
#Most Likely that someone ALREADY HAS CREATED the Functionality we are looking for - A lot of these External Modules are ALREADY STORED WIHTIN the Downloaded Python of our Computer!
#On the Website, are TOLD the 'SOURCE CODE' for these EXTERNAL Modules. e.g. 'base64': Lib/base64.py
#Also have BUILT-IN Modules, where we dont have to Load in - ALREADY BUILT INTO PYTHON Language!

#What about the External Modules NOT stored in python download file? i.e. written by OTHER DEVELOPERS - 'THIRD PARTY' External Modules? 
#Take 'Python-docx' external module for Example - Just GOOGLE and will be TOLD INSTRUCTIONS for INSTALLATION
#'pip install python-docx' - PIP = ' PACKAGE MANAGER' (included already in Python 3)
#Must DOWNLOAD this 'Python-docx' typing the installation code above using 'COMMAND PROMPT'
#(Side Note: to check we have pip installed, do 'pip --version' in Command Prompt)
#Normally these 3rd Party External Modules will be STORED WTIHIN the 'Site Packages' Folder of the 'Lib' Folder of Anaconda Folder
#'pip uninstall python-docx' to UNINSTALL Module



#%%       DICTIONARIES
# 'Key:Value pairs':  Key = ONLY 'String OR Number', Value = ANYTHING (even another dictionary!)
star_wars_characters = {"The Mandalorian": ["Mando", "Grogu", "Bo Katan", "Grief Karga", "The Armourer", "Luke Skywalker!"], 
                        "A New Hope": ["Luke Skywalker", "Obi Wan Kenobi", "Princess Leia", "Darth Vader", "Han Solo", "C3PO", "R2-D2"], 
                        "Revenge of the Sith": ["Obi Wan Kenobi", "Anakin Skywalker", "Emperor Palpatine", "Mace Windu", "General Grievous", "Padme Amadala"]}
print(star_wars_characters)
print(len(star_wars_characters))      #like Lists, can use LEN to find 'HOW MANY Key:Value pairs' we HAVE!
individual_character = {'Film appearances': 6, "Surname": "Skywalker", "Family": ["Uncle Owen", "Aunt Beru", "Leia", "Anakin", "Han (Brother-in-law)", "Padme"]}
#can mix and match!

individual_character["Ordinary, Jedi or Sith"] = "Jedi, like my father before me"
#ADDING ONE KEY:VALUE Pair

#Using '.update()' method to 'ADD ON MORE' ENTRIES TO the Dicitonary
individual_character.update({"Age": ["Teen", "Twenties", "Thirties", "Sad Semi-Senile Old Fart..."], "Colour Lightsaber": "Green", "Significant Other": "Unknown in Canon"})
#(essentially are ADDING ANOTHER DICTIONARY to an EXISTING ONE)

individual_character["Film appearances"] = 3    #'OVERWRITING' Existing Values,
individual_character["Film appearances"] += 3
print(individual_character)       #then ADDING to a 'Number' Value (+=)

# DICT COMPREHENSION (Very Useful!):
names = ["Aragorn", "Frodo", "Sam", "Boromir", "Legolas", "Gimli"]
actors = ["Viggo", "Elijah", "Sean A", "Sean B", "Orlando", "John"]
list(zip(names, actors))  # =nested 2D list with TUPLES 
combine_into_dict = {name:actor for name, actor in zip(names, actors)}
print(combine_into_dict)    #'DICT COMPREHENSION', used to 'COMBINE LIST(s)' INTO a DICTIONARY, where EACH respective elements form 'KEY:VALUE' Pairs
#Another Dict Comprehension Example:
lengths = {name:len(name) for name in names} #just dictionary of 'name' in names list WITH the String LENGTH of EACH name - SUPER EASY!!! 


dict_dict = {"Individual": individual_character, "Best Star Wars Character": "Luke Skywalker"}
print(dict_dict)        #example of a Directory 'value' IN a Directory -since 'value' can be ANYTHING!


#%%   USING Dictionaries:
actor_height_m = {"Tom Hardy": 1.75, "Tom Cruise": 1.7, "Brad Pitt": 1.8, "Danny Devito": 1.47, "Joe Pesci": 1.63, "Arnold Schwarzenegger":1.85, "Morgan_Freeman": 1.88, "Samuel L. Jackson": 1.89, "Tim Robbins": 1.96 }
print(actor_height_m["Tom Cruise"])     #(ccess VALUES of Key-Value pairs, using 'KEY' as an [INDEX] (LIKE for LISTS)

#(METHOD 1 to CHECK FOR KEY)
check_key = "Christian Bale"
if check_key in actor_height_m:   #Checks IF a Potential 'Key' is 'IN' a Dictionary
    print("This key exists!")
else:
    print("That key is not here!")

#(METHOD 2 to CHECK FOR KEY)   #TRY/EXCEPT - USEFUL!
try:
    print(actor_height_m["Christian Bale"])    #returns ASSOCIATED VALUE IF Key Exists!
except KeyError:         #Good to Specify TYPE of ERROR encountered!
    print("This key has not been added to this Dictionary!")
    
#METHOD 3 - using '.get()' = BEST WAY to GET 'Associated VALUE' in a Key:Value Pair!
      #  .get("Key", "If not existing")   -   (IF a KEY EXISTS, will RETURN Associated 'VALUE'!)
print(actor_height_m.get("Christian Bale", "This Actor's Height has not been added yet."))     
#e.g.2 - MORE PRACTICE:
actors_ages = {"Danny Trejo":78, "Oscar Isaac": 44, "Pedro Pascal": 48, "Wagner Moura":46, "Antonio Bandera":62, "Javier Bardem":54, "Benicio Del Toro":56, "Giancarlo Esposito":64}
print(actors_ages.get("Bryan Cranston", "Age Not Added"))

print(actors_ages.pop("Oscar Isaac", "Actor not in List anyway!"))
#Use .pop() to REMOVE a Key:Value Pair - by providing Key AND (like .get()) can specify what to Return IF the Value is NOT in the Dicitonary!
my_age = 23        #Just extra fun stuff! Subtracting a POPPED age from existing one!
my_age -= actors_ages.pop("Danny Trejo")
print(abs(my_age))         #Danny Trejo is 55 years older than me!!!

#Getting 'ALL KEYS':
print(list(actor_height_m))   #saves as LIST of KEYS, OR could use .keys() method
print(list(actor_height_m.keys()))      #BUT .keys() CANNOT BE MODIFIED - is JUST a VIEW of the KEYS, so must CONVERT to LIST using list()
for actor in actor_height_m.keys():
    print(actor)         #ALSO can use .keys() in 'LOOPS', so EACH Element (Key) can be Accessed INDIVIDUALLY!
 
#Getting 'ALL VALUES':
print(list(actor_height_m.values()))  #.values() Method used - SIMPLE!
sum_heights = 0
for height in actor_height_m.values():
    sum_heights += height
average_height = sum_heights/len(actor_height_m)   #COOL! So could use to find Average Height, which is 1.77m!
sum_heights

#Getting 'ALL ITEMS' (Keys AND Values):   '.items()' 
fav_series = {"Breaking Bad": 2008, "Better Call Saul": 2015, "Midnight Mass": 2021, "Invincible":2021, "Last Airbender":2005, "Merlin": 2008,"Star Wars: Clone Wars": 2003, "Star Wars: The Clone Wars":2008, "Rebels":2014, "Andor":2022, "Twin Peaks":1990, "The Office":2005, "Doctor Who (Revival)":2005, "Broadchurch": 2013, "Sherlock":2010, "Seinfeld":1989, "Daredevil":2015}
print(list(fav_series.items()))    #Gives 'dict_list' where EACH key:value pair is a 'TUPLE'!
for tvseries, date in fav_series.items():         #Access EACH PAIR Individually 
    print(f'{tvseries} was released in {date}')  # 'f-string' BETTER than CONCATENATING!
  #(lets us INDIVIDUALLY access EACH 'key' and EACH 'value', GIVEN as 'TUPLES' together, so ALL can be ACCESSED!!!)
 
#SUMMARY Examples:
letter_to_points = { 'A':1, 'B':3, 'C':3, 'D':2, 'E':1, 'F':4, 'G':2, 'H':4, 'I':1, 'J':8, 'K':5, 'L':1, 'M':3, 'N':4, 'O':1, 'P':3, 'Q':10, 'R':1, 'S':1, 'T':1, 'U':1, 'V':4, 'W':4, 'X': 8,  'Y': 4, 'Z': 10 }
def word_points(word):  #takes in WORD String
    total_points = 0
    for letter in word:    #ADDS Associated 'Value' IF PRESENT
        total_points += letter_to_points.get(letter, 0)   #i.e. if NOT a Letter, will add 0
    return total_points
print(word_points("OBI-WAN KENOBI"))

characters = {"Jedi": ["OBI-WAN KENOBI", "MACE WINDU", "ANAKIN SKYWALKER", "LUKE SKYWALKER", "YODA", "PLO KOON", "TANO"], "Sith": ["DARTH MAUL", "DARTH SIDIOUS", "COUNT DOOKU", "DARTH VADER", "GRAND INQUISITOR", "KYLO REN"], "Other": ["LEIA", "HAN SOLO", "CHEWBACCA", "C3P0", "R2-D2", "DIN DIJARIN", "GRIEF KARGA"]}
each_category_points = {}
for category, characters in characters.items():    
    letter_points = 0
    for character in characters:                 #Iterate INTO Dictionary Value's LIST ELEMENTS 
        letter_points += word_points(character)     #adding up points for EACH character in the Lists
    each_category_points[category] = letter_points   
print(each_category_points)    #{'Jedi':150, 'Sith':119, 'Other':93}

    

#%%  FILES 
#File = simply way to STORE and RETRIEVE DATA on Computers.
with open('Book1.txt', 'r') as reading_file:            #'r' = READ-MODE
    reading = reading_file.read()
print(reading)     #COOL! It actually READS OUT the TEXT file Contents AS A WHOLE!
#NOTE: to read file, the text file MUST be IN THE SAME FOLDER as this PYHTON File!
#Side Note: to CREATE TEXT file (.txt), can be done for Word OR Excel file in SAVE AS...

#Can use '.readlines()' and can ITERATE EACH LINE 
# '.readlines()' = 'LIST', where 'EACH LINE' STORED as an 'ELEMENT in a LIST':
with open('The Raven.txt') as Raven:
    for line in Raven.readlines():   #iterating through EACH 'ELEMENT/LINE' in the List
        print(line + "\n")        
#Or can INDIVIDUALLY Print A 'SPECIFIC LINE', using ',readline()':
with open('Sweeny Todd Review.txt') as Todd:
    line_1 = Todd.readline()
    line_2 = Todd.readline()
    line_4 = Todd.readline()
    print(type(line_4))        #stores the Lines as 'STRINGS'!
    if 'Tim Burton' in line_4:             #DO WHAT WE WANT with the Line!
        print("This is a Tim Burton Film!")  #e.g. Here, CHECKED for a Specific SUBSTRING within the Line! - SIMPLE!
print(line_2)

#Adding 'NUMBER ARGUMENT' to '.read()' - Can SPECIFY 'NUMBER of CHARACERS' to RUN for EACH LINE
with open("Sweeny Todd Review.txt") as Todd:
    characters = Todd.read(20)     #reads first 20
    print(characters)
    characters2 = Todd.read(20)    #reads NEXT 20
    print(characters2)
    

#We can 'WRITE/CREATE' a FILE IN Python ALSO using 'with open()' in 'w' = WRITE MODE:
with open('happiness.txt', 'w') as happy:
    happy.write('''It might seem crazy what I am 'bout to say
                Sunshine she's here, you can take a break
            I'm a hot air baloon that could go to space
            with the air, like I don't care, baby by the way'
''')   #MULTI LINE File of Text
#NOTE: If we already have a file called 'happiness.txt' will COMPLETELY OVERWRITE IT!!!
#(Can Read it, as we know, just to check):
with open('happiness.txt', 'r') as happiness:
    for line in happiness:      
        print(line)            #print EACH Line going down!
        
#Writitng a File USING a LIST of LINES:
lines = ["Here is Line 1", "Here is Line 2", "Here is Line 3"]    
with open('examplefile.txt', 'w') as File:
    for line in lines:
        File.write(line + '\n')      #Writes EACH List ELEMENT (above) as a NEW LINE in 'examplefile.txt'
    
#Can ADD A LINE TO A FILE using APPEND MODE - 'a':
new_line = "This is an Appended Line!"    
with open('examplefile.txt', 'a') as File:
    File.write(new_line + '\n' )
with open('examplefile.txt', 'r') as File:  #Just Checking it was added!
    reading = File.read()
    print(reading)
    
with open('Sweeny Todd Review.txt', 'a') as Todd:     #(More Practice)
    Todd.write('I agree 100% with all that Ebert says in this Review. This is up there with Tim Burton\'s best!')
#Side Note: 'with' is called a CONTEXT MANAGER
#This essentially OPENS the FILE for us and THEN CLOSES IT AFTERWARDS
#PREVENTS Negative impact on OTHER Programs IF it stays open. Before, would have to MANUALLY call '.close())' at end!! 'with' makes this MUCH EASIER!

#      'COPY Contents of ONE FILE' to 'ANOTHER':
# - Just 'READ' to 'Open File' we want to COPY (SOURCE File), then do NESTED 'WRITE' to 'Open ANOTHER' File (DESTINATION File).
# - COPY contents FROM SOURCE file INTO DESTINATION file:
# with open('source.txt', 'r') as source_file:
#      with open('destination.txt', 'w') as destination_file:
#           for line in source_file:
#                destination_file.write(line)
#

# OTHER 'BETTER' MODES: -  'r+' = READ AND WRITE (all in one!), 'w+' = Reading AND Writing (TRUNCATES the File), a+ = (APPENDING and READING, creating NEW File) 
#   (these are BETTER! More VERSATILE!)

# '.tell()' = CURRENT POSITION (in bytes) in the file
# '.seek(offset, from)' = CHANGES POSITION by 'offset' bytes ('from' can be 0 = begining, 1=relative to current position, 2 = end)
#  e.g. 'writefile.seek(0)' - goes to BEGINNING of that File!


#      'CSV FILES' = just a TEXT file with a PARTICULAR Data STRUCTURE - accessed EXACT 'SAME WAY'!
#(From Excel and Google Sheets can export using THIS format)
with open('video_games_CSV.csv') as games:
    comma_separated = games.read()
print(comma_separated)      #prints it IN the CSV format   

#BETTER IF we 'CONVERT EACH ROW' to an INDIVIDUAL 'DICTIONARY' using 'csv.DictReader()', where KEY = FIRST ROW Elements (i.e. HEADERS) and VALUES are ASSOCIATED ROW ENTRIES 
#Just IMPORT 'csv' LIBRARY/Module:
import csv
list_of_games = [ ]
with open('video_games_CSV.csv', newline='') as games:
    csv_dict = csv.DictReader(games)                    
    for row in csv_dict:
        list_of_games.append(row['Popular Game'])  
print(csv_dict)        #just a 'DictReader' OBJECT, so USELESS here!
print(list_of_games)   #ACCESSES 'Popular Game' Values for EACH ROW
#ITERATES THROUGH ROWS and EXTRACTS the GAMES FROM the Dictionary CSV (into a List of them)- could do so with OTHER Columns too!
#Note: 'newline' used so Line Breaks are NOT Mistaken for NEW ROWS!

#EXAMPLE 2 (SAME - JUST Practice):
fruit_list = []             #STORE the ROW DICTIONARIES as List Elements!
with open('Fruit_CSV.csv', newline='') as Fruit:
    fruit_dict = csv.DictReader(Fruit)
    for row in fruit_dict:
        fruit_list.append(row)   #So EACH ROW is stored as a DICTIONARY, where KEY = Relevant HEADERS (First row of CSV) and Values = associated Values
print(fruit_list)

#The CSV 'DELIMITER' DOESNT HAVE TO BE A 'COMMA'! 
#E.g. Could use a Semi-Colon, Pipe (|) or DASH...just 'SPECIFY as ARGUMENT' in DictReader:
with open('Box Office Numbers.csv', newline='') as movies:
    csv_movies = csv.DictReader(movies, delimiter = '|')
    for row in csv_movies:         #here, chose '|' delimiter!
        print(row)       
#IMPORTANT NOTE: TO CHANGE DELIMITER IN the Excel CSV File, FIRST go to: Region Settings - Related Settings - Region - Additional Settings - 'List Separator' can be Changed to WHATEVER Delimiter we want!
#Note: 'Save AS' CSV in Excel will STILL say 'comma delimited' but will ACTUALLY seperate by NEW Delimiter now! Thankfully ALL PREVIOUS CSV files which were COMMA Seperated will be UNCHANGED - ONLY affects current/future ones which are saved with DIFFERENT Delimiter!

#WRITING CSV File: 
#'csv,DictWriter' lets us CONVERT a 'LIST' of these 'ROW DICTIONARIES' INTO a CSV FILE:
print(fruit_list)                               #just to view List '[{'Fruit':'Apple', 'Type':'Jonalgold', 'Amount':'76'}, {.....}]
with open('Fruits_Written_CSV.csv', 'w') as output_csv:
    fields = ['Fruit', 'Type', 'Amount']    #These are the 'HEADERS' (KEYS in Dictionaries)!
    output = csv.DictWriter(output_csv, fieldnames=fields)
    output.writeheader()
    for dictionary in fruit_list:      #Access EACH dictionary IN the LIST!
        output.writerow(dictionary)
#COOL! So, First - GAVE the CSV a NAME 'games_CSV.csv', 
#THEN Identified 'FIELDS' = HEADERS. CREATED CSV File using DictWriter and WROTE the HEADERS with .writeheader().
#LASTLY Created the ROWS of CSV FROM the Dictionary Values with 'output.writerow(dictionary)'  

#JSON FILES
# = 'JavaScript Object Notation' (note: Not all JSON files are JavaScript and Not All JavaScript is valid JSON)
#FORMAT SUPER SIMILAR to DICITONARY. Using 'json' Library can CONVERT from JSON TO DICTIONARY EASILY!
import json

with open('JSON_practice.json') as practice:
    json_example = json.load(practice)
    for element in json_example:
        print(element)
print(json_example[0]['name'])   #EASY! Extracts the 'name' value for the FIRST Dictionary in this JSON List

#WRITING a JSON FILE in PYTHON - i.e. from DICTIONARY to JSON file:
dict_to_json = {"Tom Hardy": 1.75, "Tom Cruise": 1.7, "Brad Pitt": 1.8, "Danny Devito": 1.47, "Joe Pesci": 1.63, "Arnold Schwarzenegger":1.85, "Morgan_Freeman": 1.88, "Samuel L. Jackson": 1.89, "Tim Robbins": 1.96 }
with open('heights.json', 'w') as heights:
    json.dump(dict_to_json, heights)     #Argument 1 = DICTIONARY to be Converted, Argument 2 = JSON FILE
#Now have created a JSON file called 'heights'! 'DUMPED' it into the cell!

#%%Cheeky, SUPER EASY Summary Example of Files:
compromised_users = ['jean49', 'haydenashley', 'michaelastephens', 'denisephillips', 'andrew24', 'kaylaabbott', 'tmartinez', 'mholden', 'randygilbert', 'watsonlouis', 'mdavis', 'patrickprice', 'kgriffith', 'hannasarah', 'xaviermartin', 'hrodriguez', 'erodriguez', 'danielleclark', 'timothy26', 'elizabeth19']
with open('compromised_users.txt', 'w') as text_users:
    for user in compromised_users:
        text_users.write(user + "\n")    #names going down - COOL!

import json
with open('message_for_users.json', 'w') as message:
    json.dump({'Recipients':'All Users', 'Message':'All will be successfully retrieved. Dont worry!'}, message)

with open('Message_to_Jesse.csv', 'w') as Jesse:
    Jesse.write("""
I'm
In
Your
Out...""")     #csv.DictWriter would be used IF we want a sort of TABULAR formatting, seperated by Delimiters.
with open('message_to_Jesse.csv', 'r') as Jesse:
    message = Jesse.read()
    print(message)
                                               

#%%    CLASSES

# STORES DATA 'TYPES' e.g. Data stored in 'List' is Different to 'Dictionary', 'float' is Different from 'integers'....
string = "this is a string, so has a CLASS of 'str' "
print(type(string))    # 'type()' tells us CLASS: <class 'str'> 
print(type(star_wars_characters))   #<class 'dict'>

#Class provides TEMPLATE for the Data Type - DESCRIBES Data Held WITHIN IT and HOW one can INTERACT with it!
class First_Python_Class:
    pass                   #'pass' used to CLOSE the Class.
#MUST 'INTSTANTIATE' the Class (Breathe Life into it - 'LIKE FUNCTION'!)
instance_of_class = First_Python_Class()   
#'INSTANCE' ALSO CALLED an 'OBJECT' - can REPRESENT the Class THROUGH the Object (OOP)
print(type(instance_of_class))    #<class '__main__.First_Python_Class;>'
#Note: 'main' just refers to the CURRENT FILE being Run, in which the Class is Defined.

            # 'CLASS' VARIABLE = Variable SAME for ALL OBJECTS of a Class
class video_game:
    title = "Star Wars: Jedi Survivor"    #First, DEFINE the Class Variable WITHIN the CLASS
game = video_game()  #Instantiate it  
print(game.title)    #Then ACCESS Class Variable using 'OBJECT.VARIABLE' pair
#So, ACCESSES the CLASS Variable and PRINTS IT for us!

          #METHODS = FUNCTION defined WITHIN CLASS:
class pounds_to_rupees:   
     conversion_rate = 353.62
     def convert(self):
        print(f"£1 is equivalent to {self.conversion_rate} Pakistani Rupees")
                                                                           
conversion = pounds_to_rupees()
conversion.convert()    #Should PRINT the Statement!
#Note: Common Practice to call the FIRST ARGUMENT 'self' (though could technically be called anything, THIS is the Naming Convention!)
#'self' = OBJECT, can be used as OBJECT.VARIBALE Pair (self.conversion_rate)
#MUST be included - lets us ACCESS Variables OUTSIDE of the Method (/Function) from WITHIN! 
#Here, our Method 'convert()' is accessed AS an 'OBJECT.VARIABLE' Pair (AS USUAL!)

 #Method can include MULTIPLE ARGUMENTS than just self: (like any old Function)
class convert_to_rupees:
    conversion_rate = 352.62
    def convert(self, pounds):
        return self.conversion_rate * pounds
converted = convert_to_rupees()
convert_300_pounds = converted.convert(300)
print(convert_300_pounds)         #105786 Rupees - NICE!
#Note: 'self' is AUTOMATICALLY Passed - so DONT NEED to PASS IT when Calling Method OUTSIDE of Class! 


         # 'CONSTRUCTORS' = called 'DUNDER' METHODS (Double Underscore) 
#'__init__(self)' is Constructor that INITIALIZES a NEW OBJECT
#i.e. EACH TIME we INSTANTIATE the Class, this FUNCTION is CALLED!
class Sing:
    def __init__(self, word):
        if type(word) == str:
            print(f"What you wont do, do for {word}")
lyric = Sing("Love")  #will PRINT the Statement WHEN we Instantiate!
#Note: See how Constructor 'ARGUMENT' is Specified 'WITHIN INSTANTIATION'! 
#Example 2 (More Practice):
class Circumference:
    def __init__(self, diameter):
        circumference = 3.14 * diameter
        print(circumference)
circle = Circumference(25)    #78.5 
 
#Objects/Instances can HOLD OTHER KINDS of Data/Variables, NOT JUST 'CLASS' Variables and Methods!

         # 'INSTANCE' VARIABLES = One SUCH TYPE of Data HELD BY 'OBJECTS'
class Somewhere:
    pass
Line1 = Somewhere()   #INSTANTIATED TWICE to create 2 Objects, AS USUAL!
Line2 = Somewhere()
Line1.lyrics = "There's a Place for us"      
Line2.lyrics = "Somewhere, a place for us"
#Above have CREATED these 'INSTANCE' Variables as OBJECT.VARIABLE Pairs 
string_lyrics = Line1.lyrics + "\n" + Line2.lyrics
print(string_lyrics)     #Now can do WHATEVER WE WANT with the Variables! E.g. Here, just CONCATENATED the String Variables TOGETHER!
#Note: the '.lyrics' = Instance/Object ATTRIBUTE, which can be ANY NAME WE WANT!    

#   BETTER if we CREATE Instance Variables 'WITHIN CONSTRUCTOR':
class Somewhere_within:
    def __init__(self, line):
        self.lineinside = line      #This is our INSTANCE VARIABLE!   
lyric1 = Somewhere_within("There's a time for us, Someday a time for us")
lyric2 = Somewhere_within("Time together and time to spare, Time to look, time to care")
print(lyric1.lineinside)            #BETTER than ABOVE Way, Though does SAME THING!
print(lyric2.lineinside)    #Accesses desired INSTANCE Variable value which we PROVIDED AS ARGUMENT for the Object! 

#EXAMPLE 2 (More PRACTICE)
class SearchURL:
    def __init__(self, url):
        self.urlvariable = url          
YouTube = SearchURL("www.YouTube.com")   #INSTANTIATED, as usual (Specifying URL as ARGUMENT of CONSTRUCTOR) 
Reddit = SearchURL("www.Reddit.com")
print(YouTube.urlvariable)    #PRINTS "www.YouTube.com", SPECIFYING Instance Variable '.url' (attribute)
print(Reddit.urlvariable)     #MUST be 'object.variable', Otherwise just prints '<__main__.SearchURL object at 0x000001DFC8A44550>'

class Secure_SearchURL:    #EASY! Just MODIFIED ABOVE to INCLUDE the Secure PREFIX 'https://'
    secure_prefix= "https://"
    def __init__(self,url):
        self.url = url
    def secure(self):               #ADD Prefix to URL using normal METHOD!
        return f"{self.secure_prefix}{self.url}"     #Always remember! - MUST use 'self.' when Referring to anything OUTSIDE OF EXISTING Method!
YouTube = Secure_SearchURL("www.YouTube.com")
Reddit = Secure_SearchURL("www.Reddit.com")
print(YouTube.secure())
print(Reddit.secure())       #simply accessing the 'secure()' function


#A DEFAULT String Representation is displayed when we PRINT OBJECT/INSTANCE AS IS:
print(YouTube)        #gives '<__main__.Secure_SearchURL object at 0x000001D225ABFBB0>'
#THIS is pretty USELESS TO US! - only says WHERE class is defined and Computer's Memory Adress

     #INSTEAD, use 'STRING REPRESENTATION METHOD':  '_repr_(self)'        (ANOTHER type of 'DUNDER' Method)
class final_boss:
    def __init__(self, name):
        self.name = name
    def __repr__(self):       #ONLY takes 'SELF' argument 
        return f"Master Kenobi has gone to Utapau to defeat {self.name}"
    
grevious = final_boss("General Grevious")
print(grevious)    #NOW, when we PRINT, will Print STRING defined in '__rep__', which INCLUDES 'General Grevious'
#Note: as shown above, doing 'object.variable' - 'grevious.name' lets us access SPECIFIC Variables ONLY.
#but, 'String Representation' lets us return a NICE LOOKING STRING WHEN we JUST Call the OBJECT/Instance!


#CLASS and INSTANCE 'Variables' are BOTH simply 'ATTRIBUTES OF OBJECT' (of Class)!
#Can 'CHECK IF' an 'ATTRIBUTE' for an Object 'EXISTS':
#Method 1 -  hasattr(object, "attribute")    gives 'TRUE' OR 'FALSE'
input(hasattr(Line1, "wrong_song"))    #can use print, just COOL to use INPUT here!
#Method 2 - getattr(object, "attribute", default)   gives us ACTUAL VALUE of Attribute and DEFAULT = Value to Return IF Attribute does NOT EXIST.
input(getattr(Line2, "lyrics_of_another_song", "Wrong Song!"))     

    #dir() function lets us VIEW 'ATTRIBUTES' of OBJECTS
print(dir(lyric1))
print(dir(star_wars_characters))   #EVEN works for DICTIONARIES...
  #OR Functions...
#Gives us a list of ALL Attributes - the '__  __' attributes are INTERNAL Attributes. INSTANCE VARIABLE 'lyric' is at END of list!
#Note: Pretty much EVERYTHING is an 'OBJECT'! - Lists, Functions, Integers, Strings...

              #%% CLASSES SUMMARY EXAMPLES and PROJECT

#Instance Variable in Constructor Example:
class Circle:
    pi = 3.14
    def __init__(self, diameter):                #Constructor Method
        print(f"Circle has Diameter of {diameter}")
        self.radius = diameter/2
    def area(self):                              #Normal Method/Function
        area = (self.radius **2) * self.pi
        return f"This Circle has an Area of {area}"
circle_area = Circle(20)     #Instantiating (calls Constructor!)
print(circle_area.area())   #SUPER EASY!

class Grade:
    min_pass_percentage = 86
    def __init__(self, score):
        self.score = score
    
class Student:
    def __init__(self, name, year):
        self.studentname = name
        self.specificyear = year
        self.achievedgrades = [ ]   #defined 3 instance variables here
    def add_grade(self, grade):     #takes IN a LIST of Grades
        if type(grade) == int:       #Will 'APPEND IF' it is an 'INTEGER NUMBER'
            self.achievedgrades.append(grade) 
        return self.achievedgrades              #ADD grades TO LIST Instance Variable!
    def __repr__(self):
        return f"This Student is {self.studentname} Weasley. He is currently in the {self.specificyear} at Hogwarts"

Ron_score_percentage = Grade(55)
print(Ron_score_percentage.score)
Ron = Student("Ron", "Final Year")
print(Ron)
print(Ron.add_grade(50))    #just trying different accesses!



       #CLASSES Summary PROJECT - Designing a MENU for a Restaurant
#First, create MENU Class (so can CLASSIFY the Different Menus at Different Times of Day!):
class Menu:
    def __init__(self, name, items, start_time, end_time):
        self.name = name
        self.items = items     #DICTIONARY, where Keys = ITEMS, Values = Price Associated
        self.start_time = start_time
        self.end_time = end_time
    def __repr__(self):
        return "{} Menu is available from {}-{}".format(self.name, self.start_time, self.end_time)
    def calculate_bill(self, items_purchased):      #=LIST of Items purchased by customer as an argument
        bill=0     #will ADD UP bill
        for item in items_purchased:
            if item in self.items:           #As we know, CHECKS for KEY WITHIN a Dicitonary!
                bill += self.items[item]   #ADDS VALUES TO Bill - EASY!
        return "Here is your Bill: {}".format(bill)
    
#Specify the 'Items' for the specific menu:
drinks = {'Sprite (500ml)': 1.85, 'Coke Zero (500ml)': 1.85, 'Mineral Water': 1.85, 'Salt Lassi': 4.65, 'Mango Lassi': 5.40, 'Passion Juice': 4.95, 'Fresh Orange Juice': 4.16, 'Fresh Carrot Juice': 4.40, 'Fresh Apple Juice': 4.15}
drinks_menu = Menu("Drinks", drinks, 1100, 2230 )      #'drinks' dictionary = 'ITEMS' Instance Variable     

#SAME for OTHER MENUS:
Non_Veg_Snacks= {'Kebab Roll': 6.50 , 'Large Kebab Roll': 8.30, '3 Kebabs (with Chips or Paratha)': 6.50, 'Chicken Kebab Roll': 8.30, 'Chicken Tikka (with Chips or Paratha)': 10.15, 'Lamb Tikka (with Chips or Paratha)': 12.00, 'Fried Fish Roll': 10.15}
NV_Snacks_menu = Menu("Non Veg Snacks", Non_Veg_Snacks, 1100, 2230)

Vegetable_Snacks = {'Chili Paneer': 11.10, 'Zai Aloo Tikki Roll': 5.50, 'Chana Masala Roll': 7.40, 'Paneer Tikka Roll': 9.25, 'Falafel Roll':7.40, 'Daal Bhajiya': 7.40, }
Vegetable_menu = Menu("Vegetable Snacks", Vegetable_Snacks, 11, 2230)

Kids = {'Chicken Nuggets Meal':6.50, 'Fish Finger Meal':6.50, 'Grilled Lamb Burger':8.50, 'Grilled Chicken Burger': 9.00}
Kids_menu = Menu("Kids", Kids, 1100, 2100)

Breakfast = {"Halwa Puri Chanay":11.50, "Anda Paratha & Chai":11.50, "Omelette & Paratha":8.50, "Chana Aloo":10.00, "Puri":1.20, "Halwa":5.00}
Breakfast_menu = Menu("Breakfast", Breakfast, 1000, 1300)
print(Breakfast_menu)  #should return statement in __repr__

Main_Courses = {'Quarter Chicken (with Chips and Drink)':8.30, 'Half Chicken (with Chips and Drink)':10.15, '3/4 Chicken (with Chips and Drink)':12.95, 'Full Chicken (with Chips and Drink)':15.70, 'Karahi Chicken':12.00, 'Chicken Tikka Masala':13.00, 'Saag Chicken':13.80, 'Chicken Korma':14.50, 'Butter Chicken': 14.80, 'Chicken Madras':13.80, 'Saag Meat': 15.00, 'Karahi Lamb':14.00, 'Bhindi Meat':15.50, 'Tandoori Chops Masala':18.50,'Lamb Madras': 15.70, 'Chicken Biryani': 13.80, 'Lamb Biryani':13.40, 'Fish Biryani':15.40, 'Vegetable Biryani':11.00, }
Main_Courses_menu = Menu('Main Courses', Main_Courses, 11, 2230)

Sides = {'Zeera Rice':7.50, 'Pilau Rice':7.00, 'Boiled Rice':5.00, 'Butter Naan':2.00, 'Tandoori Naan':1.50, 'Butter Rotti':2.00, 'Tandoori Roti':1.50}
Sides_menu = Menu('Sides', Sides, 1100, 2230)
#Now can PRACTICE USING the Class - Calling Methods and Instance Variables.
Main_order_1 = ['Butter Chicken', 'Vegetable Biryani']
print(Main_Courses_menu)      #gives us the String Representation
print(Main_Courses_menu.calculate_bill(Main_order_1))
print(drinks_menu.end_time)   #Can access a SPECIFIC INSTANCE VARIABLE Like so! EASY!

#Useful to PUT ALL "_menu Instances/Objects" INTO a LIST. Why? Allows us to STORE VALUES so can be REUSED in OTHER CLASSES (Below)!
menus = [drinks_menu, NV_Snacks_menu, Vegetable_menu, Kids_menu, Breakfast_menu, Main_Courses_menu, Sides_menu ]   
#SECOND, define FRANCHISES for DIFFERENT RESTAURANTS, Each given available menus and their address location:
class Franchise:
    def __init__(self, address, menus):    #menus is LIST of ALL MENUS (will Only select those which are RELEVANT to THIS Particular Restaurant!)
        self.address = address
        self.menus = menus
    def __repr__(self):
        return "This Franchise is Located on {}".format(self.address)
    def available_menus(self, time):
        available_menus = []
        for menu in self.menus:         #iterating, so 'menu' could be any of those OBJECTS in 'menus' List!
            if time >= menu.start_time and time <= menu.end_time:
                available_menus.append(menu)
        return available_menus          
#SIMPLY accessed INSTANCE VARIABLES - 'menu.start_time' and 'menu.end_time' which are RELEVANT to the SPECIFIC MENU/INSTANCE
#Done by menu.start_time', since 'menu' (element in list) = INSTANCE/Object for 'Menu' Class
#(so simply us access the TIMES from WITHIN the INSTANCE! e.g. 'Breakfast_menu = Menu("Breakfast", Breakfast, 1000, 1300)' 

main_restaurant = Franchise("Watford High Street", menus)
print(main_restaurant.available_menus(1200))  #so, will give us LIST of Menus available if you come AT 12:00!
print(main_restaurant.menus)   #Accessing Instance Variable as PRACTICE -Gives us the 'menus' list we made above. 
Harrow_restaurant = Franchise("Harrow on the Hill", menus)
print(Harrow_restaurant.address)     #just more practie calling Instance Variable!
Wembley_restaurant = Franchise('Wembley', menus)

#Say we NOW have DIFFERENT TYPES of Restaurants owned by our Business:
class Business:          #PUT EVERYTHING TOGETHER NICELY!
    def __init__(self, name, franchises):
        self.name = name
        self.franchises = franchises        
#Can See HOW Classes WORK TOGETHER Now - HIARARCHY of Sorts, from LOWEST to HIGHEST Ordering (DEPENDENT Classes!)
Taste_of_Lahore = Business("Taste of Lahore", [main_restaurant, Harrow_restaurant, Wembley_restaurant])        

#Say we Create ANOTHER CHAIN of Restaurants for MANGHAL: (SAME Process Again)
Mangal_Kebabs = {'Lamb Shish':9.00, 'Adana':10.00, 'Chicken Shish':9.00, 'Chicken Beyti':10.00, 'Iskender':12.00, 'Hallumi Kebab':8.50, 'Mixed Grill':12.99}
Kebab_menu = Menu('Kebabs', Mangal_Kebabs, 1000,2200)
Wraps_and_Burgers = {'Lamb Doner Wrap':9.99, 'Chicken Doner Wrap':9.99, 'Adana Wrap':9.99, 'Chicken Beyti Wrap':9.99, 'Hallumi Wrap':8.99, 'Beef Burger':6.49, 'Cheese Burger':6.49, 'Chicken Fillet Burger':6.50, 'King Burger':7.99}
Wraps_and_Burgers = Menu('Wraps & Burgers', Wraps_and_Burgers, 1000, 2200)
Mangal_Kids = {'Chicken Nuggets & Chips':7.99, 'Scampi & Chips':7.99, 'Small Donner Meal':7.99, 'Hamburger & Chips':7.99 }
Mangal_Kids_menu = Menu('Kids Menu', Mangal_Kids, 1000, 2200)

mangal_menus = [Kebab_menu, Wraps_and_Burgers, Mangal_Kids_menu]
Mangal_Rickmansworth = Franchise('Rickmansworth', mangal_menus)
Mangal_Chalfont = Franchise('Chalfont and Latimer', mangal_menus)

Mangal = Business('Mangal Express', [Mangal_Rickmansworth, Mangal_Chalfont])
#We just REPEATED the ABOVE for THIS Restaurant Chain 'Mangal'.

#Now can ACCESS INDIVIUAL COMPONENTS OF 'INSTANCE LISTS'!- REALLY COOL!!!!
print(Mangal.franchises[0].menus[0]) #accesses FIRST ELEMENTS of Franchise List (Rickmansworth) and FIRST MENU in the 'mangal menu' list (Kebabs)
#So accesses CLASSES WITHIN CLASSES - PRETTY SIMPLE!
print(Mangal.franchises[-1].menus[-1].items) #gives the DICTIONARY of KIDS Menu for Mangal!
#EXACT SAME can be done for Taste_of_Lahore!
print(Taste_of_Lahore.franchises[1].menus[1].items['Fried Fish Roll'])  #Even can access 'Value' from Specified Key - EASY!
#Can access WHATEVER WE WANT now!    


    
#%% OTHER EXAMPLES of USES of Try/Except
balance = 192.3
while True:      #checking is NUMBER BEFORE Depositing (so prevents Nasty-looking Error!)
    try:
        deposit = float(input('Enter Amount Here:'))
        break
    except ValueError:                       #BEST PRACTICE to SPECIFY the ERROR!
        print('Not a Number - Try Again!')
balance += deposit
print(f'Balance: {balance}')


number = int(input("Number: "))
print(number)       
#If we DONT ENTER a 'Number' will give ValueError - can't do int() for a STRING/words !
#This COMPLETELY STOPS our Program - ANNOYING!

#So? Create TRY-EXCEPT Block:   
try:
    answer = 9
    number = int(input("Try a Number: "))
    print(number)
except ZeroDivisionError as err:     #Note: can write SEVERAL 'except' Statements for OTHER ERRORS!
    print(err)
except ValueError:                   #can SPECIFY the TYPE OF ERROR we want to STOP - BETTER to be MORE SPECIFIC!
    print("Invalid Input")     
#TELLS us IF there is an ERROR - CATCHES the Error BEFORE it Annoyingly STOPS the Program!
#Note: can save Specific Errors TO VARIABLES, as shown above 'as err'
#Example 2
try:
    age = int(input('Age: '))
    income = 2000
    risk = income / age
    print(age)
except ZeroDivisionError:
    print("Age cannot be 0")
except ValueError:                
    print("Invalid Input")
    
#CLASS Example - AttributeError (ATTRIBUTE does NOT EXIST):
class NoAttribute:
    pass
no_attribute = NoAttribute()    #Instantiated it!

try:
    input(no_attribute.not_existing)   #Dont really need 'input()', but might as well!
except AttributeError:
    print("Need to Define the Instance Variable!")

#hasattr() or getattr() can be used ANYWHERE - NOT JUST for these Instance Variables!
random_elements = ["Danny Trejo", "Unknown", "Oscar Isaac",  44, "Pedro Pascal",  {"number": 48}, "Wagner Moura", 46, "Antonio Bandera", 62, "Javier Bardem", 54, "Gianarlo Esposito", 64]
for element in random_elements:
    if hasattr(element, 'count'):     #checking for EVERY ELEMENT IN the List
        print(f"Count Attribute is WITHIN this {type(element)} Class!")
    else:
        print(f"{type(element)} Class cannot be Counted!")
        

#Just checks elements in list to see IF these Objects CONTAIN the 'COUNT' Attribute!



#%%   Python INPUTS (Interacting with Console to execute code)
DOB = input("Enter your Data of Birth here:")
print("You were born on " + DOB + ".")       

#Can use INPUTS to get INFORMATION FROM the User.
#When Running code, INPUT must be answered FIRST - i.e. will have to INPUT user info INTO the CONSOLE and will STORE in python!
#OTHERWISE - works JUST like ANY OTHER Variable added to a print statement!

#IMPORTANT NOTE: ONLY PRESS 'Enter' ONCE for Inputs, OTHERWISE will KEEP restarting in Console for number of times you clicked enter! (annoying!)


#Can use as a 'Calculator' of sorts for Numbers. BUT REMEMBER to CONVERT to a FLOAT:
num1 = input("Enter a Number: ")
num2 = input("Enter another Number: ")
result = float(num1) + float(num2)       
print(result)

#More Practice with INPUTS:
color = input("Enter a color: ")
plural_noun = input("Enter a Plural Noun: ")
celebrity = input("Enter a Celebrity Name: ")

print("Roses are " + color)
print(plural_noun + " aren't blue")
print("I think " + celebrity + " is cool")

#%%Build Guessing Game (WHILE LOOP Practice):
lucky_numbers = [4,5,5,6,67,5]
friends = ['Bob', 'Joe', 'Marty', 'Frank', 'Ray']
friends.extend(lucky_numbers)
print(friends)     #'extend( ) is JUST ANOTHER WAY Besides List CONCATENATION to ADD LISTS Together!

#If we want to know WHAT INDEX a variable is at IN A LIST:
print(friends.index("Marty"))   #OUTPUT is 2 (AT index 2)
#   .copy() can be used to COPY A LIST, Just save as its own Copy Variable.


#While Loop with INPUT, so becomes GUESSING GAME - SUUUUUPER EASY):
secret_word = 'Lupin'
guess = ''
guess_count = 0           #while guess != secret_word and 
guess_limit = 3
out_of_guesses = False
while guess != secret_word and out_of_guesses == False:
    if guess_count < guess_limit:
      guess = input('Enter guess: ')                         
      guess_count += 1                 #input( ) lets us type Guesses INTO CONSOLE - INTERACTIVE! 
    elif guess_count >= guess_limit:
        out_of_guesses = True    #Makes sense!!!!
        print("Tu as perdu!")
    elif guess == secret_word:
        guess = input('Enter guess: ')
        print("Assane Diop")
        
        

        
    
        
#%%ALTERNATIVE WAY to do Guessing Game:
     #(This way is BETTER - uses less lines of code + Still easy to comprehend))
answer = "Allons-y!"
guess_count = 0
guess_limit = 3
while guess_count < guess_limit:
    guess = input("Guess the Doctor Who Catchphrase: ")
    guess_count += 1
    if guess != answer and guess_count >= guess_limit:
        print("Doctor...I let you go.")
    elif guess != answer:
        print("Who turned out the lights?")
    elif guess == answer:
        print("Fantastic!")
        break
    
        
#%% Function including Dictionary + Loop Practice

def emoji_converter(message):      #Takes String 'message' Input
    words = message.split(" ")      #split string at spaces into LIST
    emojis = {
        "Happy":":)",                   # E.g. "Good Morning :)" would become ["Good", "Morning", ":)"]
        "Sad": ":(", "Serious":":|" 
        }                                  
    output = " "                       #MUCH easier than it looks!
    for word in words:
        output += emojis.get(word, word) + " "       
    return output      #IF is NOT :) or :( will JUST add the WORD + SPACE
                        #
OG_message = input("Message: ")   #Need space between Sad and ? otherwise counts it TOGETHER!
                      #Dont NEED to do input, But Could put one here IF we want!
print(emoji_converter(OG_message))    #A simple STRING Will be FINE!


# Why is 'INPUT' NOT in the Function? 
#Input comes in different forms, Input recieved from CONSOLE/Terminal,
#But sometimes can be recieved in OTHER Aplications, 
#e.g.like User Interface.......
#Input is NOT REUSABLE so will NOT be needed!








#%% Conditional Statement with INPUT used(SUPER EASY!)
#Just a VERY SIMPLE Conditional Statement which converts between Kg and Lbs:
weight = int(input('Weight: '))   #just roound to WHOLE NUMBER/integer
unit = input('Kg or Lbs: ')             
if unit.lower() == "lbs":           #.upper() just puts in upper case, as we know.
    converted = weight *0.45        
    print("This weighs {converted} Kilos".format(converted=converted))
elif unit.lower() == "kg":
    converted = weight//0.4
    print("This weights {converted} Pounds".format(converted=converted)) 

#Inputs are NOT HARD! Just a way of INTERACTING IN Console!
#MOSTLY UNECESSARY to use Inputs, can just write Variables, BUT with Input, can ENTER into CONSOLE

#%% Guessing Game - Numbers (MORE WHILE LOOP PRACTICE - SIMPLE!):
secret_number = 21
guess_count = 0
guess_limit = 3
while guess_count < guess_limit:
    guess = int(input('Guess: '))
    guess_count += 1
    if guess != secret_number and guess_count >= guess_limit:
        print("Out of Guesses!")
        print("Please Try Again Soon!")
    elif guess != secret_number:
        print("Nope! Try Again.")
    elif guess == secret_number:
        print("You got it!")  
        print("Thank you for playing!")                          #GENIUS!
        break        
         #'Break' can ALSO be used for WHILE LOOPS
 #This ONLY Prints when we RUN OUT of GUESSES! 

#%% Car Game - Pretty Cool 'While Loop' Practice!
command = ""
started = False   #not yet started
stopped = False
while command != "quit":       #OR could say "While True" (a bit simpler))
    command = input("Command:").lower()    #easier to do lower case for JUST this, rather than repeating code unecessarily.
    if command == "start":
        if started==False:         #Conditional WITHIN a Conditional! COOL!
            print("Car Started")    #i.e. if (ALREADY) started
            started = True
        else:
            print("Car has already Started...!")
    elif command == "stop":
        if stopped == False:      
            print("Car Stopped")
            stopped = True
        else:
            print("Car has already stopped!")
    elif command == "help":
        print("""start - starts the car up
stop - stops the car
quit - the game has ended""")
    elif command == 'quit':
        print("Game Over!")
        break                      #BREAK CAN be used here, since is WITHIN a WHILE loop!
    else:                    #FINAL 'else'= if NONE OF ABOVE are Satisfied!
        print("Sorry, I didn't get that...")   

 
    
        
        

        
        
        














#%%PYTHON Code CHALLENGES 2
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
def unique(word):
    count = 0
    for letter in letters:
        if letter in word:
            count += 1
    return count
print(unique('mississipi'))   #Should be '4' - Does NOT Count Repeats!

def string_between_letters(word, start, end):
    start_index = word.find(start)
    end_index = word.find(end)    #finds our index for BOTH
    print(start_index)
    print(end_index)
    slice_word = word[start_index+1:end_index]
    return slice_word

print(string_between_letters('mississipi', 'l', 'p'))
   #say start = 1 (i), end = -2 (p)

def every_other_letter(word):
    string_list = []
    for index in range(0,len(word), 2):
        string_list.append(word[index])
    lst_to_string = ''.join(string_list)
    return lst_to_string
print(every_other_letter('Skywalker'))
    
#EASIER WAY:
def every_other(word): 
    every_other = " "     #REMEMBER! Can ADD to STRINGS 
    for index in range(0, len(word), 2):
        every_other += word[index]
    return every_other

def reverse_string(word):
    reverse = " "
    for index in range(len(word)-1, -1, -1):
        reverse += word[index]
    return reverse
print(reverse_string('Skywalker'))

def spoonerism(word1, word2):
    swapped = word2[0] + word1[1:] + " " + word1[0] + word2[1:]
    return swapped
print(spoonerism('Joe', 'Rogan'))

Rogan = 'Rogan'
while len(Rogan) < 20:
    Rogan += '!'
print(Rogan)    #While characters are LESS THAN 20, will KEEP ADDING '!'

def add_ten(my_dictionary):
    for key in my_dictionary.keys():
        my_dictionary[key] += 10
    return my_dictionary
print(add_ten({'Episode1':75, 'Episode2':70, 'Episode3':85}))

def values_that_are_keys(my_dictionary):
    values_ALSO_keys = []
    for value in my_dictionary.values():
        if value in my_dictionary.keys():
            values_ALSO_keys.append(value)
    return values_ALSO_keys
print(values_that_are_keys({'Joe':'Rogan', 'Rogan':'Joe', 'Keys':'Alicia', 'Alicia':'Keys'}))

def max_key(my_dictionary):
    list_values = list(my_dictionary.values())
    max_value = max(list_values)
    for key, value in my_dictionary.items():
        if value == max_value:
            return key
print(max_key({1:100, 2:1, 3:4, 4:10}))  #EASY -Finds Max 'Value' and Returns the Max KEY ASSOCIATED with it!

def frequency_count(words):
    frequency_dictionary = {}
    for word in words:
        value = words.count(word)
        frequency_dictionary[word] = value
    return frequency_dictionary
print(frequency_count(["Kebab", "Kebab", "Kebab", "Kebab", "Jason Orange", "Steamboat Willie", "F Hitler", "F Hitler", "I Love America", "I Love America"]))

def unique_values(my_dictionary):
    list_uniques = []
    for value in my_dictionary.values():
        if not value in list_uniques:
            list_uniques.append(value)
        else:
            list_uniques = list_uniques
    return len(list_uniques)    #returns HOW MANY Unique Values are present
print(unique_values({0:3, 1:1, 4:1, 5:3}))

def count_first_letter(names):   #names = dictionary of Surname (Key) and List of Characters First Names with this Surname (Values)
     letters = {}
     for key in names.keys():
         first_letter = key[0]     #e.g. 'S' for Stark
         if not first_letter in letters:
             letters[first_letter] = len(names[key])
         elif first_letter in letters:
             letters[first_letter] += len(names[key])
     return letters
print(count_first_letter({"Stark": ["Ned", "Robb", "Sansa"], "Snow" : ["Jon"], "Lannister": ["Jaime", "Cersei", "Tywin"]}))
print(count_first_letter({'Salamanca':["Lalo", "Hector", "Tuco", "Marco", "Leonel"], "White":['Skyler', 'Walter', 'Holly', 'Flynn'], 'McGill':['Chuck', 'Jimmy']}))
#Above function simply COUNTS the Number of People who have Last Names which have the SAME FIRST LETTERS (e.g. Stark and Snow BOTH begin with 'S')

#CLASS Example (Pretty Easy!) 
class DriveBot:
    all_disabled = False    #Class Variables defined
    latitude = -999999
    longitude = -999999
    robot_count = 0
    def __init__(self, motor_speed=0, direction=180, sensor_range=10):    #Provided DEFAULT VALUES for Arguments!
        self.motor_speed = motor_speed     #Instance Variables
        self.direction = direction
        self.sensor_range = sensor_range
        DriveBot.robot_count += 1   #CLASS.CLASS_Variable - ACCESS Class Variable FROM WITHIN!!! This can be done!
        self.id = DriveBot.robot_count  #Accessing .id will give us THIS 'LABEL' for the Robots
    def control_bot(self, new_speed, new_direction):
        self.motor_speed = new_speed
        self.direction = new_direction
        return new_speed, new_direction
    def adjust_sensor(self, new_sensor_range):
        self.sensor_range = new_sensor_range     #Methods here just OVERWRITE Default Instance Variables to NEW Values.
#Now can Instantiate:
robot1 = DriveBot()   #KEEP Default Values provided!
robot1.motor_speed = 5   #Change Instance Variables FROM Default of 0
robot1.direction=90       #(FOR 'robot1' Instance)                           
robot1.sensor_range = 10
print(robot1.control_bot(10,180))

robot2 = DriveBot(35,75,25)     #UPDATE Default Values
print(robot2.control_bot(5, 90))  #(5, 90)
robot3 = DriveBot(20, 60, 10)
print(robot3.direction)   #updated to 60
print(robot1.id)     #Cool - EACH TIME DriveBot is INSTANTIATED (i.e. New Robot made), does COUNT += 1, so Creates LABELS for EACH!
print(robot2.id)  
print(robot3.id)

DriveBot.longitude=50    #UPDATES Class Variable Values (WITHIN CLASS ITSELF!)
DriveBot.latitude=-50
DriveBot.all_disabled = True   #again, this changes Class Variable DIRECTLY WITHIN the Class!
print(DriveBot.latitude)    #Accesses Class Variables USING the CLASS! (Class.Class_Variable Pair AGAIN!)
print(DriveBot.robot_count)   #Tells us we have 3 Robots - CORRECT!

#Pretty Easy, but Learnt some NEW CLASS FEATURES: 
#1. Creating COUNT for Instances   2. CHANGING CLASS Variables by doing Class.Class_variable Pairs!






