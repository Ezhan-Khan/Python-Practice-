
#%%                      MODULE 4  - 'MODEL DEVELOPMENT' 
# (LINEAR REGRESSION, MODEL EVALUATION with VISUALIZATION, POLYNOMIAL REGRESSION and PIPELINES, 'MEASURES' for IN-SAMPLE Evaluation (R-Squared and MSE), PREDICITON and DECISION-MAKING)

#     Question  -  "HOW can you DETERMINE a 'FAIR VALUE' for a USED-CAR?"

# 'MODEL' = MATHEMATICAL EQUATION, used to PREDICT a 'VALUE', GIVEN 1 (or More) 'OTHER VALUES'
# RELATING '1 or More' 'INDEPENDENT' Variables TO 'DEPENDENT' Variables
# Example: 
# - INPUT 'highway-mpg' of a Car as 'INDEPENDENT Variable' INTO a MODEL
# - Model's OUTPUT given as 'PREDICTED PRICE' = DEPENDENT Variable 

#Using 'MORE RELEVANT' DATA (MORE INDEPENDENT VARIABLES) - results in MORE ACCURATE MODEL!
#For Example - INPUTTING 'highway-mpg' AND 'curb-weight' AND 'engine-size' = MORE ACCURATE PREDICTION of 'price'
#e.g. - for '2 ALMOST IDENTICAL' Cars, have 'DIFFERENT COLOURS' ('pink' = lower price than 'red' colour car), if we DONT SPECIFY the 'colour' Varaible TOO, would get 'LESS ACCURATE' PRICE PREDICTION!
#So MUST know WHICH INDEPENDENT VARIABLES are BEST for a GIVEN SITUATION!!!

#ALSO important to TRY OUT 'DIFFERENT TYPES' of 'MODELS' (will cover some now!)



#                 SIMPLE 'LINEAR REGRESSION' 
#(Simple) Linear Regression = ONE INDEPENDENT Variable (x) used to MAKE a PREDICITON 
# x = Independent Variable (Predictor), y = Target (Dependent) Variable 
#        ' y = b0 + b1x '
#b0 = intercept, b1 = slope

#       HARD to WORK OUT the 'Price' of a Car:
# Highway 'miles per gallon' can be FOUND in OWNER'S MANUAL   
#So? Could ASSUME 'LINEAR' Relationship between 'mpg' and 'Price' 
#          'y = 3843 - 821x'   
# e.g. IF 'x=20', would get '= 38423 - (821*20)' = '22,003' PRICE PREDICITON!

#       Get 'FIT' for the Line? - STEPS:
# 1. Take DATA POINTS from Data Set
# 2. use Training Points to FIT/TRAIN the MODEL, 
# 3. this GIVES us 'b0' and 'b1' 
# 4. USE these Parameters IN the MODEL - Now we HAVE the MODEL!
# 5. STORE Data Points in 'NUMPY ARRAYS' - X = Predictor Values, Y - TARGET Value

#MANY FACTORS which Influence 'price' of car - Car's AGE, MAKE...
#Can add 'SMALL RANDOM VALUE' to Data Points = 'NOISE'!

#So, USE MODEL to 'PREDICT VALUES' NOT SHOWN!
#Example - can PREDICT 'Price' for '20mpg' car (not given data points for this - so can 'PREDICT' IT!)

# NOTE - Model is NOT ALWAYS CORRECT!!!
# Should COMPARE the 'PREDICTED' Value to 'ACTUAL' Value
#Any ERROR could be due to NOISE (or other reasons)

#           'SIMPLE' LINEAR REGRESSION in 'PYTHON':
#Below are ALL REQUIRED LIBRARIES (for this Module):
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
#1. IMPORT 'linear_model' from 'scikit-learn':
from sklearn.linear_model import LinearRegression

#2. Create a 'LINEAR REGRESSION OBJECT' with 'Constructor':
lm_simple = LinearRegression()    #(i.e. class instance)
#Then Define our 'PREDICTOR' VARIABLE and 'TARGET' Variable
filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
car_df = pd.read_csv(filepath)
X = car_df[['highway-mpg']]
Y = car_df[['price']]
#Use 'lm.fit(X,Y)' to FIT the MODEL (uses the '.fit()' Method)
lm_simple.fit(X,Y)  #USE this to FIND 'b0' and 'b1'
Yhat_simple = lm_simple.predict(X)   #get an ARRAY of PREDICTED' VALUES for 'Y' (given as 'Yhat') for EACH 'X' Array Value!
#'b0' attribute ('INTERCEPT') GIVEN by 'lm.intercept_'
lm_simple.intercept_     # '38423.306'
#'b1' attribute ('SLOPE') given by 'lm.coef_'
lm_simple.coef_          #-821.7333'

#SHOWING this Linear Model between 'Price' and 'Highway-mpg':
#     ' Price = 38423.31 - 821.73*highway-pmpg '  -  SIMPLE!
# (just replaced 'Y' and 'x1' with our 'ACTUAL VARIABLE NAMES')
Price = 38423.31 - 821.73*car_df['highway-mpg']
Price    #PREDICTED 'Prices' using SIMPLE LINEAR REGRESSION
         # = SAME as 'Yhat_simple'


#                  'MULTIPLE' LINEAR REGRESSION
# = 'MULTIPLE INDEPENDENT' VARIABLES used to MAKE a PREDICTION (for Continuous 'TARGET Y Variable')
#           'Y = b0 + b1*x1 + b2*x2 + b3*x3 + ....'
# b0=intercept (X=0), b1 = COEFFICIENT of 'x1', b2 = COEFFCIIENT of 'x2',....and so on

#Example -     'Y = 1 + 2*x1 + 3*x2'

#                         IN PYTHON (SAME as ABOVE!):
#Store 4 Predictor (Independent) Variables to a Variable 'Z'
lm_multiple = LinearRegression()
Z = car_df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Y = car_df[['price']]
#Use 'lm.fit' again (as before)
lm_multiple.fit(Z, Y)
#OBTAIN 'ARRAY of PREDICTED' Values (using PREDICTOR VARIABLES in 'Z' Array/DataFrame):
Yhat_multiple = lm_multiple.predict(Z)  #201 elements of Yhat (SAME as Length of 'PREDICTOR Variables (Z)' Sample)
#Find 'intercept (b0) and 'coefficients' (b1, b2, b3, b4) the SAME WAY:
lm_multiple.intercept_    # '-15806.6246'
lm_multiple.coef_         #Array of coefficients:  array([[53.49574423,  4.70770099, 81.53026382, 36.05748882]])

#Our Estimated Linear Model between 'Price' and '4 Predictor Variables':
#    'Price = -15678.74 + (52.66)*horsepower + (4.70)*curb-weight + (81.96)*engine-size + (33.58)*highway-mpg'
# (just replaced Yhat, x1, x2... with our 'ACTUAL VARIABLE NAMES'):
Price = -15678.74 + (52.66)*car_df['horsepower'] + (4.70)*car_df['curb-weight'] + (81.96)*car_df['engine-size'] + (33.58)*car_df['highway-mpg']


#                'MODEL EVALUATION' USING 'VISUALIZATION':

#          'REGRESSION PLOT'
# Provides GOOD ESTIMATE of:
#'Relationship' between 2 Variables
# 'Strength' of 'Correlation', 
# 'Direction of Relationship' (+ve or -ve)
#Get SCATTER PLOT (each data point with Different 'y' value)
#AND get our 'FITTED LINEAR REGRESSION Line' (Yhat - PREDICTED Values!)
sns.regplot(x="highway-mpg", y="price", data=car_df)
plt.ylim(0,)   #SUPER SIMPLE! Just like that!

#           'RESIDUALS PLOTS'
#   'RESIDUAL' can be found as 'Y - Yhat' (Target Y Value - Predicted Y Value)
#   Look at 'SPREAD of RESIDUALS' Plot: 
# - IF RANDOMLY SPREAD around 'x-axis', means LINEAR Model is BEST!
# - IF Residuals Plot is 'CURVED/NOT RANDOMLY SPREAD' - Suggests 'LINEAR' Model Assumption is INCORRECT!
#   (in this case, may need NON-LINEAR Model, covered LATER!)
sns.residplot(x = car_df['highway-mpg'], y = car_df['price'])
#Here, for 'highway-mpg', RESIDUALS have 'SLIGHT CURVATURE'!

#          'DISTRIBUTION PLOTS'
# = COUNTS 'PREDICTED' Value vs. 'ACTUAL' Value 
# - BEST for Models with 'MULTIPLE INDEPENDENT' VARIABLES (i.e. MULTIPLE Linear Regression!)
#   (SEE 'WORD DOCUMENT Notes (see the Visuals)' for HOW these are CREATED!)
#Essentially COUNTS and PLOTS the 'PREDICTED' Y Values which are EQUAL to 1, 2, 3, ... (PROPORTIONS)
#Then on SAME PLOT, we Plot the ACTUAL (Target) Y VALUES which are EQUAL to 1, 2, 3, 4, (PROPORTIONS)
#This is presented as a DISTRIBUTION PLOT ( = for CONTINUOUS VALUES, Whereas 'HISTOGRAM' is for DISCRETE Values, so is CONVERTED to 'DISTRIBUTION' here, NOT Histogram!)
ax1 = sns.distplot(car_df['price'], hist=False, color='r', label = "Actual Value")
sns.distplot(Yhat_multiple, hist=False, color="b", label="Fitted Values", ax=ax1)
#Have NICELY Plotted our 'DISTRIBUTION' Plot for 'price' Y-Variable!
#See that our 'PREDICTION PLOT' (blue)  is 'SIMILAR' to the 'ACTUAL PLOT' (red) - GOOD!

#COMPARING to if 'SINGLE Linear Regression' Prediction:
ax1 = sns.distplot(car_df['price'], hist=False, color='r', label = "Actual Value")
sns.distplot(Yhat_simple, hist=False, color="b", label="Fitted Values", ax=ax1)
#MUCH 'POORER FIT', since only 1 X (Predictor) Variable Used!



#             'POLYNOMIAL REGRESSION' and 'PIPELINES'

#What do we do when a 'LINEAR' Model is NOT BEST for our Data?
#          'POLYNOMIAL REGRESSION' (= NON-LINEAR MODEL)
#   1. 'TRANSFORM' Data INTO 'Polynomial'  
#   2. Then 'use LINEAR REGRESSION' to 'FIT' the Parameter
# = 'CURVILINEAR' Relationships (i.e. 'SQUARING'/Setting 'HIGHER ORDERS' of 'PREDICTOR' Variable)

#     'QUADRATIC' (2nd Order Polynomial) 
# -    'Yhat = b0 + b1*x1 + b2*(x1)**2     (Squared/Raised by Order of 2)

#     'CUBIC' (3rd Order Polynomial)
# -    'Yhat = b0 + b1*x1 + b2*(x1)**2 + b3(x1) **3     

#      'HIGHER ORDER' Polynomials...
#Overall? - MUST PICK the CORRECT 'ORDER' of Polynomial which BEST FITS our DATA!
x = car_df['highway-mpg']
y = car_df['price']     #Note: 'np.polyfit' ONLY works with 1D Vector/SERIES - NOT a 'Dataframe', so ONLY used '[]' SINGLE Brackets!
#'1D' Example - 'np.polyfit(x,y,order)'      (NUMPY Function)
f = np.polyfit(x,y,3)   #'3rd ORDER' Polynomial 
p = np.poly1d(f)       #STICKS to '1D' here
print(p)    #PRINTS OUT our 'MODEL FUNCTION'- COOL!
# ' -1.557 x**3 + 204.8 x**2 - 8965 x + 1.379e+05 '

# CAN get MORE COMPLEX if we add 'MULTIPLE VARIABLES' Polynomial Linear Regression!!!
#   ('MULTIPLE' VARIABLES/also called 'FEATURES'):
from sklearn.preprocessing import PolynomialFeatures
# - Create 'PolynomialFeatures' OBJECT/INSTANCE
pr = PolynomialFeatures(degree=2, include_bias = False)
# - TRANSFORM our X-Values/Features INTO POLYNOMIAL:
x_poly = pr.fit_transform(car_df[['horsepower', 'curb-weight']])
#Here, did '2nd order' Polynomial with '2 VARIABLES' ('horsepower' and 'curb-weight')

#BEST to Stick to 'SIMPLER 2D' Example:
pr = PolynomialFeatures(degree=2)     #2nd Degree/2D Polynomial of our 'FEATURES' (=PREDICTOR VARIABLES)
pr.fit_transform([1,2], include_bias = False)
#Now, have NEW SET of Features which are a TRANSFORMED VERSION of our ORIGINAL FEATURES!

#Can 'NORMALIZE/SCALE' EACH 'FEATURE/Variable' SIMULTANEOUSLY (as we get MORE Features):
#Use 'Pre-Processing Module' to SIMPLIFY:
from sklearn.preprocessing import StandardScaler
SCALE = StandardScaler()     #using 'StandardScaler()' Constructor
SCALE.fit(car_df[['horsepower', 'highway-mpg']])
x_scale = SCALE.transform(car_df[['horsepower', 'highway-mpg']])


#                   'PIPELINES'
# = Great way to 'SIMPLIFY CODE'
#'MANY STEPS' to OBTAINING an 'ACCURATE PREDICTION'
# - 'Normalization', 'Polynomial Transformation', 'Linear Regression'
#So? can SIMPLIFY PROCESS by using 'PIPELINES'
#   = Performing 'SEQUENCE' of 'TRANSFORMATIONS':

#1. Import all Libraries we need:
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
Z = car_df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
y = car_df['price']
#Create 'Input' as 'LIST OF TUPLES' (for EACH STEP of Process!)
# (For EACH, FIRST Tuple Element = 'Estimator MODEL', SECOND Tuple Element ='Model CONSTRUCTOR Method/Function')
Input = [('scale', StandardScaler()),('polynomial', PolynomialFeatures(degree=2)),('model',LinearRegression())]
pipe = Pipeline(Input)     # = 'PIPELINE OBJECT'
#INPUT the LIST 'Input' INTO the 'Pipeline(Input)' CONSTRUCTOR
#Now can 'TRAIN' this PIPELINE OBJECT:
pipe.fit(Z, y)
piped_yhat = pipe.predict(Z)
piped_yhat[0:10]  #viewing first 10 'Predicted' Y Values!
#OVERALL STEPS:
# 1.  'Normalizes' Data (with 'Standard Scaler')
# 2.  Performs 'Polynomial Transform'
# 3.  Finally - Outputs a 'PREDICTION' (LINEAR REGRESSION)
#PIPELINE allows you to perform ALL 3 STEPS 'TOGETHER'!
#Now PLOTTING 'DISTRIBUTION PLOT' to EVALUATE Model:
ax1 = sns.distplot(car_df['price'], hist=False, color='r', label = "Actual Value")
sns.distplot(piped_yhat, hist=False, color="b", label="Fitted Values", ax=ax1)
#as we, this 'POLYNOMIAL' is a GREAT FIT!



#            'MEASURES' for 'IN-SAMPLE EVALUATION'

#(NUMERICALLY) MEASURING 'HOW WELL' our MODEL FITS the Data?
# -  'Mean Squared Error' (MSE) and 'R-Squared'

#           MEAN SQUARED ERROR (MSE)
#find DIFFERNCE between ACTUAL Value (Y) and PREDICTED Value (Yhat)
#Then 'SQUARE' IT!
#e.g. ( 'Y1 - Yhat1')**2 = (150 - 50)**" = (100)**2
#Then can find 'MEAN' of 'ALL ERRORS' (= SUM / Number of Samples) 

#In PYTHON:
from sklearn.metrics import mean_squared_error
#use 'mean_squared_error()' function:
mean_squared_error(actual_value(Y), predicted_value(Yhat) )  #DONT RUN -JUST for DISPLAY!


#           R-SQUARED ('Coefficient of Determination')
# = MEASURE to determine 'FIT' - i.e. 'HOW CLOSE' Data 'IS' to 'FITTED REGRESSION LINE'
# Consider 'Base Model' is 'Ybar' = AVERAGE of 'y-values'
# 'R-Squared' COMPARES the 'REGRESSION LINE' TO this 'AVERAGE'

#   (SEE WORD Notes for 'THEORY' of GOOD FIT EXPLAINED - Optional)
# 'R-Squared' is Found to be BETWEEN '0 to 1'

#IN PYTHON:       R-Squared = 'lm.score(X, Y)'
X = car_df[['highway-mpg']]
Y = car_df[['price']]
lm_simple.fit(X,Y)  #Create the Linear Regression Fit (AS USUAL!)
#Use '.score()' method to get 'R-Squared' Value:
lm_simple.score(X, Y)    #  = '0.497'
# = APPROXIMATELY '49.7%' of 'VARIATION' in 'price' is EXPLAINED by this SIMPLE LINEAR MODEL



#                'MAKIING PREDICTIONS' and DECISION MAKING'

#i.e. HOW can we DETERIME IF our MODEL is CORRECT?

#             1. Do PREDICTED Values MAKE SENSE?
lm_simple.fit(car_df[['highway-mpg']], car_df['price'])
#EXAMPLE -  Predicting 'PRICE' for Car with '30 mpg' as 'Highway-mpg': 
prediction = lm_simple.predict(np.array(30).reshape(-1, 1))   
#(Note: MUST use '.reshape(-1,1)' = Converts to 'SINGLE COLUMN' ARRAY
prediction  #Predicted 'Price' Value = '13771.30' Dollars
#MAKES SENSE - Not Too High or Too Low.
#Helps to ALSO EXAMINE the COEFFICIENTS:
lm_simple.coef_     # '-821.73' 
# 'SLOPE' -  so for '1 Unit INCREASE' in 'highway-mpg' = DECREASE in Car VALUE by '821' dollars (since '-821')
#Note: MAY get 'NEGATIVE PRICE' Values = UNREALISTIC!
#      Reason? - Linear Assumption may be INCORRECT, OR Data for CARS is NOT AVAILABLE for THAT RANGE! 
#MOST likely that this is just NOT REASONABLE 'mpg' RANGE for a Car - so STILL VALID!

#(EXPLAINING) - Generating 'SEQUENCE of Values' in PYTHON:
new_input = np.arange(1,101,1).reshape(1,-1)
new_input  #Sequence STARTS at 1, 'INCREMENTING by 1' UNTIL we REACH '100'
# 'np.arange' = ARRAY-RANGE = outputs a SEQUENCE/RANGE of a Values AS an 'ARRAY'!
#Use this to 'PREDICT NEW Values' In this RANGE:
lm_simple.fit(X, Y)
#APPLY 'model.predict(new_input)' TO 'new_input' Values:
yhat = lm_simple.predict(new_input)  #have MANY NEGATIVE Values!
yhat[0:5]   #first 5 predicted values 
#PLOT the Data:
plt.plot(new_input, yhat)
plt.show()


#                     2. Use VISUALIZATION
#FIRST - Can use 'Regression Plot' to simply visualize data
sns.regplot(x="highway-mpg", y="price", data=car_df)
plt.ylim(0,)    #AS SEEN ABOVE ALREADY!
#RESIDUAL Plot
sns.residplot(x = car_df['highway-mpg'], y = car_df['price'])
#Any Small CURVATURE in RESIDUALS Plot indicates 'NON-LINEAR' Behaviour
#'DISTRIBUTION Plot' = Good for Evaluating 'MULTIPLE' LINEAR REGRESSION
#(compares DISTRIBUTION of 'ACTUAL Y Values' to PREDICTED 'Yhat' Values)
#by 'ADDING MORE DATA' (More Independent Variables) = MORE ACCURATE PREDICTIONS


#3. NUMERICAL MEASURES for EVALUATION (Covered ABOVE ALREADY!)
#'MSE' = 'most intuitive' numerical measure. 
#'LARGER MSE = More SPREAD' of Data Points FROM REGRESSION LINE = LESS ACCURATE!
# (so SMALLEST 'MSE' = BEST MODEL/BEST FIT)
#R-SQUARED' = ANOTHER Popular Method for Model Evaluation
# According to 'Falk and Miller' - ACCEPTABLE VALUE for 'R-Squared' is '>= 0.10'


#4.  'COMPARING MODELS' (DECISION-MAKING)
#Comparing 'MULTIPLE LR' with 'SIMPLE LR'
#Does a 'LOWER MSE' = 'BETTER FIT'?
#NOT NECESSARILY! 
# - for MLR Model, MSE will be SMALLER THAN MSE for 'SLR'
#Why? because 'MLR' has 'MORE INDEPENDENT VARIABLES = LESS ERROR in Data'
#'POLYNOMIAL' Regression ALSO has 'SMALLER MSE' than Regular (Linear) Regression!
# (OPPOSITE Case for 'R-Squared', CLOSER to 1 is NOT NECESSARILY 'Better Fit', when COMPARING MODELS)

#WHICHEVER Model has LOWEST MSE and HIGHEST R-Squared, indicates it is BEST MODEL for the Given Data!



#%%                 MODEL DEVELOPMENT: PRACTICE 1 - Car Data
#Questions which 'MODEL Development' HELPS to ANSWER:
#"Is the Dealer offering a FAIR VALUE (Price)?"
#"Did I Put a FAIR VALUE (Price) on my Car?"

#Will 'PREDICT' the 'PRICE' (y-variable) of a Car using DIFFERENT MODELS:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load our 'Car DataFrame':
filepath =  'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
car_df = pd.read_csv(filepath)

# 1. SIMPLE Linear Regression Method:
from sklearn.linear_model import LinearRegression
lm = LinearRegression()   #create Linear Regression OBJECT
#Does 'highway-mpg' HELP us PREDICT 'car price'?
X = car_df[['highway-mpg']]
Y = car_df[['price']]
lm.fit(X,Y)           #'TRAINING' the MODEL
Yhat = lm.predict(X)    #Create ARRAY for PREDICTED 'prices' ('REGRESSION LINE' Data Points)
Yhat[0:5]   #viewing First 5 Elements/Predicted Values
            #Given as a sort of 'SINGLE COLUMN' ARRAY 'array([row1], [row2], ...)'
#Converting 'yhat' to 'DataFrame' - Just for Fun!
yhat_as_dataframe = pd.DataFrame(Yhat) 
yhat_as_dataframe.rename(columns = {0:"Yhat"}, inplace=True)   #just RENAMED Column Header to 'Yhat'

#Viewing 'intercept' and 'slope':
lm.intercept_    #38423.30585816
lm.coef_         # -821.73337832
#'EQUATION' for Estimated LINEAR MODEL:   
Price = 38423.31 - 821.73*car_df['highway-mpg']   # = 'Yhat'

#REPEAT this, NOW for 'X = engine-size':
lm1 = LinearRegression()  #NEW instance/object for 'DIFFERENT X Variable' Now
X = car_df[['engine-size']]
Y = car_df[['price']]
lm1.fit(X,Y)      
Yhat = lm1.predict(X)     
lm1.intercept_     #'-7963.3'
lm1.coef_          #'166.86'
#Writing 'EQUATION' for the LINEAR MODEL:
Price = -7963.3 + 166.86*car_df[['engine-size']]
Price   

# 2. 'MULTIPLE' LINEAR REGRESSION Model:
# (MOST 'Real-World' Regression Models have 'MULTIPLE PREDICTORS')
Z = car_df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, car_df['price'])   #TRAINING the MODEL
lm.intercept_  # = -15806.62462632919
lm.coef_       # = array([53.49574423,  4.70770099, 81.53026382, 36.05748882]
#Writing EQUATION for MULTIPLE LINEAR REGRESSION MODEL:
Price = -15806.62 + 53.5*car_df['horsepower'] + 4.7077*car_df['curb-weight'] + 81.53*car_df['engine-size'] + 33.583*car_df['highway-mpg']
Price

#REPEAT for 'normalized-losses' and 'highway-mpg':
lm2 = LinearRegression()
Z = car_df[['normalized-losses', 'highway-mpg']]
Y = car_df[['price']]
lm2.fit(Z,Y)
lm2.coef_       # array([[1.49789586, -820.45434016]])
lm2.intercept_  # array([38201.31])


# 3. 'MODEL EVALUATION' with VISUALIZATION:
import seaborn as sns
width = 12
height = 10      #(OPTIONAL - Just set SIZE of the Figure)
plt.figure(figsize=(width, height))  
sns.regplot(x="highway-mpg", y="price", data=car_df)
plt.ylim(0,)  
# = NEGATIVE CORRELATION between 'highwaw-mpg' and 'price'
# (higher mpg = LOWER Price Car)
#Pay attention to SPREAD of DATA AROUND the REGRESSION LINE!

#'peak-rpm' Regression Plot:
sns.regplot(x='peak-rpm', y='price', data=car_df)
plt.ylim(0,)  #  = WEAK CORRELATION - MORE SPREAD around the REGRESSION Line.

#PROVE this NUMERICALLY with '.corr()':
print(car_df[['peak-rpm', 'price']].corr())   # = '-0.101616'
print(car_df[['highway-mpg', 'price']].corr()) # =  '-0.704692'

#         'RESIDUAL PLOT':
width = 12
height = 10
plt.figure(figsize=(width, height))   #(Optional - just Set Size of Figure)
sns.residplot(x=car_df['highway-mpg'], y=car_df['price'])
plt.show()   
#have SLIGHT CURVATURE of Graph - NOT RANDOM SPREAD so 'NON-LINEAR' Model may be BETTER!

# Visualize 'MULTIPLE LINEAR REGRESSION'?
#Using 'DISTRIBUTION PLOT':
Z = car_df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
#First 'PREDICTING' to GET 'Yhat'
Y_hat = lm.predict(Z)
#PLOTTING 'Actual' vs. ' Predicted/Fitted Values' 
ax1 = sns.distplot(car_df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values", ax=ax1)
#Adding LABELS:
plt.title("Actual vs Fitted Values for Price")
plt.xlabel("Price (in dollars)")
plt.ylabel("Proportion of Cars")
plt.show()  #CLOSELY Overlaping - NOT BAD Predicted Y-Values!
plt.close()


#         POLYNOMIAL REGRESSION and PIPELINES:
# = 'NON-LINEAR' Relationships (with 'HIGHER-ORDER' or 'SQUARED' terms)
#If 'LINEAR' Relationship is NOT ENOUGH, can INSTEAD FIT 'POLYNOMIAL Model':

#'USER-DEFINED FUNCTION' to PLOT Data:
def PlotPoly(model, indep_var, dep_var, Name):
    x_new = np.linspace(indep_var.min(), indep_var.max(), 100)   # = RANGE for x-axis 
    y_new = model(x_new)   #get 'PREDICTED y-values' for 'POLYNOMIAL CURVE' 
    #- (just PROVIDING 'x_new' as 'x_values' for 'model')
    
    plt.plot(indep_var, dep_var, '.', x_new, y_new, "-")
    #just plots 'Price vs. CPU_frequency' as DOTS '.' 
    #'x_new' vs. 'y_new (Model PREDICTED VALUES)' as LINE '-'
    plt.title("Polynomial Fit with Matplotlib for Price ~ Length")
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel("Price of Cars")
    
    plt.show()
    plt.close()

#Now, Try 'highway-mpg' and 'price' with POLYNOMIAL
x = car_df['highway-mpg']
y = car_df['price']
#'FIT' to 'POLYNOMIAL' with 'polyfit':
f = np.polyfit(x,y,3)
f  #GIVES us the 'COEFFICIENTS' for the Model Function (in equation below)
p = np.poly1d(f)   #Gives FUNCTION of POLYNOMIAL MODEL
print(p)  #   '-1.557 x**3 + 204.8 x**2 - 8965 x + 1.379e+05'

#'PLOTTING' the 'FUNCTION':
PlotPoly(p, x, y, "highway-mpg")
#Clearly this 'Polynomial' Model FITS the Data MUCH BETTER!

#Simple! - Could do NOW for '11 Order' Polynomial:
f_11 = np.polyfit(x,y,11)
f
p_11 = np.poly1d(f_11)
print(p_11)
PlotPoly(p_11, x, y, "highway-mpg")
#HIGHER ORDER = More CLOSELY FIT the Data (as shown!)

# 'MULTI-VARIATE' Polynomial Function (MULTIPLE VARIABLES)
# - Gets COMPLEX!!!
from sklearn.preprocessing import PolynomialFeatures
Z = car_df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
y = car_df['price']
pr = PolynomialFeatures(degree=2) #Create 'OBJECT' for Polynomial
Z_pr = pr.fit_transform(Z)   #'POLYNOMIAL TRANSFORM'

Z.shape   #start with '201 samples, 4 features'
Z_pr.shape  #AFTER 'TRANSFORMATION' have '201 samples' and '15 features'

#      PIPELINE - SIMPLIFIES PROCESSING of DATA (when SEVERAL STEPS NEEDED)
#-here, will ALSO use 'StandardScaler'
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias = False)), ('model', LinearRegression())]
pipe = Pipeline(Input)     # = 'PIPELINE OBJECT'
pipe   #Creates PIPE for STEPS 'StandardScaler' - 'PolynomialFeatures' - 'LinearRegression'
#INPUT the LIST 'Input' INTO the 'Pipeline(Input)' CONSTRUCTOR
#Now can 'TRAIN' this PIPELINE OBJECT:
pipe.fit(Z, y)
ypipe = pipe.predict(Z)   # = PREDICTED Y Values (AFTER STEPS of PIPE!)
list(ypipe[0:4])     #Viewing First 4 Elements (as 'list') -  '[13102.74784201, 13102.74784201, 18225.54572197, 10390.29636555]


#      'MEASURES' for 'In-Sample Evaluation'
#ACCURACY of MODEL determined by 'R-Squared' and 'MSE'
# 1. For 'SIMPLE' Linear Regression:
#    Calculate R-Squared 
X = car_df[['highway-mpg']]
Y = car_df[['price']]
lm.fit(X,Y)  #Create the Linear Regression Plot (AS USUAL!)
#Use '.score()' method to get 'R-Squared' Value:
print(f"R-Squared Value is: {lm.score(X, Y)}") 
#so  '49.7%' of Variation in 'price' is EXPLAINED by 'Simple Linear Regression Model'

#     Calculate 'MSE':
Yhat = lm.predict(X)
print(f"First Four Predicted Values: {Yhat[0:4]}")
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y, Yhat)
mse     # 31635042.944639888

# 2. For 'MULTIPLE' Linear Regression:
Z   #for 'horsepower', 'curb-weight', 'engine-size', 'highway-mpg'
lm.fit(Z, Y)
print(f"R-Squared is: {lm.score(Z, Y)}")
# '-80.496%' of VARIATION in 'price' can be Explained by this 'MULTIPLE Linear Regression' Model

#    Calculate MSE:
Yhat_multifit = lm.predict(Z)
print(f"First Four Predicted Values: {Yhat[0:4]}")
mse_multifit = mean_squared_error(Y, Yhat_multifit)
mse_multifit   # '11980366.87072649'

# 3.  For 'POLYNOMIAL' Fit:   (DIFFERENT)
#For 'POLYNOMIAL', we CANT use 'lm.score' (obviously)
#So IMPORT 'r2_score(y, p(x))' FROM 'sklearn.metrics'
from sklearn.metrics import r2_score
r_squared = r2_score(y, p(x))
print(f"R-Squared for this POLYNOMIAL Fit is: {r_squared}")
# = '67.4% of variation can be explaned by this POLYNOMIAL Fit!

#     Calculate MSE (SAME WAY! With 'mean_squared_error')
mean_squared_error(y, p(x))  #'20474146.426361218'




#%%          MODEL DEVELOPMENT: PRACTICE 2 - 'LAPTOP DATA'
#Import ALL REQUIRED LIBRARIES:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv'
laptop_df = pd.read_csv(filepath)
laptop_df.head()

#             SINGLE LINEAR REGRESSION:
#'CPU_frequency' and 'Price'
lm = LinearRegression()
X = laptop_df[['CPU_frequency']]  
#Important Note -  NEED '[[Column]]' (CONVERTS to 2D Array/DataFrame) otherwise gives 'ERROR' for SINGLE BRACKETS! SOMETIMES WONT give error, but BEST PRACTICE is to DO it!
Y = laptop_df['Price']
lm.fit(X, Y)
Yhat = lm.predict(X)  #PREDICTED 'Y' Values (Regression Line)

#'Distribution Plot' of 'PREDICTED' against 'ACTUAL':
ax1 = sns.distplot(Y, hist=False, color='r', label = "Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values", ax=ax1)
plt.title("Actual vs Fitted Values for Price")
plt.xlabel("Price")
plt.ylabel("Proportion of Laptops")
plt.legend(['Actual Value', 'Predicted Value'])
plt.show()   #VERY POOR FIT of 'Actual' Data!

#MSE and R-Squared:
r_squared = lm.score(X,Y)    # '0.1344'
MSE  = mean_squared_error(Y, Yhat)  #'284583.44'
#LOW VALUE of 'R-Squared' - indicates pretty POOR Fit

#           MULTIPLE LINEAR REGRESSION:
Z = laptop_df[['CPU_frequency', "RAM_GB", "Storage_GB_SSD", "CPU_core", "OS", "GPU", "Category"]]
lm_mult = LinearRegression()
lm_mult.fit(Z,Y)
Yhat_mult = lm_mult.predict(Z)
#'Distribution Plot':
ax1 = sns.distplot(Y, hist=False, color='r', label = "Actual Value")
sns.distplot(Yhat_mult, hist=False, color="b", label="Fitted Values", ax=ax1)
plt.title("Actual vs Fitted Values for Price")
plt.xlabel("Price")
plt.ylabel("Proportion of Laptops")
plt.legend(['Actual Value', 'Predicted Value'])
plt.show()    #'Slightly BETTER' FIT (closer to 'ACTUAL')
#MSE and R-Squared:
r_squared_mult = lm_mult.score(Z,Y)    # '0.50825'
MSE_mult  = mean_squared_error(Y, Yhat_mult) 
# = '161680.573'
#Overall, is BETTER than 'SINGLE' Linear Regression - Higher R-Squared, 

#            POLYNOMIAL REGRESSION:
x = laptop_df['CPU_frequency']
y = laptop_df['Price']
#Trying out 'np.polyfit' for DIFFERENT 'ORDERS' - SIMPLE!
f1 = np.polyfit(x,y,1)
print(f1)
p1 = np.poly1d(f1)
print(p1)
f3 = np.polyfit(x,y,3)
print(f3)
p3= np.poly1d(f3)
print(p3)
f5= np.polyfit(x,y,5)
print(f5)
p5= np.poly1d(f5)
print(p5)

#  Using 'PlotPoly' (ABOVE) to PLOT OUT thse 3 MODELS:
PlotPoly(p1, x, y, "CPU_frequency")
PlotPoly(p3, x, y, "CPU_frequency")
PlotPoly(p5, x, y, "CPU_frequency")


#Finding 'R-SQUARED' and 'MSE' for EACH of these FITS!
from sklearn.metrics import r2_score
r_squared = r2_score(y, p1(x))    # = 0.134
print(f"R-Squared for this POLYNOMIAL Fit is: {r_squared}")
r_squared = r2_score(y, p3(x))   # = 0.267
print(f"R-Squared for this POLYNOMIAL Fit is: {r_squared}")
r_squared = r2_score(y, p5(x))   # = 0.303
print(f"R-Squared for this POLYNOMIAL Fit is: {r_squared}")
from sklearn.metrics import mean_squared_error
MSE_1 = mean_squared_error(y, p1(x))
MSE_3 = mean_squared_error(y, p3(x))
MSE_5 = mean_squared_error(y, p5(x))
print(f"MSE for p1(x): {MSE_1}, p3(x): {MSE_3} and p5(x): {MSE_5}")

#          -  'PIPELINE' to perform SET of OPERATIONS:
#Want to do Parameter SCALING, POLYNOMIAL and LINEAR REGRESSION:    
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias = False)), ('model', LinearRegression())]
piped = Pipeline(Input)     # = 'PIPELINE OBJECT'
piped   # = PIPE for STEPS 'StandardScaler' - 'PolynomialFeatures' - 'LinearRegression'
#Now 'TRAIN' this PIPELINE OBJECT, FIT 'Z' and 'y'!
piped.fit(Z, y)
ypiped = piped.predict(Z)   # = PREDICTED Y Values 
ypiped[0:10]   #viewing first 10 Predicted Y Values

#Finding MSE and R-SQUARED for this:
from sklearn.metrics import mean_squared_error
MSE_pipe = mean_squared_error(y, ypiped)
MSE_pipe     #120595.66
#use 'r2_score(Y, Yhat)' for PIPELINE (since it INCLUDES 'Polynomial' here):
R_Squared_Piped = r2_score(y, ypiped)   
R_Squared_Piped     #0.633  
# -  GOOD FIT! HIGHER R-Squared Value!

#Conclusion - VALUES of R-Squared INCREASES from 'SIMPLE' Linear Regression TO 'MULTIPLE' Linear Regression
#EVEN FUTHER as we go to 'PIPELINE' with Multiple Linear Regression AND 'POLYNOMIAL' Features!


#%%                      MODULE 5 - 'MODEL EVALUATION (ACCURACY)'
# (Model REFINEMENT, OVERFITTING, UNDERFITTING, Model SELECTION, 'RIDGE' Regression, GRID SEARCH)

#NOW will look at how to EVALUATE Model in 'REAL-WORLD'!



#                    'MODEL EVALUATION' and 'REFINEMENT'
#Here will discuss HOW MODEL will 'PERFORM' in 'REAL-WORLD'
#ABOVE, covered 'IN-SAMPLE' Model Evaluation = JUST on the 'DATA GIVEN' to 'TRAIN' with
#LIMITED - Does 'NOT' Tell us HOW the TRAINED MODEL can be USED to 'PREDICT NEW' Data!

#'SPLIT DATA' into 'TRAINING' AND 'TESTING' SETS:
#TRAIN - use IN-SAMPLE/TRAINING Data to TRAIN the MODEL
#TEST - REMAINING Data = 'OUT-OF-SAMPLE' EVALUATION/'TEST' SET
#     (TEST Data = 'BEST' APPROXIMATE 'HOW Data' will 'PERFORM' in 'REAL-WORLD'!)

#'LARGER PORITON' of Data used for 'TRAINING' than Testing:
#Example - SPLIT DATASET into '70% Training', '30% Testing'
#1. BUILD and 'TRAIN/FIT' MODEL with 'TRAINING SET'
#2. Then Use 'TESTING' Set to 'ASSESS PERFORMANCE' of PREDICTIVE MODEL (i.e. find R-Squared/Other Metric...)
#'AFTER ALL TESTING' is DONE - Use 'ALL DATA' for 'TRAINING MODEL' (for BEST PERFORMANCE!)

#'Scikit-learn' Package has FUNCTION to SPLIT: 
# "train_test_split'  - 'RANDOMLY SPLITS' Dataset into TESTING and TRAINING Subsets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3,random_state=0 )
#x_data = FEATURES/INDEPENDENT Variables (e.g. car_df['highway-mpg'] )
#y_data = Dataset TARGET (df['price'])
#x_train, y_train = Parts of AVAILABLE DATA as TRAINING SET
#x_test, y_test = Parts of AVAILABLE DATA as TESTING SET
#'test_size'= 'PERCENTAGE' of Data for 'TESTING'
#'random_state = 0 or 1' - just ensures the 'SAME' RANDOM SPLIT for 'EACH TIME' the CODE is RUN! 


#                'GENERALIZATION ERROR':
# = MEASURE 'HOW WELL' Data can PREDICT 'PREVIOUSLY UNSEEN' Data
#'ERROR from TESTING' Data = APPROXIMATION of 'Generalization' Error
#e.g. Generating 'DISTPLOT' (Predicted Value): 
#DIFFEERNT if using 'TRAINING Data' than 'TEST' Data 
#This 'DIFFERENCE' is DUE to 'GENERALIZATION ERROR' 
# = REPRESENATION of 'REAL WORLD' 

#Using 'LARGER SAMPLE SIZE' for 'TRAINING' Data: 
# = 'MORE ACCURATE'(close approximation of Generalization error), to show REAL-WORLD, 
#    BUT 'LOWER PRECISION' (EACH 'REPEATED RESULT' will be VERY DIFFERENT from 'EACH OTHER')
# i.e. 'Visualize a BULLSEYE' - Centre = 'CORRECT GENERALIZATION ERROR'
# - Take 'RANDOM SAMPLE' of Data, '90% Training', '10% Testing'
# REPEATING with DIFFERENT Training-Testing Samples:
# - Repeated Results are CLOSE to 'TRUE GENERALIZATION ERROR' (=HIGH 'ACCURACY')
# but EACH Result is DISTINCT from EACH OTHER = POOR 'PRECISION'!
#If LARGER SAMPLE for 'TESTING' = POOR ACCURACY (further from 'generalization error'), but HIGHER PRECISION (more close together)

#But HOW can we OVERCOME this ISSUE?


#                Use 'CROSS VALIDATION' 
#= COMMON METRIC for 'OUT-OF-SAMPLE' EVALUATION
#- SPLIT DATASET into 'k' EQUAL GROUPS
# - EACH Group is called a 'FOLD'  (e.g. 4 Folds...)
# - SOME 'FOLDS' can be USED as a 'TRAINING SET' (train model)
# - LEFT-OVER 'Folds' used to 'TEST'

#Example - 3 Folds for Training, 1 for Testing 
#REPEAT until 'EACH PARTITION'/FOLD has been USED for BOTH 'Testing' AND 'Training'
#Use 'AVERAGE' RESULT to ESTIMATE 'OUT-OF-SAMPLE ERROR'

#'CROSS VALIDATION SCORE' FUNCTION given as 'cross_val_score()'
#This PERFORMS MULTIPLE 'Out-Of-Sample' EVALUATIONS:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lr, x_data, y_data, cv=3)
#1st Argument = 'MODEL USED' (e.g. above, is 'Linear Regression' lr)
#'cv' = 'NUMBER of PARTITIONS'
scores     # = ARRAY of 'R-Squared Values' for 'EACH PARTITION/FOLD'
np.mean(scores)  #THEN can find 'AVERAGE R-Squared' using 'np.mean(scores)'
#(NOTE - 'DEFAULT' SCORE Metric = 'R-Squared', but can use 'scoring=""' to SPECIFY ANOTHER METRIC (like 'MSE'..))


#e.g. - So have 3 FOLDS - 2 for Training, 1 for Testing
#Model will PRODUCE an 'OUTPUT' 
#USE the OUTPUT to CALCULATE a 'SCORE'
# DEFAULT 'SCORE' Metric = 'R-Squared', STORED in an ARRAY (estimated the 'OUT-OF-SAMPLE' R-Squared with 'MEAN' Function)
#REPEAT AGAIN, EACH TIME using DIFFERENT COMBINATIONS of - '2 Folds Training, 1 Testing' 
#'scores' STORES the ARRAY of 'R-Squared' for 'EACH COMBINATION' we PERFORM!
#'cross_val_score' function DOES THIS! - Returns 'CROSS VALIDATION RESULT'!

#           Cross-Validation 'PREDICTIONS':
#What if we want to know 'ACTUAL PREDICTED Y-VALUES' (yhat)? (BEFORE the 'R-Squared Value' METRIC is CALCULATED)
from sklearn.model_selection import cross_val_predict
yhat = cross_val_predict(lr2e, x_data, y_data, cv=3)
#Exact 'SAME INPUTS' as 'cross_val_score'!
#But INSTEAD, GIVES us 'PREDICTED Y-VALUES' as OUTPUT!

#e.g. - SPLIT Data into 3 Folds, 2 for TRAIN, 1 for TEST
#MODEL produces OUTPUT, STORED in an ARRAY
#REPEAT PROCESS using 2 Folds Training, 1 Fold Testing
#OUTPUT is PRODUCED AGAIN, for EACH COMBINATION we REPEAT
#Outputs are STORED in an ARRAY (='PREDICTED'-VALUES)




#              'OVERFITTING', 'UNDERFITTING' - 'MODEL SELECTION'

# = PICKING the 'BEST ORDER' for 'POLYNOMIAL' REGRESSION?

#e.g. ASSUME 'TRAINING POINTS' are FROM a 'POLYNOMIAL' FUNCTION
#Given as:        'y(x) + noise'  
#(Note -  will ALWAYS get SOME Random 'NOISE' around the Fit of the Data = 'IRREDUCIBLE Error')

#WHICH ORDER Polynomial produces BEST ESTIMATE of 'y(x)'?

#Trying 'LINEAR Function' - 'y = b0 + b1*x'
# - would NOT FIT DATA! NOT COMPLEX Enough!
#              = 'UNDERFITTING'
#INCREASING 'ORDER' to 'POLYNOMIAL' = BETTER, but STILL NOT ENOUGH! - STILL 'UNDERFITTING'

#Increasing to '8th ORDER' Polynomial = FITS data WELL and ESTIMATES FUNCTION WELL (even at ‘inflection points’= ends of line)
#Increasing FURTHER to '16th ORDER' - TRACKS POINTS WELL
# - BUT gives 'poor estimates' of 'FUNCTION'!
# - Just OSCILLATES, doesn’t TRACK Function well!
#    = 'OVERFITTING'!!! (=fits 'NOISE', NOT the 'FUNCTION' ITSELF!!! 'y(x)')
#                     (i.e. follows the Individual Points TOO CLOSELY, NOT the FUNCTION OVERALL)

#Plotting 'MSE' vs. DIFFERENT 'ORDERS' for 'Testing' and 'Training' Sets:
#AS 'ORDER' Increases:
#For 'TRAINING' Set - 'MSE ERROR' DECREASES (Not Useful!)
#For 'TESTING' Set - MSE ERROR starts by decreasing, UNTIL the BEST ORDER is found (LOWEST DIP in MSE), then MSE 'INCREASES' 
#'TEST' Error = 'BEST Way' to ESTIMATE 'ERROR' of Polynomial
#So just SELECT the 'ORDER' which MINIMIZES 'ERROR'

#Note: will STILL have SOME ERROR (y(x) + 'noise' - RANDOM and UNPREDICTABLE!)
# = 'IRREDUCIBLE' ERROR

#Maybe our 'POLYNOMIAL' Assumption is just WRONG!
#e.g. Sample Points are from 'SINE WAVE' INSTEAD of 'Polynomial'
#-Polynomial Function DOES NOT FIT the SINE WAVE well!

#For 'REAL' DATA - Model may be TOO DIFFICULT to FIT/WRONG DATA...
# (SEE 'WORD FILE NOTES' - Great 'REAL-WORLD' Example)


#CALCULATE 'R-Squared' Values for 'DIFFERENT ORDERS' (using 'TEST' Data):
Rsqu_test  = []   
order = [1,2,3,4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    lr.fit(x_train_pr, y_train)
    Rqu_test.append(lr.score(x_test_pr, y_test))
#just LOOPED for EACH 'order' polynomial:
# - created 'pr=PolynomialFeatures' Object/Instance
# - 'TRANSFORMED' Training and Test Data INTO POLYNOMIAL
# - FIT our 'Linear Regression' Model USING the Transformed 'TRAINING' Data
# - Then CALCULATE 'R-Squared' USING 'TEST' Data and STORE it in an ARRAY/LIST!
#SIMPLE!



#                       'RIDGE REGRESSION'
#Models with MULTIPLE 'Independent Features' (Variables)
#AND those with 'POLYNOMIAL Feature' Extrapolation
#these COMMONLY have 'COLINEAR' COMBINATIONS of Features
# - ALL THIS can lead to 'OVERFITTING' of TRAINING Data!

# -So? 'REGULARIZED' by using 'HYPERPARAMETERS', like 'ALPHA'

#'RIDGE REGRESSION' = 'REGULARIZING' Feature Set
#How? - USING the 'ALPHA' HYPERPARAMETER
#-this lets us AVOID 'Over-Fitting' and REDUCE 'Standard ERRORS' for Regression Models

#Here, lets look at 'POLYNOMIAL REGRESSION' 

    #(SEE WORD FILE for EVERYTHING aqbout 'ALPHA' Parameter!)

#'LARGER ALPHA' = 'BETTER FIT', BUT....
#IF Alpha is TOO LARGE, can get 'UNDERFITTING'!!! 
#- Must AVOID this by selecting 'BEST ALPHA':

#SELECT 'BEST ALPHA' Value by 'CROSS-VALIDATION':
# - use Some TRAINING DATA
# - use SECOND SET, called 'VALIDATION DATA' 
# ('Validation Data'= similar to 'TEST' data, but used to Select 'PARAMETERS', like 'Alpha' here!)
# - START from SMALLEST ALPHA Value (e.g. 0.1)
# - TRAIN the MODEL, then PREDICT (using this 'Validation Data)
# - Finally, CALCULATE 'R-Squared' and STORE the Values 
# - REPEAT for EACH VALUE of 'Alpha' (1 and 10)
#FINALLY - SELECT 'Alpha' which GIVES MAXIMUM 'R-Squared' Value
#(Note: could have used 'MSE' as metric to choose Alpha TOO. R-Squared is just ONE POSSIBLE Metric!)


#'Ridge Regression' in PYTHON:
from sklearn.linear_model import Ridge
RidgeModel = Ridge(alpha = 0.1)  #here, chose '0.1' as 'alpha' = ARGUMENT of the CONSTRUCTOR/Object
RidgeModel.fit(x_train,y_train)    #'TRAIN' the MODEL
Yhat = RidgeModel.predict(x_test)   #MAKE PREDICTION (using 'TEST' Data)!

#'OVERFITTING' is EVEN WORSE when we have MULTIPLE FEATURES
#e.g. Have a POLYNOMIAL Regression Model with MULTIPLE FEATURES!
#Plot 'R-Squared vs. Alpha' 
# - AS ‘Alpha’ INCREASES, ‘R-Squared’ Value INCREASES, before it eventually CONVERGES at around ‘0.75’
# - Select 'BEST' Alpha Value 'POSSIBLE
#(i.e. Alpha which Produces HIGHEST POSSIBLE 'R-Squared')
#Note: will GENERATE this Plot in PRACTICE Labs BELOW!

#Note - 'OVERFITTING' is ALSO a problem when JUST MULTIPLE 'INDEPENDENT VARIABLES/FEATURES', NOT JUST for 'POLYNOMIAL' Model!



#                        'GRID SEARCH'
#=way to SCAN through MULTIPLE Free PARAMETERS very EASILY!
#'HYPER'-PARAMETERS (like 'Alpha') = NOT PART of FITTING/TRAINING PROCESS
#Grid Serach = 'ITERATE OVER' these 'Hyperparameters' with 'CROSS-VALIDATION'
#TAKES Model/Objects you want to TRAIN, and 'DIFFERENT VALUES' of Hyperparameters
#Then CALCULATES 'MSE' or 'R-Squared' for DIFFERENT Hyperparameter VALUES
# = this way, can CHOOSE the 'BEST VALUES'

# - CONTINUE Testing DIFFERENT HYPERPARAMETERS until have used ALL Free Parameter VALUES
# - EACH Model PRODUCES an 'Error' 
# - SELECT Model with MINIMUM ERROR (MSE...)

#SPLIT Dataset into 3 Parts: 'TRAINING', 'VALIDATION' and 'TEST' Sets
# 1. 'TRAIN' MODEL for DIFFERENT HYPERPARAMETERS
# 2. Select Hyperparameter which MINIMIZED 'MSE' or MAXIMISES 'R-Squared' on 'VALIDATION SET'
# 3. Finally, 'TEST' Model PERFORMANCE, using TEST Data

#(Note - 'Attributes' of a Class Object are ALSO called 'Parameters' - DON'T get CONFUSED!)
#HERE, will focus on 'ALPHA' and 'NORMALIZATION' Parameters:
parameters = [{'alpha':[1,10,100,1000]}] #stored as LIST of 'DICTIONARY' for EACH Parameter
#Also have our MODEL - here 'Ridge()'

#   GRID SEARCH in PYTHON:
#TAKES IN 'Scoring Method' (i.e. 'R-Squared' or 'MSE')
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
parameters1 = [{'alpha':[0.001, 0.1, 1, 10, 1000, 10000, 100000, 100000]}]
#creating 'Ridge Regression OBJECT/Model'
RR = Ridge()   
#Create 'GridSearchCV' Object/Model (Inputs are 'RR', 'Parameter Values' and cv = Number of Folds)
Grid1 = GridSearchCV(RR, parameters1, cv=4)
#Note - 'R-Squared' is DEFAULT 'SCORING METHOD'
#FIT the 'Grid1' Object TO our 'DATA' - 'Grid.fit(x_data, y_data)'
Grid1.fit(car_df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], car_df['price'])
#Now Finding 'BEST VALUES' for 'Free Parameters' - using '.best_estimator_' attribute
Grid1.best_estimator_  
#Tells us 'alpha=1000' is BEST ESTIMATOR!
#EVEN get 'MEAN SCORE' of 'VALIDATION DATA' - using '.cv_results_' attribute
scores = Grid1.cv_results_    
scores         #get the SCORES as 'DICTIONARY'
scores['mean_test_score']   #returns 'ARRAY' of MEAN SCORES!
#(just got 'Values' for 'mean_test_score' 'Key')!


#'Ridge Regression' ALSO lets us 'NORMALIZE' the DATA:
parameters = [{'alpha':[1,10,100,1000], 'normalize':[True,False]} ]
#(Just added 'normalize' Parameter TOO! - True = NORMALIZE, False = NOT NORMALIZE)
#Essentially get DIFFERENT COMBINATIONS for 'Alpha' and 'Normalize' TOGETHER
# = Table/GRID of DIFFERENT 'PARAMETER VALUES' (See Word Notes for what this Table LOOKS Like!)
#       THE CODE IS VERY SIMILAR to ABOVE!!!

#Can 'PRINT OUT SCORES' for 'DIFFERENT Free PARAMETER VALUES':
for param, mean_val, mean_test in zip(scores['params'], scores['mean_test_score']), scores['mean_train_score']:
    print(param, "R^2 on test data", mean_val, "R^2 on train data:", mean_test)
#Will cover ALL this in the LABS (Below!)



#%%            'MODEL EVALUATION' - PRACTICE 1  (Car Dataset)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
car_df = pd.read_csv(filepath)
car_df.head()

#Here are 2 USER-DEFINED FUNCTIONS we will USE for PLOTTING:
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    #'RedFunction' = ACTUAL y-data, 'BlueFunction' = PREDICTED y-Data
    width = 12
    height = 10   #just specifying specific SIZE of PLOT
    plt.figure(figsize=(width, height))
    #PLOTTING the 2 'kdeplots' for the 2 Functions:
    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)
    #(Note: 'kdeplot' = 'KERNEL-Distribution-Estimation' Plot, SIMILAR to 'Histogram' Distribution. Used for 'CONTINUOUS PROBABILITY DENSITY' Curve)
    #('kdeplot' is just ALTERNATIVE to 'distplot'!)
    
    #Just plotting the LABELS:
    plt.title(Title)      
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_model):
    width = 12
    height = 10   
    plt.figure(figsize=(width, height))
    #training data, testing data
    #lr (=linear regression object), poly_transform (=polynomial Transformation Object)
    
    #Create the 'x-axis' Scale for 'PREDICTED Function'
    xmax = max([xtrain.values.max(), xtest.values.max()])
    xmin = min([xtrain.values.min(), xtest.values.min()])
    x = np.arange(xmin, xmax, 0.1)  #Create 'x_axis SCALE'
    
    #PLOTTING 'Training Data' AND 'Testing Data' AND 'PREDICTED Function' (as 'POLYNOMIAL FIT')
    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_model.fit_transform(x.reshape(-1,1))), label = 'Predicted Function')
    #Note - above, just plots as 'POLYNOMIAL' (for 'Predicted Values' - 1. Converts to Polynomial, then 2. Fits with 'LINEAR' Regression)
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')

#           1. 'TRAINING' and 'TESTING':     
#Must SPLIT Data into 'TRAINING' and 'TESTING' Data
#Place 'TARGET y-variable' (price) in SEPARATE DATAFRAME:
y_data = car_df['price']
#DROP 'price' Column from OG Dataframe to Create 'x_data' Dataframe:
x_data = car_df.drop('price', axis=1)
#'RANDOMLY SPLIT' this data into 'TESTING' and 'TRAINING':
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10,random_state=1 )
print(f"Test Samples - {x_test.shape[0]}")       #'21' Samples
print(f"Training Samples - {x_train.shape[0]}")  #'180' samples
#Chose 'test_size' = '10%' of 'TOTAL DATASET'

#NOW, setting 'test_size' = 40% (just for practice!)
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.40,random_state=0 )
print(f"Test Samples - {x_test1.shape[0]}")       #'21' Samples
print(f"Training Samples - {x_train1.shape[0]}")  #'180' samples

#FIT 'LINEAR REGRESSION MODEL' to 'TRAINING' DATA:
from sklearn.linear_model import LinearRegression
lre = LinearRegression()
lre.fit(x_train[['horsepower']], y_train)   #SIMPLE Linear Regression for 'horsepower' Predictor Variable
#Calculating R-Squared for 'TESTING DATA'
lre.score(x_test[['horsepower']], y_test)  # ='0.3636'
#(Note -For 'Training Data' is Much 'LARGER R-Squared'):
lre.score(x_train[['horsepower']], y_train) # ='0.662'

#What is 'R-Squared' if '40% of Data' used for TEST?
lre.fit(x_train1[['horsepower']], y_train1)   #SIMPLE Linear Regression for 'horsepower' Predictor Variable
#Calculating R-Squared for 'TESTING DATA'
lre.score(x_test1[['horsepower']], y_test1)  # ='0.7139'
#IF MORE of DATA for 'TEST' = LARGER R-Squared (=BETTER FIT!)


#IF we DONT have ENOUGH 'TESTING DATA'?
#Can find 'CROSS VALIDATION' SCORE ('R-Squared' Scores):
from sklearn.model_selection import cross_val_score
#Using 'Linear Regression Model' on 'horsepower' feature, for cv = 4 
scores = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
scores   #=Array of Different 'R-Squared' Values, Calculated for EACH PARTITION/FOLD!
#Calculate AVERAGE and SD of 'Folds':
np.mean(scores)     #='0.522'
np.std(scores)      #='0.291' (spread around mean)

#Using 'ANOTHER METRIC' than 'R-Squared' (DEFAULT):
#INSTEAD, could use 'Negative MSE' 
#just SPECIFY 'scoring="neg_mean_squared_error"' as ARGUMENT:
MSE_scores = -1*cross_val_score(lre, x_data[['horsepower']], y_data, cv=4, scoring="neg_mean_squared_error")
MSE_scores   #ARRAY of MSE Values for EACH PARTITION/FOLD - SAME THING!!

#Calculating 'AVERAGE R-Squared' using '2 FOLDS' (cv=2):
two_fold_scores = cross_val_score(lre, x_data[['horsepower']], y_data, cv=2)
#(cv = 2). Viewing 'Average Score' (R-Squared):
np.mean(two_fold_scores)   # = 0.5167

#Could PREDICT OUTPUT using 'cross_val_predict':
from sklearn.model_selection import cross_val_predict
yhat = cross_val_predict(lre, x_data[['horsepower']], y_data, cv=4)
#here, did for 'cv = 4 folds', with 'horsepower' Variable 
yhat[0:5]    #gives us PREDICTED Y-VALUES as OUTPUT!



#             'OVERFITTING, UNDERFITTING, MODEL SELECTION':
#'TEST DATA' (out of sample data) = BETTER MEASURE of PERFORMANCE of Data 'in REAL-WORLD'
#Why? due to OVERFITTING
lr1 = LinearRegression()
#FIT the Model, to TRAIN 'MULTIPLE LINEAR' Regression Model:
lr1.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
#Make PREDICTION using 'TRAINING' Data:
yhat_train = lr1.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_train[0:5]   
#REPEATING 'PREDICTION' for 'TEST' Data:
yhat_test = lr1.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_test[0:5]        

#Plotting DISTRIBUTION of 'TRAINING' Data (Actual vs. Predicted):
Title = "Distribution Plot of 'Predicted Value Using Training Data (yhat)' vs 'Training Data (y) Distribution'"
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
#Plotting DISTRIBUTION of 'TESTING' Data (Actual vs. Predicted):
Title = "Distribution Plot of 'Predicted Value Using Testing Data (yhat)' vs 'Testing Data (y) Distribution'"
DistributionPlot(y_test, yhat_test, "Actual Values (Test)", "Predicted Values (Test)", Title)

#SEE that 'TRAIN' Data FITS the Model BETTER! 'TEST' Data is very POOR FIT of 'Predictions' to 'Actual' Data!!!

#            WHAT about 'POLYNOMIAL REGRESSION Model'?
#(does 'Polynomial' ALSO have Worse Fit for 'TEST' Data?)
from sklearn.preprocessing import PolynomialFeatures
#'OVERFITTING' - when Model FITS 'NOISE', INSTEAD of the FUNCTION itself!
#Now doing '55% Training' vs. '45% TEST' SPLIT
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45,random_state=0)
#Example - Create 'Order/degree = 5' Polynomial Model:
pr = PolynomialFeatures(degree=5)
pr     #NOW performing 'POLYNOMIAL TRANSFORM':
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
#Create 'LINEAR REGRESSION' MODEL 'FOR' this 'Polynomial'
poly = LinearRegression()
poly.fit(x_train_pr, y_train)
yhat = poly.predict(x_test_pr)    
print(f"Predicted Values: {yhat[0:4]}")      #calculating PREDICTED VALUES for this Linear (Polynomial) Model!
print(f"Actual Values: {list(y_test[0:4])}")  #ACTUAL Values to COMPARE to PREDICTED
#COMPARING 'Predicted' to 'Actual' Values
#BEST SHOWN with 'PollyPlot' (defined ABOVE!):
PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly, pr)
#red dots = TRAINING Data, green dots = TESTING Data
#POLYNOMIAL CURVE appears 'OVERFITTED!'

#Comparing 'R-Squared' of 'TRAINING vs. TEST':
poly.score(x_train_pr, y_train)   #'0.5568'
poly.score(x_test_pr, y_test)     #'-29.87'
#PROVES that 'TEST' Data is 'OVERFITTED'!!! 
#NEGATIVE R-Squared = SIGN of 'OVERFITTING'

#LOOPING to see POLYNOMIAL 'R-Squared' at DIFFERENT ORDERS:
lr = LinearRegression()
rsqu_test = [ ]   
order = [1,2,3,4,5]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    lr.fit(x_train_pr, y_train)
    rsqu_test.append(lr.score(x_test_pr, y_test))
print(f"R-Squared for 1st, 2nd, 3rd adn 4th Order Polynomials: {rsqu_test}")

#PLOTTING THESE for EACH ORDER (Easy!):
plt.plot(order, rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.test(3, 0.75, "Maximum R^2")
#See that, AFTER '3rd Order', get 'OVERFITTING' (DRAMATICALLY LOWER 'R-Squared' Value'!

#NOW, 'POLYNOMIAL' Model, with 'MULTIPLE FEATURES' (SAME EXACT PROCESS!):
pr1 = PolynomialFeatures(degree=2)
x_train_pr1 = pr1.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_train_pr1.shape  #NOW have '15 Features (Dimensions)'
x_test_pr1 = pr1.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
poly1 = LinearRegression()
poly1.fit(x_train_pr1, y_train)      #EXACT SAME PROCESS!
yhat = poly1.predict(x_test_pr1)    
print(f"Predicted Values: {yhat[0:4]}")      
print(f"Actual Values: {list(y_test[0:4])}")  
#Plotting 'TEST' DISTRIBUTION of 'ACTUAL' vs. 'PREDICTED' (yhat) Values:
Title = "'Test Data' Distribution Plot of Predicted Values vs Test Data Distribution"
DistributionPlot(y_test, yhat, "Actual (Test) Values", "Predicted (Test) Values", Title)
#Mostly is PRETTY GOOD FIT! But we see that around '$5000 - $15000', have slight OVER ESTIMATION of 'PRICE' for Blue (Predicted) Distribution
#ALSO at around '$30,000 - $40,000' have slight DIP/UNDER-ESTIMATION of Blue (Predicted) Distribution = LOWER PRICE than EXPECTED.



#               'RIDGE REGRESSION'
#Models with MULTIPLE FEATURES and POLYNOMIAL 
# = MORE LIKELY to get 'OVERFITTING'!!!
#So? can introduce 'ALPHA' Parameter to 'PREVENT OVERFITTING'!
from sklearn.linear_model import Ridge
#Create our 2nd Order POLYNOMIAL Model (with 'MANY FEATURES'):
pr = PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

#Create 'RIDGE REGRESSION' Object (using 'alpha = 1')    
RidgeModel = Ridge(alpha = 1)
#FITTING 'Ridge Model' (using 'fit' AS USUAL!)
RidgeModel.fit(x_train_pr,y_train)    #'TRAIN' the MODEL
Yhat = RidgeModel.predict(x_test_pr)  
#MADE PREDICTION (using 'TEST' Data)! Now COMPARING:
print(f'predicted: {yhat[0:4]}')
print(f'test set: {list(y_test[0:4])}')

#HOW can we Select 'Alpha' Value to 'MINIMIZE TEST ERROR' (Overfitting)?
#Create 'FOR LOOP' to 'TEST' for RANGE of 'alpha' Values:
from tqdm import tqdm   #lets us get 'PROGRESS BAR' (see HOW MANY Iterations are Complete SO FAR..) 
Rsqu_test = []   #='Validation' Data!
Rsqu_train = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)  #'Progress Bar' is NOT NECESSARY - But is PRETTY COOL WAY to TRACK our ITERATIONS!

for alpha in pbar:
    #Create RidgeModel (AS USUAL! Just ITERATING through DIFFERENT 'alpha' Values!)
    RidgeModel = Ridge(alpha = alpha)
    RidgeModel.fit(x_train_pr, y_train)
    #Use 'RidgeModel.score()' to find R-squared Value!
    test_score, train_score = RidgeModel.score(x_test_pr, y_test), RidgeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score":test_score, "Train Score":train_score})
    
    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)
    
#PLOT OUT 'R-Squared' Value for 'DIFFERENT Alphas':
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()     #(SAME as PLOT from 'WORD NOTES'!)
#Shows that INCREASE in ALPHA - INCREASE 'R-Squared' in Validation Data
#BUT CONVERGES to a MAX R-Squared Value - this indicates a 'MAX ALPHA'!
#AS Alpha Increases, 'TRAINING' Data DECREASES (= WORSE Performance on 'Training')

#ANOTHER EXAMPLE - Finding 'R-Squared' for 'Alpha=10'
RidgeModel = Ridge(alpha = 10)
RidgeModel.fit(x_train_pr, y_train)    
RidgeModel.score(x_test_pr, y_test)
# = '0.5419'


#            GRID SEARCH (SAME EXAMPLE as 'NOTES' ABOVE!) - Just REINFORCING the PROCESS!
#Using 'GridSearchCV' Class 
#=makes FINDING the BEST 'HYPERPARAMETER' (like 'Alpha') MUCH EASIER!
from sklearn.model_selection import GridSearchCV
parameters1 = [{'alpha':[0.001, 0.1, 1, 10, 1000, 10000, 100000, 1000000]}]
#creating 'Ridge Regression OBJECT/Model'
RR = Ridge()   
#Create 'GridSearchCV' Object/Model (Inputs are 'RR', 'Parameter Values' and cv = Number of Folds)
Grid1 = GridSearchCV(RR, parameters1, cv=4)
#Note - 'R-Squared' is DEFAULT 'SCORING METHOD'

#FIT the 'Grid1' Object TO our 'DATA' - 'Grid.fit(x_data, y_data)'
Grid1.fit(car_df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], car_df['price'])

#Now Finding 'BEST VALUES' for 'Free Parameters' - using '.best_estimator_' attribute
BestRR = Grid1.best_estimator_  
BestRR   #Tells us 'alpha=1000' is BEST ESTIMATOR!

#EVEN get 'MEAN SCORE' of 'VALIDATION DATA' - using '.cv_results_' attribute
scores = Grid1.cv_results_    
scores    #= DICTIONARY of SCORES (R-Squared' Values here) 
scores['mean_test_score']   #returns 'ARRAY' of MEAN SCORES!


#for 'BestRR', TEST the Model on 'TEST DATA':
# (get 'R-Squared')
BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)
#  = 0.84112   (BEST 'R-Squared' Value!)

#NOW Performing 'Grid Search' (Just More Practice!!)
#for 'ALPHA' Parameter AND 'NORMALIZATION PARAMETER':
parameters2 = [{'alpha':[0.001, 0.1,1, 10,100,1000, 10000, 100000]}]
RR = Ridge()
Grid2 = GridSearchCV(RR, parameters2, cv=4)
Grid2.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
BestRR2 = Grid2.best_estimator_
BestRR2    #'Ridge(alpha = 10000)', Normalize = True is BEST COMBINATION!
#Finding 'BEST R-Squared Value (for 'BEST ALPHA' Object) 
BestRR2.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)

best_ridge_model = Ridge(alpha = 10000)
best_ridge_model.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)


#%%          'MODEL EVALUATION' - PRACTICE 2  (Laptop Dataset)
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv'
laptop_df = pd.read_csv(filepath)
laptop_df.head()   #'unamed' columns NOT NEEDED - so DROP EM!
laptop_df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)

#     1. CROSS VALIDATION to IMPROVE MODEL:
#DIVIDE Dataset into 'x_data' and 'y_data' Parameters:
y_data = laptop_df['Price'] 
x_data = laptop_df.drop('Price', axis=1)
#SPLITTING Data so '10% is used for TESTING':
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10,random_state=0 )
print(f"Test Samples - {x_test.shape[0]}")       #'24' Samples
print(f"Training Samples - {x_train.shape[0]}")  #'214' samples

#Creating 'SINGLE Linear Regression' for 'CPU_frequency':
from sklearn.linear_model import LinearRegression
lre = LinearRegression()
lre.fit(x_train[['CPU_frequency']], y_train)   #use 'TRAINING' Data to FIT MODEL!
#Calculating R-Squared for 'TESTING' and 'TRANINING' 
lre.score(x_test[['CPU_frequency']], y_test)  # ='-0.0973'
lre.score(x_train[['CPU_frequency']], y_train) # ='0.1472'

#Run '4 FOLD' CROSS VALIDATION to find 'MEAN R-Squared' and SD:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lre, x_data[['CPU_frequency']], y_data, cv=4)
scores   #=Array of Different 'R-Squared' Values, for 'EACH PARTITION/FOLD'!
#Calculate AVERAGE and SD of 'Folds':
np.mean(scores)     #='-0.1611'
np.std(scores)      #='0.385' (spread around Mean)
#So Data is pretty 'POOR FIT' of 'LINEAR' Regression Model

#                'OVERFITTING' 
#SPLIT Data into '50% Testing' now:
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.50,random_state=0 )
#Creating 'POLYNOMIAL REGRESSION MODEL' to IDENTIFY 'OVERFITTING' in the Model
from sklearn.preprocessing import PolynomialFeatures
#LOOPING to identify 'R-Squared' Score for 'Order 1-5':
lr = LinearRegression()
rsqu_test = [ ]   
order = [1,2,3,4,5]
for n in order:  #Usual Steps of Polynomial Regression!
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['CPU_frequency']])
    x_test_pr = pr.fit_transform(x_test[['CPU_frequency']])
    lr.fit(x_train_pr, y_train)  #FIT using 'TRAIN' Data
    rsqu_test.append(lr.score(x_test_pr, y_test)) #R-Squared Values, appended to List 
print(f"R-Squared for 1st, 2nd, 3rd, 4th and 5th Order Polynomials: {rsqu_test}")

#PLOTTING 'R-Squared' Scores vs. 'order'
plt.plot(order, rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
#Get DRAMATIC DECLINE in 'R-Squared' Value after '3rd Order'



#               'RIDGE REGRESSION'
#NOW will work with 'MULTIPLE FEATURES' for a POLYNOMIAL MODEL:
pr1 = PolynomialFeatures(degree=2)
x_train_pr = pr1.fit_transform(x_train[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']])
x_test_pr = pr1.fit_transform(x_test[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']])
poly1 = LinearRegression()
poly1.fit(x_train_pr, y_train)      
yhat = poly1.predict(x_test_pr)     #'predicted' Values (yhat)

#Create a 'RIDGE REGRESSION MODEL':
# - Evaluate 'Alpha' from '0.001 to 1' with 'Increments of 0.001':
Rsqu_test = []   #='Validation' Data!
Rsqu_train = []
Alpha = np.arange(0.001,1, 0.001)
#SIDE NOTE - 'np.arange' lets us use 'FLOATS'
# (wheras 'range' is ONLY for INTEGERS! Hence why we use 'np.arange()' here!)
pbar = tqdm(Alpha)  #'Progress Bar' is NOT NECESSARY - But is PRETTY COOL WAY to TRACK our ITERATIONS!

for alpha in pbar:
    #Create RidgeModel (AS USUAL! Just ITERATING through DIFFERENT 'alpha' Values!)
    RidgeModel = Ridge(alpha = alpha)
    RidgeModel.fit(x_train_pr, y_train)
    #Use 'RidgeModel.score()' to find R-squared Value!
    test_score, train_score = RidgeModel.score(x_test_pr, y_test), RidgeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score":test_score, "Train Score":train_score})
    
    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

#PLOTTING 'R-Squared' for 'Training and Testing' Sets vs. 'Alpha':
plt.figure(figsize=(10, 6))
plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()     #(SAME as PLOT from 'WORD NOTES'!)

    
#              'GRID SEARCH' (Nothing New!)
parameters = [{'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10 ]}]
RR = Ridge()
Grid = GridSearchCV(RR, parameters, cv=4)
Grid.fit(x_data[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']], y_data)
BestRR = Grid.best_estimator_
BestRR    
best_ridge_model = Ridge(alpha = 0.0001)
best_ridge_model.fit(x_data[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']], y_data)
best_ridge_model.score(x_test[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']], y_test)
#  '= 0.44124', given as BEST 'R-Squared' for 'BEST ALPHA' (with LEAST 'UNDERFITTING' - BEST FIT!)






















