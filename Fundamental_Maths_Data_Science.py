# -*- coding: utf-8 -*-
"""
Created on Mon May  8 00:11:26 2023

@author: Ezhan Khan
"""

#%%    PROBABILITY
#Measure/Quantify UNCERTAINTY (Industries - Meteorology, Medicine, Sports, Insurance....)
#SETS = COLLECTIONS of things, order does not matter (but only 'UNIQUE' ELEMENTS are allowed)  
     # e.g.  A = {1,2,3,4,5}
#SUBSET = All elements in Subset are ALSO in SET
     # e.g.   B = {1,2,3}    'B' is SUBSET of A
#EXPERIMENT Example (=produce Observations with LEVEL of UNCERTAINTY)
      #e.g. Flip Coin TWICE, H or T:   S = {HH, TT, HT, TH}
# here, 'S' is Sample Space (set of all possible Outcomes)    Each Possible Outcome is called 'Sample Point'
#EVENT = Specific Outcome/Set of Specific Outcomes
      #e.g.  A = {HH}  or C = {HT, TH}
#FREQUENTIST Probability = THEORETICAL Probabiliity, where run INFINITE TIMES. Probability of EACH Event = PROPORTION of Times Occurs
#ESTIMATED Probablity (=PROPORTION of TImes an Event Occurs)
#     P(Event) = Number of TImes Event Occured/Total Trials
      #e.g.  say 'HH' happened 252 times OUT of 1000. THis is just  252/1000 = 0.252 Probability 

# LAW OF LARGE NUMBERS = Do MORE TRIALS means MORE LIKELY to CONVERGE to TRUE Probability
import random
from matplotlib import pyplot as plt
#SIMULATING the Coin Flip (HH):
def coinflip():
    coin1 = ['Heads', 'Tails']
    coin2 = ['Heads', 'Tails']
    coin1_result = random.choice(coin1)
    coin2_result = random.choice(coin2)
    if coin1_result == 'Heads' and coin2_result =='Heads':
        return 1
    else:
        return 0
Trials = 100000
prop = []            #Proportion of HH at EACH STAGE
flips = []           #just keep track of number of flips done (i.e. EACH ITERATION of Loop Below)  
heads_counter = 0
for flip in range(Trials):
    heads_counter += coinflip()     #if Heads and Heads, will return '1' so ADDS to COUNT! Otherwise is just '0'
    prop.append(heads_counter/(flip+1))   #'+1', since index is 1 less (range starts from '0')
    flips.append(flip+1) 
print(heads_counter)         #RANDOM Selection, so CHANGES Every time.
print(prop)

plt.plot(flips, prop, label="Experimental Probability")
plt.xlabel("Number of Flips")
plt.ylabel("Proportion of HH")
plt.hlines(0.25, 0, Trials, colors = 'red', label = 'True Probability')
plt.legend()
plt.show()        #'hlines' is just HORIZONTAL LINE, where we defined 'y point', THEN x-axis Range.
#TRUE PROBABILITY indicated by Horizontal Line = 1/2 * 1/2 = 0.25 (Makes Sense!)
#INCREASING TRIALS will mean CONVERGES TO the TRUE Probability 
#       This PROVES the Law of Large Numbers!!!
#%%    RULES OF PROBABILITY
#What about MULTIPLE Random Events? - Flipping a Coin AND Rolling a Die Together - What is PROBABILITY of 'tails' AND '5 on Die'
#UNION (A OR B)= 2 SETS, EITHER ONE OR BOTH
     #e.g. Set A = {1,3,5} rolling ODD Number on 6 sided Die
     #     Set B = {3,4,5,6} Number GREATER THAN 2
     #so 'Union' = Either A, B OR BOTH
#INTERSECTION (A AND B) = 2 Sets, IN BOTH SETS
     #e.g. this would be '{3,5}' for above example
#COMPLEMENT of a Set = ALL OUTCOMES OUTSIDE OF SET
     #e.g. For Set A = ALL ODD, 'Complement' = EVEN {2,4,6} - EASY!
#Easy Example - A=die less or equal to 3 {1,2,3} and B={1,2,3,4,6}
#This gives INTERSECTION = {1,2,3}, UNION = {1,2,3,4,6}

#  INDEPENDENT EVENTS
#  = Occurance of ONE EVENT DOES NOT AFFECT PROBABILITY OF OTHER
     #e.g.  Flip Coin 5 TImes and get 5 Heads = JUST CHANCE! PAST Flip CANNOT Influence any FUTURE Flip!
#  DEPENDENT EVENTS 
#  = Case where PREVIOUS Events DO Affect OUTCOME of NEXT Event
    #e.g. Bag of 5 marbles (2 Blue, 3 Red) 
    #If we TAKE OUT ONE Marble 'WITHOUT REPLACEMENT', 
    #the Probabiltiy that SECOND MARBLE is BLUE WILL DEPEND Upon WHAT Color Marble is Taken out in FIRST EVENT!
    #BUT, IF we 'REPLACE' the 1st Marble = INDEPENDENT

#Example 1 - 52 Cards DECK (26 Red = 13 Diamonds, 13 Hearts, 26 Black = 13 Clubs, 13 Spades. EACH '13' has 3 FACE Cards (King, Queen, Jack), 1 Ace, 9 Numbered Cards (2-10) )
#Pick 2 Cards out WITHOUT Replacement = DEPENDENT! P(Ace on 1st Draw) = 4/52. P(Ace on 2nd Draw) = 3/51 IF Ace in 1st Draw and = 4/51 IF NOT Ace in 1st Draw.
#Example 2 - Roll Die Twice. P(3 on 1st) = 1/6. P(3 on 2nd - 1/6) = INDEPENDENT     

#   MUTUALLY EXCLUSIVE 
#   = 2 events which CANNOT OCCUR AT SAME TIME (Imagine as 2 Circles NOT OVERLAPPING - NO INTERSECTION!)
#   #e.g. Events 'Heads' and 'Tails' - CANNOT Occur at same time of course!!
#Example 1 - (NOT Mutually Exclusive) Die Roll, Event A  = ODD, Event B = Greater than 4. CAN be BOTH Greater than 4 AND ODD
#Example 2 - (Mutually Exclusive) Die Roll, Event A = Less than 2, Event B = EVEN. CANNOT be Even AND Less than 2! '1' is only option and is NOT EVEN!


#         CALCULATING PROBABILITIES
#  ADDITION RULE (ONE OR BOTH):  P(A or B) = P(A) + P(B) - P(A and B)     #makes sense! Intersection is included Twice so must SUBTRACT P(A and B)!
#                 For MUTUALLY EXCLUSIVE Events=P(A) + P(B)     #Since NO Intersection!
#Made FUNCTION to calculate this:
def prob_a_or_b(a, b, all_outcomes):     
    prob_a = len(a)/len(all_outcomes)
    prob_b = len(b)/len(all_outcomes)
    inter = a.intersection(b)      #Have this Method called '.intersection()' to find INTERSECTION point BETWEEN 2 Sets of Data! i.e. elements which are IN BOTH of the Sets
    prob_inter = len(inter)/len(all_outcomes)
    return prob_a + prob_b - prob_inter
#Example 1 - Rolling Die Once, A = Even, B = Odd
print(prob_a_or_b({2,4,6}, {1,3,5}, {1,2,3,4,5,6}))  #='1' AS Expected - either Odd OR Even, MUTUALLY EXCLUSIVE so NO INtersection - just 1/2 + 1/2
#Example 2 - Rolling Die Once, A = Odd, B = Greater than 2
print(prob_a_or_b({1,3,5}, {3,4,5,6}, {1,2,3,4,5,6}))  #=0;8333333. Since = 3/6 + 4/6 - 2/6 
 
#  Probability of 2 DEPENDENT EVENTS = CONDITIONAL PROBABILITY
#   = Probabiliity of an Event Occuring GIVEN THAT ANOTHER Has ALREADY Occured.
   #e.g. For Marble Example above - Probability of Choosing 'Red' GIVEN the FIRST taken out was BLUE
   #     P(RED 2nd | BLUE SECOND) = 3/4  EASY!
   #e.g.2. Bag of 10 marbles - 6 Orange and 4 Blue
   #Take out 1 Orange WITHOUT Replacement, so P(Blue Second|Orange First) = 4/9 = 0.44
#IF 'WITH REPLACEMENT' = INDEPENDENT, so P(A|B) = P(A), P(B|A) = P(B)   EASY!

#   Probability 2 Events happen SIMULTANEOUSLY = MULTIPLICATION RULE:
#   P(A AND B) = P(A)*P(B|A)  for DEPENDENT Events (No Replacement)
#   P(A AND B) = P(A)*P(B)  for INDEPENDENT Events (Replacement)
    #e.g. P(1st Blue AND 2nd Blue) = P(1st Blue) * P(2nd Blue|1st Blue)
    #     Dependent = 2/5 * 1/4 = 1/10. Independent = 2/5 * 2/5 = 4/25    SIMPLE!


#SEE NOTES for Visual 'TREE DIAGRAM' when Representing Probabilities for Multiple Events
    #REAL-WORLD SUMMARY EXAMPLE 'Strep Throat (ST)'  (Testing People if they have it or not)
# Branch 1: P(ST) = 0.20, P(NO ST) = 0.80
# Branch 2: Test People for ST, but have SOME ERROR when Testing - not always correct diagnosis!
# IF Person HAS ST -  P(+|ST) = 85% POSITIVE TEST (Tested to have it) and P(-|ST) = 0.15 NEGATIVE TEST (Tested to NOT HAVE IT - wrong!)
# IF Person DOES NOT HAVE ST - P(+|NO ST) = 0.02 POSITIVE TEST and P(-|NO ST) = 0.98 NEGATIVE TEST (tested to NOT have it, as is true)

# OUTCOMES (EASY = just multiply across branches!): 
#P(ST and +) = P(ST)*P(+|ST) = 0.20*0.85 = 0.17
#P(ST and -) = 0.20* 0.15 = 0.03                 P(ST) = 0.17+0.03 = 0.20 so CORRECT!
#P(NO ST and +) = 0.80*0.02 = 0.016
#P(NO ST and -) = 0.80*0.98 = 0.784              P(NO ST) = 0.016+0.784 = 0.80 so CORRECT!

#          BAYES THEOREM 'P(B|A) = P(A|B)*P(B)/P(A)'   (VERY USEFUL!!!)
#What about REVERSE? e.g. Probability of ST GIVEN +ve Test?
#Here, would be:   P(ST|+) = P(+|ST)*P(ST)/P(+) = 0.85*0.20 / (0.17+0.016) = 0.914 for ST GIVEN +ve Test. ST very likely after a +ve Test!
#Same for REST:   P(ST|-) = P(-|ST)*P(ST)/P(-) = 0.15*0.20/(0.03+0.784) = 0.037  small chance of test being wrong - good!
#                P(NO ST|+) = P(+|NO ST)*P(NO ST)/P(+) = 0.85*0.80/(0.17+0.016) = 0.086021, SMALL chance of being wrong, so good!
#               P(NO ST|-) = P(-|NO ST)*P(NO ST)/P(-) = 0.98*0.80/(0.03+0.784) = 0.963    HIGH so GOOD!


#    ANOTHER Real-World Example (JUST MORE PRACTICE!):
#Patients tested for Virus -  '95% Accurate' GIVEN patient HAS Virus (5% Chance of WRONG +ve Test, EVEN THOUGH Patient DOES NOT Have VIRUS!)
#Is ONLY '80% Accurate' GIVEN patients DOES NOT HAVE Virus (so 20% Chance of WRONG -ve Test, EVEN THOUGH Patient DOES have VIRUS!)
# 4% of City HAS Virus (therefore 96% DONT have it)
# IF patient Tests NEGATIVE ('WITHOUT' Virus), Find Chance that Patient ACTUALLY HAS Virus!                     

#  P(Virus) = 0.04, P(No Virus) = 0.96
#  P(+|Virus) = 0.95, P(-|No Virus) = 0.80
#  P(+|No Virus) = 0.05, P(-|Virus) = 0.20

#Want to find 'P(VIRUS|-ve)' - WORK BACKWARDS with BAYES THEOREM:
#So,  'P(Virus|-ve) = P(-ve|Virus)*P(Virus) / P(-ve)'
# DENOMINATOR: P(-ve) = P(Virus AND -ve) + P(No Virus AND -ve) = (0.04*0.2) + (0.96*0.80) 
#                     = 0.776
#SO, P(Virus|-ve) = 0.20*0.04 / 0.776  = 0.01
# There is VERY SMALL 1% Probability of HAVING a VIRUS, GIVEN a NEGATIVE TEST!



#%%    PROBABILITY DISTRIBUTIONS

#'RANDOM VARIABLE' = 'FUNCTION', used to REPRESENT 'RANDOM EVENTS'
#MUST be 'Numeric' i.e. a NUMBER e.g. Coin Flip could be Represented by 1 = Heads, 0 = Tails.  e.g.2 Outcome of Die Roll 1-6
#SIMULATED using 'random.choice(a, size, replace = True/False)':
#replace = True, means INDEPENDENT (KEEP value WITHIN List 'a'), replace = False, means DEPENDENT (REMOVE Value from 'a' ONCE CHOSEN)
import random
import numpy as np
die_roll = range(1,7)
rolls = 10
result = np.random.choice(die_roll, size=rolls, replace=True)

print(result)   #returns LIST of Randomly Chosen Variables
#DISCRETE RANDOM Variable = COUNTABLE/WHOLE NUMBER Possible Values (e.g.Counting people entering a store. e.g.2. 6-sided die LIMITED to 1,2,3,4,5, or 6!)
#CONTINUOUS RANDOM Variable = UNCOUNTABLE/MEASUREMENT Possible Values (e.g. Measurements like 'Heights', TEMP, TIME...) 

#     Discrete Random Variables PROBABILITY: 
    
# PMF (Probability Mass Function) = Probability of Observing a SPECIFIC Value of DISCRETE Variable
# BD (BINOMIAL DISTRIBUTION) = PMF for LIKELIHOOD of EACH OUTCOME (Specified Value). 'BI' = 2 Possible Outcomes (Heads or Tails!)
# BD Parameters: n=Number of Trials, p=Probability of SUCCESS in EACH Trial (i.e. OBSERVING the Outcome)
#e.g. For Flipping Coin, 10 Times, Binomial(n=10, p=0.5) for Number of Observed HEADS (SIMPLE!)
#Note: GRAPHICALLY ALL BARS ADD UP= 1. MORE Trials (n) = MORE SPREAD OUT Distribution, Since have MORE VALUES - 'Bell Shaped'.

#CALCULATE PMF Binomial Distribution AT ANY VALUE!
#  use 'stats.binom.pmf(x=interested_value, n=trials, p=Probability of Success)
#Example: Probability observing '6 heads' for 10 flips(n), 0.5
import scipy.stats as stats    #Use SCIPY.STATS Library!
probability = stats.binom.pmf(6,10,0.5)  
print(probability)   #0.20507812   

#PMF ACCROSS RANGE OF VALUES 
#JUST ADD UP for EACH VALUE e.g. P(1<=X<=3) = P(X=1) + P(X=2) + P(X=3)
#Example 1: Probability BETWEEN 2-4 Heads, 10 Flips
heads_2_4 = stats.binom.pmf(2,10,0.5) + stats.binom.pmf(3,10,0.5) + stats.binom.pmf(4,10,0.5)
print(heads_2_4)   #0.3662109
#Example 2: LESS THAN 3 Heads
heads_3_less = stats.binom.pmf(0,10,0.5) + stats.binom.pmf(1,10,0.5) + stats.binom.pmf(2,10,0.5)
print(heads_3_less)  #Add FROM '0 to 2',  0.0546975
#Example 3: EASIER to do '1 - Values_we_DONT_WANT', WHEN MANY VALUES!
heads_8_less = 1 - (stats.binom.pmf(9,10,0.5) +stats.binom.pmf(10,10,0.5)) 
print(heads_8_less)  #10 Flips, so 8/10 OR LESS is just 1 - P(9) - P(10) 
#Example 4: Probability of MORE THAN 2, 10 FLips
more_than_2 = 1 - (stats.binom.pmf(0,10,0.5) + stats.binom.pmf(1,10,0.5) + stats.binom.pmf(2,10,0.5))
print(more_than_2)   #SAME THING! 1 - P(2 or Less)

# CDF (CUMULATIVE DISTRIBUTION FUNCTION) - derived FROM PMF 
# = Probability of 'SPECIFIC VALUE OR LESS'
# 'Cummulative', so is CONSTANTLY INCREASING/Larger Value for LARGER Number!
#e.g. for LESS THAN 3 Heads, simply do CDF(X=2) - MUCH EASIER here THAN doing PMF!
#e.g.2 For RANGE 'BETWEEN 3-6 Heads' = P(6 or Less) - P(2 or Less)  - MAKES SENSE! 

#CDF CALCULATED using 'stats.binom.cdf( x=value OR LESS, n, p)' SAME THING!
#Example 1: P(6 or Fewer Heads)
six_or_less = stats.binom.cdf(6,10,0.5)
print(six_or_less)    #0.828125
#Example 2: P(BETWEEN 4 and 8 Heads)
between_4_8 = stats.binom.cdf(8,10,0.5) - stats.binom.cdf(3,10,0.5)
print(between_4_8)     #0.81738
#Example 3: P(MORE THAN 6 Heads)
more_than_6 = 1 - stats.binom.cdf(6,10,0.5)
print(more_than_6)   #0.171875


#     CONTINUOUS RANDOM VARIABLES Probability

# PDF (Probability DENSITY Functions) = ALL POSSIBLE VALUES of the Continuous Random Variable
# is a BELL SHAPED CURVE, where TOTAL AREA UNDER Curve = 1
# =Probability ACROSS A RANGE WITHIN Curve, NOT Single Point (AT Single Point, Probabilty = 0!)
# (so AREA UNDER Curve = Probability of Random Variable being a Value WITHIN that Range))

# 'NORMAL DISTRIBUTION' CURVE - Normal(Mean, SD)
    #e.g. Could be something like 'Heights' (continuous, measurable)
    #Probability of choosing someone 'LESS than 158cm', so would FIND AREA UNDER that Curve 'FROM 0-158'

# CALCULATING 'Normal Distribution' Probabilities:  (similar way to CMF!!!)
    #  norm.cdf(x, loc = MEAN, scale = SD)
#Example 1:  Probability Randomly Chosen LESS THAN 175cm
less_175 = stats.norm.cdf(175, loc=167.64, scale=8)
print(less_175)    #0.8212136
#Example 2: Probability of 'BETWEEN 165cm to 175cm' being randomly observed
between_165_175 = stats.norm.cdf(175,167.74,8) - stats.norm.cdf(165,167.74,8)
print(between_165_175)   #o.45194   
#Example 3: For Weather, Mean = 20, SD - 3. Find Probability of GREATER THAN 24 for Randomly Selected Weather:
temp_greater_24 = 1 - stats.norm.cdf(24, 20, 3)
print(temp_greater_24)   #0.0912  - so LOW Probability of Weather Exceeding 24!!


           #POISSON DISTRIBUTION    
# = NUMBER OF TIMES an EVENT OCCURS WITHIN FIXED TIME/Space INTERVAL
# RATE PARAMETER (LAMBDA) = AVERAGE/EXPECTED Value - PEAK of the Poisson Distribution Graph!   
#  e.g. 'Number of Calls' recieved, BETWEEN 1pm-5pm...
#       EXPECTED Number is '7', so LAMBDA = 7 here!

# POISSON is DISCRETE, so use PMF and CDF to Find it.
     #    poisson.pmf(Probability of a Number, LAMBDA)
#Example 1: EXPECT Rain 10 Times in Next 30 Days
#   Find Probability of Raining EXACTLY 6 TIMES:
six_times = stats.poisson.pmf(6,10)  #LAMBDA = 10, since 10 times Expected!
print(six_times)     #0.0630554  -VERY SMALL chance of Raining 6 times, since EXPECT 10 Times!
#Example 2: Probability of 12-14 TIMES of Rain (JUST ADD UP, Like PMF!)
twelve_fourteen = stats.poisson.pmf(12,10) + stats.poisson.pmf(13,10) + stats.poisson.pmf(14,10) 

#SAME WAY, ALSO can use CDF for POISSON Distributions
     #     poisson.cdf()  Probability of a Number OR LESS    
#Example 1: 6 OR FEWER RAIN Events
print(stats.poisson.cdf(6,10))   #0.13014 SUPER EASY!!!
#Example 2: 12 OR MORE Rain Events
print(1 - stats.poisson.cdf(11,10))  #0.303224
#Example 3: Probability of 12-18 Rain Events
print(stats.poisson.cdf(18,10) - stats.poisson.cdf(11,10))  #0.29603735

#       PROVING that 'LAMBDA' = 'Expected/AVERAGE':
#e.g. say EXPECTED to make '15 sales PER Week' - NOT EXACT, JUST 'Average'/Expected!
#Take SAMPLE of 1000 RANDOM Values FROM Poisson Distribution
random_sample = stats.poisson.rvs(15,size=1000) #Just did 'Lambda' and 'Sample Size' as Arguments
print(random_sample.mean())    #MEAN value varies AROUND '15' = LAMBDA Approximately - PROVES THIS!!!
#Now, could PLOT on HISTOGRAM (IMPORTING Function which Creates Histogram FROM ANOTHER File IN the Directory!) 
from histogram_function import histogram_function
histogram_function(random_sample)   #COOL! As we see, have PLOT OF this Poisson Distribution! Also note how PEAK is AT the LAMBDA value, AS EXPECTED!


#  VARIANCE = Measure of SPREAD of Values and Probabilities IN the DISTRIBUTIONS
# 'LAMBDA = Variance' for Poisson (EQUIVALENT!!)
# 'LARGER Lambda = GREATER SPREAD of DISTRIBUTION' TOO! So can Assume they are SAME!
# Use 'np.var' method to CALCULATE VARIANCE:
rand_vars=stats.poisson.rvs(4,size=1000)
print(np.var(rand_vars))    #4.1574 Variance CLOSE TO Lambda, so APPROXIMATELY SAME!!!
#PROVE this FURTHER if we do MIN and MAX at 2 Different Lambda Values:
print(min(rand_vars), max(rand_vars))    #0  14
rand_vars2 = stats.poisson.rvs(10, size=1000)
print(min(rand_vars2), max(rand_vars2))    #2 23  - LARGER SPREAD/Variance DUE to Larger Lambda! SIMPLE!

#   EXPECTED VALUE and VARIANCE for BINOMIAL Distribution:
#        Expected Value = ' n*p ' for Binomial, (=number of events * Probability of SPECIFIC Outcome) 
#Example - Flip Coin 10 times, so EXPECT '5' Heads. because is just '10 * 0.5 = 5' = EXPECTED VALUE, AS we are Told!
#        Variance = ' n*p*(1-p) '  For Heads, is '5*(1-0.5)' = 2.5  SIMPLE!! 

#   RULES for Expected/Average Values and Variance (Apply to ALL DISTRIBUTIONS)
#  1. E(X+Y) = E(X) + E(Y)   e.g. 10 Quarter flips, 6 Nickel Flips. Expected Value TOGETHER = 5 (Quarter) + 3 (Nickel)
#  2. E(aX) = a*E(X)    Multiplying by Constant a is just a * Expected Value
#  3. E(X+a) = E(X) + a   JUST ADD ON to existing Expected/Average Value!!
#  1.  Var(X+a) = Var(X)   Variance OF CONSTANTS = 0!
#  2.  Var(aX) = a^2 * Var(X)   Constant SCALED by SQUARING it
#  3.  Var(X+Y) = Var(X) + Var(Y)  Variance of 2 INDEPENDENT Random Variables - ADDED SEPERATELY  e.g. X=heads on coin flip Y=rolling 2 on 6-sided Die. 


#    PERCENTILE (Percentage) - WORKING BACKWARDS FROM CDF
# A product is EXPECTED to have '7' defects EACH Day (Lambda), taking RANDOM SAMPLE of 365
# Find HOW MANY DEFECTS in a Day put us in '90th PERCENTILE' (i.e. what are Defects for 90% of Days?)
year_defects = stats.poisson.rvs(7,size=365)     #as usual, take the random variable sample (rvs)
defects_at_90 = stats.poisson.ppf(0.90, 7)  #use 'stats.poisson.ppf(percentile, lambda)' 
print(defects_at_90)   #get '10 OR LESS' Defects. So 90% OF DAYS, will have 10 OR FEWER Defects!
#CHECK THIS by using 'cdf':
print(stats.poisson.cdf(10, 7))    #Gives us '0.9014' - CORRECT!

#%%    SAMPLING 

#SAMPLING DISTRIBUTIONS (INFERENTIAL STATS)
#Practically IMPOSSIBLE to Collect data for ENTIRE Populations!
#SO? can use SMALLER 'Samples' of data  
      #e.g. Average Weight of Sample of '50 Fish' taken to be REPRESENTATIVE OF ALL Population
      #Of course, taking NEW Sample of 50 - MAY get DIFFERENT Average Weight
#USEFUL to EXTRAPOLATE FROM SAMPLE to Describe PROBABILITY/Uncertainty OF the FULL Population!
#LARGER SAMPLE SIZE = 'Sample' Mean CLOSER to 'POPULATION' MEAN (so Extreme Values have SMALLER Impact) 

#   RANDOM SAMPLING - use 'np.random.choice' to CHOOSE the Sample!
import scipy.stats as stats
import numpy as np
import pandas as pd     #GREAT way to IMPORT CSV Files INTO Python (EASIER than Method Covered in 'Python_Refresher'!
import seaborn as sns     #SEABORN = GREAT way to PLOT 'Histograms'!
import matplotlib.pyplot as plt

#Example - Theoretical 'Population' of Salmon Weights
#FIRST, PLOT 'POPULATION' and 'Population MEAN':
population = pd.read_csv("salmon_weights.csv")     
print(type(population))  #just to get brief LOOK at the Data, can see is a PANDAS 'DataFrame' 
population = np.array(population['Salmon_Weights'])  #EXTRACTS Column 'Salmon_Weights' since this is what we want! CHANGES into 'ARRAY' 
population
pop_mean = round(np.mean(population), 3)   #Just ROUNDED Population Mean to 3 d.p.

sns.histplot(population, stat='density') 
plt.axvline(pop_mean, color='r', linestyle='dashed')  #Dashed VERTICAL Line to SHOW WHERE the MEAN is!
plt.title(F"Population Mean: {pop_mean}")   # 'F STRING' FORMATTING - EASIER than Way in notes!
plt.xlabel("Weight (lbs)")                
plt.show() 
plt.clf()   #Closes the Current Plot 
#Now can take SAMPLE FROM the Population: 
sample = np.random.choice(np.array(population), size=30, replace=False)
#From this List of Samples, can find 'SAMPLE' MEAN:
sample_mean = round(np.mean(sample), 3)
print(sample_mean) 
#PLOTTING 'DISTRIBUTION' of 'SAMPLE'  (in SAME WAY as above):
sns.histplot(sample, stat='density')
plt.axvline(sample_mean, color='r', linestyle='dashed')
plt.title(F"Sample Mean: {sample_mean}")
plt.xlabel("Weight (lbs)")
plt.xlabel("Weight (lbs)")
plt.show()
plt.clf()


#       SAMPLING DISTRIBUTIONS 'FOR A STATISTIC' (here will do for 'Mean')
#Notice - EACH Random Sample FROM the Population Gives DIFFERENT 'Sample MEANS'
#So? - Can PLOT a HISTOGRAM of the DIFFERENT SAMPLE MEANS DISTRIBUTION!!

#Example - Estimating Sampling Distribution for our Mean:
sample_size = 50
sample_means = []   #store a LIST OF the Sample MEANS
for i in range(500):      #will take sample mean 500 times here
    samp = np.random.choice(population, sample_size, replace=False)
    sample_mean = np.mean(samp)      #calculated as usual
    sample_means.append(sample_mean)   #add EACH to LIST

sns.histplot(sample_means, stat='density')
plt.title("Sampling Distribution of the Mean")
plt.show()    #gives us the Distribution for the Sample Means

#Note: this could be done for ANY OTHER STATISTIC - e.g. Maximum, Minimum, Variance. NOT JUST 'Mean'!


#     CLT - 'CENTRAL LIMIT THEOREM' (Normally Distributed Sampling Distribution for Mean)
#SPECIFICALLY describes 'MEAN' Sampling Distribution ONLY
#Sampling Distribution is 'NORMALLY DISTRIBUTED' for Mean, IF Sample SIZE is LARGE ENOUGH!
#Recommended Sample Size of 'n>30'. Greater Sample Size = MORE NORMALLY Distributed (as expected!) 
#In Contrast, POPULATION distribution was 'RIGHT-SKEWED' (asymmetrical towards Right, with long tail)

#Quantitatively, Normal Distribution is described by 'POPULATION MEAN' and 'POPULATION SD'
#Given n>30:   (using CLT)
# 1. 'MEAN OF the Sampling Distribution' (OF the 'Mean'!) = 'POPULATION MEAN' (approximately)
# 2.  SAMPLING SD = POPULATION SD / sqrt(n)      where n=sample size
#LARGER Population Sd = MORE VARIATION/SPREAD in Population AND THEREFORE MORE VARIATION in Sample Means!
#LARGER Sample SIZE = LESS VARIATION in Sample Means - OUTLIERS/Extreme Values have LESS NEGATIVE IMPACT!

  #e.g. For the Salmon Sampling Distribution Above:
sampling_distribution_mean = round(np.mean(sample_means),3)
print(sampling_distribution_mean) #65.204 - SAME/SIMILAR to 'POPULATION' MEAN '65.341' 
#Note: when SMALL Sample Size (n<30), must Calculate Sampling SD using 'Numpy': np.std(sample_means) - i.e. directly FROM the Sampling Distribution

#Example: Population Mean = 10, Population SD = 10, n=50
population_mean = 10
population_SD = 10 
#HERE, NOT GIVEN 'Population' DATASET - so must use 'np.random.normal(population_mean, population_SD, size)'
population = np.random.normal(population_mean,population_SD, size = 100000) #size AS LARGE AS POSSIBLE!
sns.histplot(population, stat='density')
plt.title(F"Population Mean: {population_mean}")
plt.xlabel(" ")
plt.show()
plt.clf()     #Note how Population distribution appears 'Normal' - So Sampling Distribution will DEFINITELY be Normal TOO!

sample_size = 50      #AS USUAL, find Sampling Distribution 
sample_means = []
for i in range(500):
    samp = np.random.choice(population, sample_size, replace=False)
    sample_means.append(np.mean(samp))
#Finding SAMPLING Mean and SD From the Population Values:
sampling_mean = round(np.mean(sample_means),3) #Population Mean = 10 AS EXPECTED!
print(sampling_distribution_mean)
sampling_SD = np.std(population)/(sample_size**0.5)
other_way = np.std(sample_means)  #this ALSO will give us SD of Sampling Distribution!
print(sampling_SD)     # sampling_SD = 1.4144

sns.histplot(sample_means, stat='density')   
x = np.linspace(sampling_mean - 3*sampling_SD, sampling_mean + 3*sampling_SD, 100)
plt.plot(x, stats.norm.pdf(x,sampling_mean, sampling_SD), color='k', label='Normal PDF')
plt.title(F"Sampling Distribution Mean: {sampling_mean}")
plt.xlabel("")
plt.show()   
plt.clf()   #Plotted NORMAL DISTRIBUTION Curve WITH the SAMPLING DISTRIBUTION!
#did this Just to show that Sampling Distribution is Approximately NORMAL.


#'Sampling' SD ALSO called 'STANDARD ERROR (SE)' of Estimate of Mean
#Population SD is mostly UNKNOWN - population is IMPOSSIBLY LARGE!
#So?     SE = SD of SAMPLE / sqrt(n)   #since Sample represents Population then!
# AS Sample Size Increases = Standard Error Decreases too
#Smaller SE therefore = NARROWER sampling distribution (LESS SPREAD). Vice Versa - Larger Sample SD = GREATER SPREAD of Distribution
#Note: when plotting, could do plt.xlim(lowest, highest) or plt.ylim() to set LIMITS of the x and y axis respectively.

#AS MENTIONED, CLT is ONLY for 'MEAN' Statistic's Sampling Distribution
#ALSO get Sampling Distributions for OTHER STATS - Median, Max/Min, Variance....)
# 'UNBIASED ESTIMATOR'= when 'MEAN of Sampling Distribution' OF the Statistic (Mean, median....) is EQUAL TO 'POPULATION' Value OF Statistic (Population Mean, Median...) 
                        #e.g. for 'mean' stat sampling distribution which we did above, Sampling Mean = Population Stat 'Mean'.  
#  'BIASED ESTIMATOR' = 'Mean' of Sampling Distribution OF a Stat is NOT EQUAL to Population Value OF that Stat!  
                       #e.g. For MAXIMUM Statistic, Mean of Max Sampling Distribution is NOT EQUAL to Population Maximum! 
                       
#Example -NOW Sampling Distribution of 'MAXIMUM' Stat:                
stat_name = "Maximum"
def statistic(x):
    return np.max(x)
pop_mean=50
pop_SD = 15
population = np.random.normal(pop_mean, pop_SD, 1000)
pop_stat = round(statistic(population), 2)   #around 90-105..
#Plotting Population Distribution AS USUAL:
sns.histplot(population, stat='density')
plt.axvline(pop_stat, color='r', linestyle='dashed')
plt.title(F"Population {stat_name}: {pop_stat}")
plt.show()
plt.clf()   #Side Note: MUST CLOSE -Otherwise will try to plot two different graphs on one - get AXIS ERROR!!!
#Now Plotting SAMPLING DISTRIBUTION for MAX Statistic:

sample_stats = []
for i in range(500):
    samp = np.random.choice(population, 50, replace=False)
    sample_stats.append(statistic(samp))
sampling_mean = round(np.mean(sample_stats),2)
sns.histplot(sample_stats, stat='density')
plt.title(f"Sampling Distribution of {stat_name} \n Mean: {sampling_mean}")
plt.show()
plt.clf()
#as we see, MAXIMUM is BIASED ESTIMATOR - Sampling Mean NOT EQUAL TO Population Maximum!!! 
#Could Try this for ANOTHER Stat e.g. Variance (=BIASED Estimator TOO, since Sampling Mean NOT equal to Population Variance)


#     PROBABILITIES FROM Sampling Distribution of MEAN:
#i.e. Can Estimate PROBABILITY of Observing SPECIFIC RANGE of Sample Means. 
#Use CDF (NORMAL Distribution) 'stats.norm.cdf(x, mean, STANDARD ERROR)' AS WE KNOW!

#Example - Salmon 'POPULATION' has AVERAGE WEIGHT of 60lbs, SD is 40lbs
#Crates SUPPORT 750lbs, want to Transport 10 Fish AT A TIME
#Find Probability that AVERAGE Weight of these 10 Fish is LESS THAN OR EQUAL TO 75:
x=75    #75 OR LESS (since 750/10 = 75)
pop_mean=60
pop_SD = 40
samp_size = 10  
standard_error = pop_SD/(samp_size**0.5)   #CLT, so use STANDARD ERROR (=Sampling Distribution SD)!
print(stats.norm.cdf(x, pop_mean, standard_error))  #SUPER EASY!
#Example 2 - (MORE PRACTICE)
#Cod Fish POPULATION has Average Weight of 36lbs, SD of 20
#Want 25 Fish IN a Crate of UP TO 750lbs (750/25 = 30lbs)
#Want Probabilty that Average Fish is LESS THAN OR EQUAL TO 30lbs
standard_error = 20/(25**0.5)
print(stats.norm.cdf(30, 36,standard_error ))  #0.0668 - VERY SMALL Chance that Fish would be Equal to/LESS than 30lbs! NEED LARGER CRATE!!! 

#%%    MORE ON CLT (just practice really)
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

#CLT VERY USEFUL 
#Helps Quantify Uncertainty AROUND Sample MEAN Estimates
#Also is BASIS OF 'HYPOTHESIS TESTING' (Z- and t-tests)

#SUMMARY of what we did - e.g. Find Average Hr Wage UNDER 150 dollars per hour
#Take RANDOM SAMPLE of 150, Record EACH PERSON's Wage
#From this calculate SAMPLE MEAN. PLOT DISTRIBUTION OF 'SAMPLE'
# = APPROXIMATION OF the Population. 
#Realistically, take 1 sample, but theoretically, COULD take LARGE 10,000 Random Samples OF n='150' people
# FROM this, Plot SAMPLING Distribution OF the MEANS of EACH Sample -EASY!
#Should be NORMALLY DISTRIBUTED (since n>30)
 
#PERCENTILES for SAMPLING DISTRIBUTION- What % of Sample Means Fall WITHIN a SPECIFIC % RANGE?
#e.g.  95% Percentile of Sample Means FROM Sampling Distribution
percentile = np.percentile(sample_means, [2.5, 97.5])    
print(percentile)      #[7.20232553  12.552738] 
# 97.5 - 2.5 = 95% Percentile - MAKES SENSE! = CENTRAL PART of Normal Distribution!


# Brief INTRO to CONFIDENCE INTERVAL (= Way to Express UNCERTAINTY, When Full Population is NOT KNOWN)
#Example - When Full Population NOT GIVEN (COVERED ALREADY!)
n = 35
population_mean = 10
population_SD = 10
overall_population = np.random.normal(population_mean, population_SD, size=100000)

means_of_samples = []    #take LARGE Pop size
for i in range(1000):  #JUST ANOTHER WAY TO TAKE A SAMPLE, like np.random.choice()
    samp= random.sample(list(overall_population), n)
    means_of_samples.append(np.mean(samp))
    
sns.histplot(means_of_samples, stat='density')
plt.title(F"Sampling Distribution - Sampling Mean: {round(np.mean(means_of_samples),3)}")
plt.axvline(np.mean(means_of_samples), color='r', linestyle='dashed')
plt.xlim(-10, 30)  #AS EXPECTED  = NORMAL Distribution
plt.show()

#Can use CLT to ESTIMATE the POPULATION Mean (Sampling Mean equal to Population Mean)
#INSTEAD will have to use 'SAMPLE SD', in REALITY:  (APPROXIMATION OF Population SD)
#Example - Single Sample, n=150, is taken.
sample_SD = np.std(np.random.choice(overall_population, 150, replace=False))
standard_error = sample_SD/(150**0.5)   #sample size n=150 used here.
# '95%' of Normally Distributed Values lie 'WITHIN 1.96 SD OF the MEAN'. (Often Approximated as '2 SD')
#So, ESTIMATE WIDTH of Sampling Distribution:
SE_multiplied_196 = 1.96*standard_error
print(SE_multiplied_196)    #1.65
#i.e. 95% Probability that Observed Sample Mean is NO MORE THAN 1.65 AWAY FROM POPULATION Mean
#THIS IS 95% CONFIDENCE INTERVAL!!! DW - Will Cover in more depth soon!

#%% SUMMARY PROJECT - SAMPLING DISTRIBUTIONS
#First read in our DataFrame we want to use:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
spotify_data = pd.read_csv('spotify_data.csv')
print(spotify_data.head()) #just prints the first few Rows of this DataFrame
column_names = list(spotify_data.columns.values)     
len(column_names)   #counting number of 
print(spotify_data.tail())  #views LAST FEW ROWS of the DataFrame
#Want to see Uncertainties surrounding TEMPOS OF SONGS. FASTER Tempos = BETTER!

#DEFINED FUNCTIONS which we will use in Seperate Script 'sampling_distribution_project'
from sampling_distributions_project import choose_statistic, population_distribution, sampling_distribution
#Now, will EXTRACT the 'TEMPO' Column FROM the DataFrame:
song_tempos = np.array(spotify_data["Song Tempo"])    
print(song_tempos)   #note: dont have to use 'np.array()', but is GOOD PRACTICE to do so!

population_distribution(song_tempos)  #Prints Population Distribution EASILY! - Slightly Normal, Left_Skewed
#Similarly, do Sampling Distributions of all:
sampling_distribution(song_tempos, 30, "Mean")   # = UNBIASED Estimator, NORMALLY Distributed
sampling_distribution(song_tempos, 30, "Maximum")   #BIASED Estimator, since Mean of Sampling Distribution is VERY DIFFERENT to POPULATION Max!
sampling_distribution(song_tempos, 30, "Minimum")     #Biased Estimator
sampling_distribution(song_tempos, 30, "Variance")   #BIASED, BUT CLOSE!!!

#Sample 'VARIANCE' is ALMOST Unbiased, so can use FORMULAT to CALCULATE IT:
#  Population Variance = SUM(Observation - Population Mean)**2 / n
#  Sample Variance = SUM(Observation - Population Mean)**2 / (n-1)
  #These formulae will make Sample Variance an UNBIASED ESTIMATOR of Population Variance!!!
#IN PYTHON, do this by:  np.var(x, ddof=1)  - WILL COVER VARIANCE SOON!
#NOW, Plotting, SHOULD be Unbiased Estimator!

population_mean = np.mean(song_tempos)
print(population_mean)    #118.5
population_std = np.std(song_tempos)
print(population_std)     #24.77   
standard_error=population_std/(30**0.5)
#PROBABILITY of Observing Average TEMPO of 140bpm OR LESS from sample:
print(stats.norm.cdf(140,population_mean, standard_error))  #0.9999989 - ALMOST ALL are 140bpm OR LESS!!
#PROBABILITY of 150bpm OR HIGHER:
print(1 - stats.norm.cdf(150, population_mean, standard_error))    #1.7759e-12 - NEGLIGIBLE Probability of being 150bpm or higher!!

#PROBABILITY of 'SAMPLE MIN' LESS THAN 130bpm - SAME THING!
  #Just use SD and MEAN of SAMPLING DISTRIBUTION for MINIMUM!
#Sampling Mean of Min, from Title: 74.41   
#Sampling SD of Min: 16.89
print(stats.norm.cdf(130, 74.41, 16.89))  #EXACT SAME THING! 

#%%     DESCRIPTIVE STATISTICS (Summary Stats)

# = USEFUL for LARGE DATASETS to CONDENSE and Provide INSIGHTS Into ENTIRE Dataset!
#    e.g. for 10 Contestants, EACH race 100 races
#    Easier to find AVERAGE for EACH Contestant, How CONSITENT Each Racer is (i.e. Very Spread Out from Average - any notable UPS or DOWNS?)

#    VARIABLE TYPES:   BASIC DEFINITIONS!
#Have Flat/TABULAR Datasets - Columns=Variables (attributes/features/FIELDS) and Rows=Observations (instances/records) 
#  1. QUANTITATIVE (Numeric) VARIABLES = COUNTED or MEASURED
#        ('How much/many?', 'Average'?, 'How Often?') 
#    a. DISCRETE = Countable/WHOLE Numbers (integers) e.g. People, coin flips, side on die....     
#    b. CONTINUOUS - MEASUREMENTS, Decimals (Floats)

#  2. CATEGORICAL VARIABLES = Ways to GROUP and SEPERATE DATA (WITHOUT Counting or Measuring!)
#    a. ORDINAL = SPECIFIC ORDER or RANKING 
#      e.g. answering question: 'Strongly Disagree' < 'Disagree' < 'Neutral' < 'Agree' < 'Strongly Agree'     
#      (these responses are RANKED in ORDER, so are ORDINAL!)
#      (Other examples - Age Ranges, customer rating, Ranking in a Competition...)
#      Note: Differences BETWEEN Individual Categories are SUBJECTIVE - NOT ALL SAME/Can VARY! e.g. Difference between 'Satisfied and Very Satisfied' MAY be Different to Difference between 'Dissatisfied and VERY Disatisfied' 
#    b. NOMINAL = NO RELATIONAL ORDER between two or more categories
#       e.g. US States, Ethnicities, Colors, Pets, Food
#       NOTE: Ordinal variables CAN be DEPENDENT ON NOMINAL Variables - so COULD have 'Ordinal' ATTRIBUTES ATTACHED TO Nominal Variables
#       e.g. 'Cities' (Nominal) are ASSIGNED ORDERS based on an (Ordinal) TEMPERATURE description - 'Cool, Coldest, Warm, Warmest'
#    c. BINARY = TYPE of Nominal with ONLY 2 CATEGORIES. e.g. True/False, 1/0, Yes/No (BOTH Categories are MUTUALLY EXCLUSIVE/Only One or the Other!)


#       MEAN/Average   (Measure of CENTRE of Dataset)
# Answer Question: "When are adults MOST Creative and Productive?"
#Dataset of 'One Hundred Greatest Novels of All Time' by 'Le Monde' Magazine. 
#So want to know AVERAGE AGE of Authors WHEN their Books were Published!
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

greatest_books = pd.read_csv("Le Monde - Top 100 Greatest Novels.csv")
author_ages = greatest_books["Ages"]            


average_age = np.mean(author_ages)  #42.12
print(f"Average Age of 100 Greatest Authors, according to Le Monde, is: {average_age}")
#BUT - NOT GOOD MEASURE of 'Creativity/Productivity'"
#MANY FACTORS IMPACT this - 'Date of Publication' is NOT Necessarily Year that Author was MOST Creative. When did they START writing?  
#Mean = CONCISE and PRECISE MEASURE of CENTRALITY

#      MEDIAN  (ANOTHER Measure of CENTRALITY - EXACTLY IN MIDDLE!)
#ASSUME Data set is ORDERED SMALLEST-LARGEST: Even Number of Values = Either BOTH MIDDLE TWO OR AVERAGE of the MIDDLE TWO
median_age = np.median(author_ages)
print(median_age)   # 41 - CLOSE to Mean!
#STILL, CANNOT make DEFINITE CLAIM that Authors are MOST Productive at this Age!!
#Ages VARY! Youngest = 18, Oldest = 76 - so REALLY DEPENDS!

#     MODE   (MOST FREQUENTLY OCCURING OBSERVATION)
# (of course, can have multiple modes IF several have SAME Count!)
from scipy import stats   #use stats.mode()!!
example_array = [24,16,12,10,28,38,12,28,24]
example_mode = stats.mode(example_array)
print(example_mode)  # 'ModeResult(mode=array[12], count=array([2]))'
#Gives us MODE AND the COUNT of the Mode
example_array.value_counts()
two_modes_example = stats.mode([24,16,12,10,12,24,38,12,28,24])
print(two_modes_example)   
#Will ONLY Return for Mode with SMALLEST VALUE - 2 modes are 12 and 24, so ONLY returns '12'


mode_age = stats.mode(author_ages) 
print(f'The Mode Age is {mode_age[0][0]}, with a Count of  {mode_age[1][0]}')  #Mode=38, Count=7
#Note: Accessed 'mode' and 'count' Individually, like a 2D list FROM ModeResult 
#Mode is NOT ALWAYS IN the TALLEST Bucket of Histogram! Here, lies OUTSIDE of CENTRED range around 40-50 (median and mode)
#(SIMPLY the MOST FREQUENT Observation, NOT SPECIFICALLY a Measure of CENTRALITY!)

plt.hist(author_ages, range=(10,80), bins=14, edgecolor='black')
plt.title("Ages of Top 100 Authors at Publication")
plt.axvline(average_age, color='r', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(median_age, color='y', linestyle='dashed', linewidth = 2, label='Median')
plt.axvline(mode_age[0][0], color='orange', linestyle='dashed', linewidth=3, label='Mode')
plt.xlabel("Age")            # 'label' is Label given in the 'Legend'
plt.ylabel("Count")
plt.legend()
plt.show()     #SIMPLE PLOT using pyplot- EASY!


#       VARIANCE   (HOW SPREAD OUT Points are in Dataset) 
#Not always easy to judge 'Spread' visually - so GOOD to use NUMERICAL VALUE to DESCRIBE THIS 'Level of Confidence'!
#Example - Grades for two different teachers classes
teacher_one_grades = [83.42, 88.04, 82.12, 85.02, 82.52, 87.47, 84.69, 85.18, 86.29, 85.53, 81.29, 82.54, 83.47, 83.91, 86.83, 88.5, 84.95, 83.79, 84.74, 84.03, 87.62, 81.15, 83.45, 80.24, 82.76, 83.98, 84.95, 83.37, 84.89, 87.29]
teacher_two_grades = [85.15, 95.64, 84.73, 71.46, 95.99, 81.61, 86.55, 79.81, 77.06, 92.86, 83.67, 73.63, 90.12, 80.64, 78.46, 76.86, 104.4, 88.53, 74.62, 91.27, 76.53, 94.37, 84.74, 81.84, 97.69, 70.77, 84.44, 88.06, 91.62, 65.82]
print(np.mean(teacher_one_grades))    #84.47
print(np.mean(teacher_two_grades))    #84.298

#Will Visualise using SUBPLOT - Just two plots NEXT TO EACH OTHER- GOOD FOR COMPARISONS!
plt.subplot(211)           #'211' is just to do with Scale and Positioning OF the Subplot.                 
plt.title("Teacher One Grades")
plt.xlabel('Grades')
plt.hist(teacher_one_grades)                
plt.xlim(65,105)

plt.subplot(212)
plt.title("Teacher Two Grades")
plt.xlabel("Grades")
plt.hist(teacher_two_grades, bins=20)
plt.xlim(65, 105)

plt.tight_layout()   #Ensures Subplots DONT OVERLAP - Together but seperate!
plt.show()   
plt.clf()   
#OR, ANOTHER WAY to PLOT ON SAME GRAPH Together:
plt.hist(teacher_one_grades, alpha=0.75, label='Teacher 1 Scores', bins=7)
plt.hist(teacher_two_grades, alpha = 0.5, label='Teacher 2 Scores', bins=30)
plt.xlabel('Grades')   #'alpha' = How OPAQUE/TRANSPARENT Plot is
plt.legend()           #Simply so we can visualise better!   
plt.show()
plt.clf()

#Both have SIMILAR MEANS but plot Helps us SEE DIFFERENCES in Variance!
#Teacher One Grades are LESS/narrower SPREAD, More CONSISTENTLY quite High
#Teacher Two Grades are Very SPREAD OUT (higher variance)

#MATHEMATICAL Definition of Variance  (SIMPLE!)
# = DIFFERENCE between EACH DATA POINT and MEAN, SQUARED to get rid of negatives, THEN DIVIDE by NUMBER OF DATA POINTS
# so Variance = SUM(Xi - Mean)**2 / N  = SUM OF SQUARES / Total Number of Data Points

teacher_one_variance = np.var(teacher_one_grades)  #4.2665
teacher_two_variance = np.var(teacher_two_grades)  #78.132
#AS EXPECTED, teacher two grades have MUCH GREATER SPREAD around mean - Up and Down, Some students are exceptional, others may require a little work!


#      STANDARD DEVIATION  (BETTER WAY to Describe SPREAD!)   

#Will use Two datasets - 1. NBA Player Heights (inches)  2. Heights (inches) of Dating Site 'OkCupid' Users
from standard_deviation_data import nba_data, okcupid_data

plt.hist(nba_data, alpha=0.75, label='NBA Data', bins=20)
plt.hist(okcupid_data, alpha=0.5, label="OKCupid Data", bins=20)
plt.axvline(np.mean(nba_data), color='r', linestyle='dashed', label='NBA Mean')
plt.axvline(np.mean(okcupid_data), color='b', linestyle='dashed', label='okcupid Mean')
plt.xlabel("Heights (inches)")
plt.legend()                            
plt.show()            
                 
#So Why is SD Better than Variance? 
#Variance has UNITS of 'inches SQUARED' (since is Sum of Squares/N)
print(np.var(okcupid_data), np.mean(okcupid_data))  #15.4 for NBA Data and 68.414 for OkCupid Data (in inches squared)
#  Standard Deviation = SQUARE ROOT OF VARIANCE!! (INCHES) - EASIER to use WITH MEAN for Comparisons....!
nba_SD = np.std(nba_data)          #3.650 inches
okcupid_SD = np.std(okcupid_data)   #3.924 inches

#FROM this can find NUMBER of SD of Data Point FROM MEAN: 
#     Note: 'MEAN' is CENTRE of Dataset, SD is SPREAD AROUND the Mean/Centre. 
#EXPECTED: '68% of Data' is 1 SD AWAY from Mean, 95% Data is 2 SD Away, 99.7% of data is WITHIN 3 SD From mean)
#     A Data Point OVER 3 SD from mean is VERY RARE/UNUSUAL! 

# EXAMPLE - Lebron James is 80 inches Tall:
# SD FROM Mean = 'Difference From Mean / SD of DATASET' - MAKES SENSE!
nba_difference = 80 - np.mean(nba_data)    #2.016  
okcupid_difference = 80 - np.mean(okcupid_data)  #11.586
num_SD_nba = nba_difference/nba_SD    #0.552 SD FROM MEAN
num_SD_okcupid = okcupid_difference/okcupid_SD  #2.952 SD FROM Mean
#CONCLUSION? 80 inches is VERY CLOSE to NBA Mean Height - as expected, just a LITTLE ABOVE Mean! 
#BUT 80 inches is ALMOST 3 SD from NBA Mean - UNUSUAL!        

#EXAMPLE 2 - Earl Boykins, 65 inches (5 foot 5 Inches - one of Smallest NBA Players!)   
Boykins_SD_nba = (65-np.mean(nba_data))/nba_SD       
Boykins_SD_okcupid = (65-np.mean(okcupid_data))/okcupid_SD 
print(f"{Boykins_SD_nba}, {Boykins_SD_okcupid}")
# -3.557 SD From NBA mean (MUCH LESS THAN MEAN - VERY UNUSUAL FOR NBA PLAYER)
# -0.586995 SD away FROM OkCupid Mean - WELL WITHIN 68% of Data (1 SD)!

#Can GRAPHICALLY VISUALIZE SDs From Mean (axvlines):
my_height = 67   #see where MY height lies! (5 foot 7 inches)
plt.subplot(211)
plt.title("NBA Player Heights")
plt.xlabel("Height (inches)") 
plt.hist(nba_data, bins=20)
plt.axvline(np.mean(nba_data), color='r', linestyle='solid', linewidth=2, label='Mean')
#Now plot 1 SD away, 2 SD away, 3 SD away (EITHER SIDE) - SIMPLE!
plt.axvline(np.mean(nba_data) + nba_SD, color='y', linestyle='solid', linewidth=2)   
plt.axvline(np.mean(nba_data) - nba_SD, color='y', linestyle='solid', linewidth=2)
plt.axvline(np.mean(nba_data) + (2*nba_SD), color='y', linestyle='solid', linewidth=2)   
plt.axvline(np.mean(nba_data) - (2*nba_SD), color='y', linestyle='solid', linewidth=2)
plt.axvline(np.mean(nba_data) + (3*nba_SD), color='y', linestyle='solid', linewidth=2)   
plt.axvline(np.mean(nba_data) - (3*nba_SD), color='y', linestyle='solid', linewidth=2)

plt.axvline(my_height, color='black', linestyle='solid', linewidth='3',label='My Height')
plt.xlim(55, 90)
plt.legend()
plt.show()
plt.clf( )   #My height is EXACTLY at OUTERMOST LIMIT! -i.e. EXACTLY 3 SD LESS than Mean - VERY SHORT, AS EXPECTED, for NBA Player!
print((my_height-np.mean(nba_data))/nba_SD)  #-3.0092 

#%%  PROJECT SUMMARY (Also, practice MODFYING DataFrames!)
#Looking at Weather Variation over a Year in London, to plan trip to see WHEN weather is BEST to Visit.
#Aside from Descriptice Stats, will learn to FILTER/EXTRACT and MANIPULATE DATA
import numpy as np
import pandas as pd  #Will use PANDAS to DO this!
import pickle

london_data = pd.read_csv('London_Weather_data_2015.csv')
print(london_data.head())
last_few = london_data.tail()
#Weather Data for Year 2015, DAILY (all 365 days!)
#ONLY Really interested in TEMPERATURE and MONTHS Associated:
temp = london_data['Temp']   #as we know, Access COLUMNS like this!

#Access SPECIFIC ROWS RANGE using '.iloc[]':
print(london_data.iloc[100:200])  #Rows 100 to 199
print(len(london_data))   #365 Rows/DATAPOINTS AS EXPECTED!

mean_temp = np.mean(temp)   #12.155
temp_SD = np.std(temp)     #4.626
#BUT...NOT USEFUL - Is AVERAGE Temp OVER ENTIRE YEAR!

#BETTER to See MEAN Temp AT a GIVEN MONTH (1, 2, 3, 4...):
# '.loc[rows range][Column]' lets us EXTRACT SPECIFIC Rows FROM COLUMN we WANT.
june = london_data.loc[london_data['Month']==6]['Temp']
july = london_data.loc[london_data['Month']==7]['Temp']
print(f'June Mean Temp: {np.mean(june)}, July Mean Temp: {np.mean(july)}')
#June = 16.49C and July = 18.42C  MEAN TEMP
print(f'June SD: {np.std(june)}, July SD: {np.std(july)}')
#June = 2.383, July Sd = 2.44   - SIMILAR SD - SMALL 
#Larger SD would mean EXTREME WEATHER CHANGES (one day very cold, Other day very hot....)
#QUICKLY REPEAT this for EVERY MONTH by LOOPING - NICE!:
for i in range(1,13):
    month = london_data.loc[london_data['Month']==i]['Temp']
    print(F"Month {i} - Mean is {round(np.mean(month),2)} and SD is {round(np.std(month),2)}")
#REALLY COOL STUFF! Cover this more in Data Analysis Python Course!     


#%%        CATEGORICAL DATA
#Different Summay Stats are needed for CATEGORICAL Data 
#(above, for 'Quantitative/Numeric Data, used CENTRALITY/Central Tendency)
import pandas as pd
import numpy as np
#Example 1 - 2015 NCY Tree Census Data (Survey of 50000 Trees in City, collected by park department employees...)
nyc_trees = pd.read_csv("./nyc_tree_census.csv")
print(nyc_trees.head())  
# "status", "health, "spc_common", "neighborhood" are ALL CATEGORICAL VARIABLES!
tree_species = nyc_trees["spc_common"] #Extract Columns AS USUAL!


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

#MODE - SAME WAY AS ABOVE!

#   MEDIAN
# 1. First View LIST of 'UNIQUE' CATEGORIES/VALUES (WIHTIN the Categorical COLUMN/Variable):
health_statuses = list(nyc_trees["health"].unique())
                      #['Good', 'Poor', 'Fair' nan]
# 2  Must Manually PUT IN ORDER - Lowest to Highest (REMOVING 'nan')            
health_ordered = ['Poor', 'Fair', 'Good'] 
# 3. Need to assign a 'LEVEL' (RANK) to EACH Category:
#    CONVERT to 'CATEGORY' Type!
nyc_trees['health'] = pd.Categorical(nyc_trees['health'], health_ordered, ordered=True)
    #(3 Inputs - 'Categorical Column', Ordered Categories, 'ordered=True')
# 4. NOW, CONVERT Categories TO NUMBERS (like indices):
#    Use 'cat.codes' ATTRIBUTES - Calculate MEDIAN! 
median_index = np.median(nyc_trees['health'].cat.codes)
print(median_index) # '2' - tells us Index of CATEGORY IN 'Unique' List
median_category = health_ordered[int(median_index)]
print(median_category)  #'Good' is MEDIAN category!

#     IMPORTANT POINT - 'MEAN' CANNOT be done for ORDINAL! 
#'Ordered List' of Unique Categories is EQUALLY SPACED - Represented as EQUALLY SPACED 'INTEGER' NUMBERS
#BUT,'Categories' as NOT ALWAYS EQUALLY SPACED THEMSELVES!
     #(Differences BETWEEN Individual Categories MAY VARY)
#For MEAN - SPACING 'MUST' BE EQUAL!
#So, CANNOT Use these 'cat.codes Numbers' for MEAN - Since SPACING MATTERS FOR MEAN!
#e.g. Happiness Score - 1=very unhappy, 5='very happy' - CANT Always Assume that DIFFERENCE between Categories are Equal!
#        Example Why - MEAN 'Trunk Diam 'Category':  
tree_census2 = pd.read_csv('nyc_tree_census2.csv')    
#EASY AS USUAL for simple 'Quantitative' Variable like 'trunk diameter':
mean_diam = np.mean(tree_census2['trunk_diam'])
print(mean_diam)      #11.27 inches Mean Diameter
#BUT Now ALSO have an ORDINAL 'Trunk Diameter CATEGORIES':
          #JUST FOLLOW USUAL PROCEEDURE:
print(tree_census2['tree_diam_category'].unique())
#    ['Medium-Large (10-18in)', 'Large(18-24in)'....]
tree_size_categories = ['Small (0-3in)', 'Medium (3-10in)', 'Medium-Large (10-18in)', 'Large (18-24in)', 'Very large (>24in)']
tree_census2['tree_diam_category'] = pd.Categorical(tree_census2['tree_diam_category'],tree_size_categories, ordered=True)
mean_diam_index = np.mean(tree_census2['tree_diam_category'].cat.codes)
print(mean_diam_index)  #1.97 
mean_diam_category = tree_size_categories[int(mean_diam_index)]          
print(mean_diam_category)   #IN 'Medium (3-10in)' 
#NOT CORRECT! - Calculated Quantintatively as '11.27 inches' - which is in 'Medium-Large'
#Calculated WRONG CATEGORY - Mean for ORDINAL does NOT WORK! 


#  PERCENTILES/IQR - 'SPREAD' for ORDINAL Categories: SAME WAY (using cat.codes!!!!)
# SD and Variance are ALSO NOT Interpretable since Depend ON MEAN
#SO? Use PROPORTIONS of Data IN a RANGE = PERCENTILES/Quantiles
 #e.g. 'RANGE' of 80% of Data' = FROM '10th Percentile' TO '90th Percentile' (90-10=80%!)
 #   This 'Range' Between 10-90th Percentiles = 'IQR' (INTERQUARTILE RANGE)    

#Example - For Our TREE CENSUS 2, 'tree_diam_category' Ordinal Variable:
tree_census2['tree_diam_category'] = pd.Categorical(tree_census2['tree_diam_category'], tree_size_categories, ordered=True)
#Calculate 25th and 75th Percentiles: JUST USE CAT CODES - SAME WAY AS MEDIAN!!! 
p_25_index = np.percentile(tree_census2['tree_diam_category'].cat.codes, 25)
p_25_category = tree_size_categories[int(p_25_index)]
print(p_25_category)    # 'Medium (3-10in)' 
p_75_index = np.percentile(tree_census2['tree_diam_category'].cat.codes, 75)
p_75_category = tree_size_categories[int(p_75_index)]
print(p_75_category)       # 'Large (18-24in)'
#EASY! So, IQR=75-25='50% of Data' is from 'MEDIUM' TO 'LARGE' Tree Diameter!


#   TABLE of 'PROPORTIONS' (% OF the TOTAL in Data)
#'Mode' for Categorical uses '.value_counts()' AS we KNOW
#This GIVES a 'TABLE OF 'FREQUENCIES'
#BUT, is BETTER to CONVERT to 'PROPORTIONS' (%) 
#EXAMPLE - Tree Census 'Status':   
proportions = tree_census2['status'].value_counts()/len(tree_census2['status'])
# 1. So just DIVIDE by LENGTH OF the Tree Census 'Status' Column
print(proportions)  #Alive=0.953, Stump=0.0267,Dead=0.0194
# 2. OR, can Just do 'normalize=True' WITHIN value_counts:
proportions = tree_census2['status'].value_counts(normalize=True)

#IMPORTANT- 'MISSING VALUES' are Coded as 'NaN' in Table of Proportions
#BY DEFAULT these are NOT COUNTED!
#INCLUDE them by saying 'dropna = False' in .value_counts() (DONT DROP IT!)
#Note: '/len(..)' Method ALREADY DOES THIS! RESULT VALUES MAY VARY Slightly IF 'NaN' Present - DEPENDS ON METHOD USED!!
print(tree_census2['health'].value_counts(dropna = False, normalize=True))  
print(tree_census2['health'].value_counts(normalize=True))  
#As expected, Proportions are LARGER WIHTOUT 'NaN' Values!


#BINARY CATEGORICAL VARIABLES 
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
car_eval = pd.read_csv('car_eval_dataset.csv')

#Want to Know 'FREQUENCIES' for 'Manufacturer Country' Categories:
frequencies_table = car_eval['manufacturer_country'].value_counts()
print(frequencies_table)   #MOST Cars are from JAPAN
#Accessing for United States:
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


#%%      INFERENTIAL STATISTICS - INTRODUCTORY NOTES        


#HYPOTHESIS TESTS let us make INFERENCES ABOUT POPULATION(S)
#Take SMALLER SAMPLE of data, 
# e.g.1 - See if People with Vaccine are LESS LIKELY to GET a Disease.
#IF it WORKS IN SAMPLE, Need to know if it is RANDOM FLUKE OR TRUE for REST of Population  
#   i.e. IS SAMPLE REPRESENTATIVE OF Population?
#Random Samples NOT ALWAYS PERFECT REPRESENTATTION!

# e.g.2 - Testing Response Rate for 'Texting' vs. 'Calling' Customers is Most Effective:
# CANT do BOTH for ALL Population!! So? TAKE SMALLER SAMPLES!
# RANDOMLY ASSIGN Samples to TEXT OR CALL. THEN Can finds DIFFERENCE in RESPONSE RATE
# Suppose Text '12% MORE LIKELY' to respond.
# BUT - WOULD this be the TRUE for FULL POPULATION TOO?
# THIS IS WHERE WE USE 'HYPOTHESIS TESTS' to Estimate Probability that FULL Population HAS the SAME Result AS SAMPLE! 

# e.g.3 - RELATIONSHIP between Student 'HOMEWORK' and 'TEST' Scores
# Does Higher Student Homework Score = Higher Test Scores?
# Take RANDOM SAMPLE, INSPECT RELATIONSHIP 
# Use 'REGRESSION ANALYSIS' to see IF SIMILAR in POPULATION OR NOT! - SIMPLE!



#        ONE SAMPLE T-TEST
#Must ASK QUESTIONS ABOUT DATASET, Using PROBABILITY STATEMENTS

#1. ASK QUESTION 
#School Test gives Average Total Score of '29.92' = POPULATION
#100 Students are RANDOMLY CHOSEN to do a PREP Course BEFOREHAND
#Average Score OF this 'SAMPLE of 100' is '31.16'- HIGHER!! 
# Is this DIFFERENCE 'Random CHANCE' OR Does the Prep Course Actually IMPROVE Scores???


#2.   NULL HYPOTHESIS and ALTERNATIVE HYPOTHESIS
#50/50 chance that 100 Student Sample is HIGHER THAN AVERAGE (Either HIGHER OR NOT)
#Is this DIFFERENCE due to Prep Course 'SIGNIFICANT'(i.e. Large ENOUGH)???
#REFRAME the 'Question' for the 'POPULATION' (NOT INDIVIDUAL Sample)

# Hypothesis 1: NULL HYPOTHESIS 
# ="Sample of 100 Prep Course Students are FROM POPULATION with Average Score 'EQUAL TO 29.92'"
#IF NULL = TRUE, Prep Students were Higher BY 'RANDOM CHANCE'! (NOT SIGNIFICANT!)

# Hypothesis 2: ALTERNATIVE HYPOTHESIS  
# ="100 Prep Course Students are FROM a POPULATION with Average Score 'DIFFERENT TO 29.92'
#ASSUME '2 THEORETICAL Populations' -   
#   1. Population where ALL STUDENTS Took Prep Course   
#   2. Population where NONE Took Prep Course
#IF Alternative = TRUE, SAMPLE Came from 'DIFFERENT POPULATION' than OTHER Students!!! 

#Population 1 - 
#1. Either Score 'GREATER THAN' 29.92
#2. NOT EQUAL TO (Greater than 'OR' Less than)
#3. 'LESS THAN' 29.92


#3.    NULL DISTRIBUTION    (Just a 'Sampling' Distribution)
# This is 'Statistic' Distribution IF 'NULL=TRUE'
#Here, Statistic is 'Average' Score for REPEATED Samples of Size=100 (JUST Typical Sampling Distribuion!)
#JUST do SAMPLING Distribution of Mean Scores - According to CLT, is 'Normal Distribution'!!
#If Null=True, getting Average Score of '31.16' is JUST 'BY RANDOM CHANCE', From the Distribution!


#4.   Calculate P-VALUE ('CONFIDENCE INTERVAL'/ 'CERTAINTY in RESULT')
#Given Null=TRUE, HOW LIKELY that SAMPLE Average Score='31.16'? 
#PROBABILITY of getting this 'EXACT' Score is TINY! (Makes Sense!) 
#So? INSTEAD - Want Probability of Getting 'WITHIN A RANGE' Of SCORES
#    For the 3 POSSIBLE 'Alternative Hypotheses':
# 1. Population Average Score 'GREATER THAN 29.92' (POPULATION Average)
#    P-Value = PROBABILITY 'Greater than/Equal to 31.16' ('SAMPLE' Average)  'Sample Average is '1.24 points ABOVE Population Average' (31.16-29.92 = '1.24') 
#    '= ONE-SIDED TEST'  
# 2. Population with Average Score 'NOT EQUAL TO 29.92' (EITHER 'GREATER' OR LESS THAN) 
#    SAMPLE Average '31.16' is 1.24 points ABOVE POPULATION Average
#    Want Probability of Sample Average AT LEAST '1.24' ABOVE AND BELOW POPULATION Average!
#     = 'MORE THAN 31.16' or 'LESS THAN 28.68'    
#     P-Value = TWICE AS LARGE as 'Hypothesis 1' 
#    '=TWO-SIDED TEST'  (Default in Python and R)
# 3. Population with Average Score 'LESS THAN 29.92' 
#    P-Value = Probability 'LESS THAN OR EQUAL TO 31.16'


#INTERPRET RESULTS: 
# Say we got 'p-value=0.031' for 1st Alternative Hypothesis 
#"So, IF RANDOMLY SELECT '100 Prep Students', have '3.1% Chance' that their AVERAGE Score is '31.16 OR HIGHER'
#   (i.e. SMALL PROBABILITY of Scoring HIGHER BY RANDOM CHANCE!)
#   (OBSERVED DATA is UNLIKELY Under NULL Hypothesis - Alternative MORE CONSISTENT!)

#EXAMPLE - Random Sample of 300 Runners given 'SPECIAL SHOES' in Marathon
# Average Finish Time was '230 Mins' for SAMPLE, POPULATION Average Finish TIme = 233 minutes
# IS this SAMPLE Average Time 'SIGNIFICANTLY DIFFERENT' From Average Finishing Time of POPULATION?  
 #NULL = Average Time for Samples 'IS' EQUAL TO POPULATION Average TIme (233 Minutes)
 #ALTERNATIVE = Sample Average 'NOT' EQUAL TO Population Average (Greater than OR Less than Population Average)
#Say we ran '2-SIDED' One Sample t-test, got 'p-value=0.10'

#INTERPRET:  "IF 300 Runners RANDOMLY SELECTED from 'Population' (with Average Time=233 mins), 
#             have '10% CHANCE' that Sample Finish Time is AT LEAST '3 minutes DIFFERENT' (Greater OR Less) THAN POPULATION Average Time!"      


#               SIGNIFICANCE THRESHOLD Introduction: 
# = 'PREDETERMINED Threshold' to Decide IF 'p-value' is SIGNIFICANT OR NOT.
# 'p-value BELOW THRESHOLD = SIGNIFICANT Difference' (Reject Null, Accept ALTERNATIVE)
#       COMMON Threshold: 'Alpha = 0.05' 
#(So, Lower Threshold Value means Less Likely to be Significant)

#Example - p-value = '0.031' WITH 'Significant Threshold = 0.05'
# 0.031 < 0.05, so We REJECT NULL Hypothesis (IS SIGNIFICANT Difference)
#BUT, if was '2-Sided' Test (p=0.062), is ABOVE Threshold - so ACCEPT Null here!
#NOTE: IMPORTANT to CHOOSE ALTERNATIVE HYPOTHESIS EARLY ON, BEFORE Data Collection!
#         (p-Value DEPENDS ON this Alternative Hypothesis!)

#%%   ONE SAMPLE T-TESTS (Python)
#  (Compare 'SAMPLE Average' TO HYPOTHETICAL 'POPULATION' Average)
#Examples of Questions Answered: "Is average time spent by visitors on Website Different FROM 5 Minutes (greater or Less than?)", "IS Average Money Spent by Customers MORE THAN 10 USD?"....
 
  
#  EXAMPLE - Manager wants ONLINE ORDERS to COST '1000 Rupees ON AVERAGE' for Population:
#Today, SAMPLE of '50 PEOPLE' made AVERAGE Payments 'LESS THAN 1000 Rupees'
# QUESTION: "IS spending 'LESS THAN 1000 Rupees on Average' EXPECTED, OR is it JUST RANDOM CHANCE (and Small Sample Size)?        
from scipy.stats import ttest_1samp  #IMPORTS FUNCTION to PERFORM 1 Sample t-test
import numpy as np
prices = pd.read_csv('prices.csv')

prices = np.genfromtxt("prices.csv")  #Small Dataset of 'PRICES' for Purchases of '50 Customers' TODAY!
prices_mean = np.mean(prices)  # ='980' as AVERAGE Purchase Price (SLIGHLTY LESS than 1000)
#NULL Hypothesis = "Average Cost of Purchase Order IS 1000 Rupees - Any Devation is JUST due to RANDOM CHANCE!"
#Alternative Hypothesis = "Average cost is 'NOT' 1000 Rupees (IS a SIGNIFICANT DIFFERENCE!)"

# 'tstat, pval = ttest_1samp('sample' distribution, 'EXPECTED' MEAN)'  
tstat, pval = ttest_1samp(prices, 1000)  # 'tstat' NOT RELEVANT TO THIS COURSE! JUST want 'p-value'
print(pval)    # 'p-value = 0.49207'
#LARGE p-value = 'ACCEPT NULL' HYPOTHESIS! NOT Significant, so is Less than 1000 Rupees 'by RANDOM CHANCE'


#           'ASSSUMPTIONS' of 'One-Sample t-test':
#  Sample is 'RANDOMLY SELECTED' from 'ENTIRE Population' (e.g. Not Just data from Site Members - ALL Population!)
#  'INDEPENDENT' Individual Observations' (e.g. Person recommended by Friend to buy Same Thing - NOT Independent!)
#  'NORMALLY DISTRIBUTED' (given 'LARGE Sample Size' (>40) and 'NO OUTLIERS')
#Note: Can We STILL DO the Test IF ONE of these is Not Met? YES! - JUST 'ACKNOWLEDGE WHAT IS NOT MET'!! 
plt.hist(prices)
plt.show( )        #Thanfully, is 'APPROXIMATELY NORMAL' data, 'WITHOUT OUTLIERS' - COOL!


#   EXAMPLE 2 - NOW have Purchase Prices for SEVERAL DAYS of Week:
# (Access SPECIFIC Days with INDEXING '[]' and do ' delimiter="," ' to SPECIFY)
# FIND 'p-values' for 'FIRST 10 DAYS' OF Dataset
daily_prices = np.genfromtxt("daily_prices.csv", delimiter=",")
print(daily_prices[0])   #just accesses FIRST DAY
for i in range(0,10):
    tstat, pval = ttest_1samp(daily_prices[i],1000)
    print(F"Day {i+1} p-value: {round(pval, 3)}")
#REALLY EASY WIth LOOP!     
#Note: if we CHANGED Population Mean (i.e. Changed Null Hypothesis), will get DIFFERENT 'p-values' as expected 
#      FURTHER From 1000 = SMALLER p-value (Eventually=0) 


#%%       BINOMIAL TEST 

# COMPARES 'FREQUENCY of an Outcome' TO 'EXPECTED PROBABILITY' of Outcome
# e.g. 90% of Passengers EXPECTED to show up, BUT ONLY 80% Show up!
# "IS the '80%' SIGNIFICANTLY DIFFERENT From EXPECTED '90%'? 
#(Binomial Tests are for 'BINARY CATEGORICAL' Data - since are comparing Sample 'FREQUENCY' TO EXPECTED Population Probability!) 

#EXAMPLE - Doing Test MANUALLY:     (Note: could use imported function, but FIRST want to UNDERSTAND IT!):
import numpy as np
import pandas as pd    
import matplotlib.pyplot as plt
monthly_report =  pd.read_csv("monthly_report.csv")
print(monthly_report.head())  #views first 5 rows
#Is BINARY since have 'y' or 'n' (Purchased OR NOT). Other Categorical Variable is 'item' which was purchased.

#EXPECTED that '10%' of visitors make a PURCHASE (Expected Probability)
#QUESTION: "IS the Purchase Rate BELOW Expectation OR NOT?"

#Need 2 Things: 1. SAMPLE SIZE (number of visitors) 2. Number who PURCHASED something
#ASSUME EACH ROW is UNIQUE Visitor. Sample Size = Just TOTAL ROWS then. NICE!
#Use CONDITIONAL STATEMENT to ADD UP Rows with PURCHASE = 'y' : (ALREADY COVERED in 'Descriptive Statistics' for Categorical Variables!)
sample_size = len(monthly_report)    #500 Visitors Sample 
num_purchased = np.sum(monthly_report['purchase']=='y')
print(num_purchased)    #'41' Visitors PURCHASES an Item

#'EXPECT' that 500*10% = '50 Visitors PURCHASE' (since IF EACH of '500' Visitors had '10% CHANCE' of Purchasing, EXPECT '50 Visitors' to Purchase
#  "IS 41 SIGNIFICANTLY DIFFERENT to 50?"
# NOT ALWAYS EXACTLY 10% since is RANDOM PROCESS!!! - CANNOT PREDICT if a Single Visitor will Purchase or Not!
simulated_visitors = np.random.choice(['y','n'], size=500, p=[0.1,0.9])
           #SIMULATED THIS RANDOM PROCESS   
purchased_visitors = np.sum(simulated_visitors =='y')    #SIMILAR to Above - JUST GET FREQUENCY for Binary Categorical Variables!
       #Gives a RANDOM Value varying AROUND 50!


#SIMULATING 'NULL DISTRIBUTION'  (REPEAT SAME Process MANY TIMES to get DISTRIBUTION)
null_outcomes = []     #USUAL for Sampling Distribution!!
for i in range(10000):      
    simulated_visitors = np.random.choice(['y','n'], size=500, p=[0.1,0.9])
    purchased_visitors = np.sum(simulated_visitors =='y')
    null_outcomes.append(purchased_visitors)
print(null_outcomes[0:10])  #view First 10 Random Numbers of Purchased Visitors

null_min = np.min(null_outcomes)   #varies around '25'
null_max = np.max(null_outcomes)   #varies around '76' 
#NOW 'PLOTTING' the Null Distribution:
plt.hist(null_outcomes)
plt.axvline(41, color='r') #See WHERE our '41 Purchases' IS ON the Distribution
plt.show()      #Approximately a NORMAL DISTRIBUTION! 
#CLOSER to Area with MOST DENSITY = MORE LIKELY
#So '41 Purchases' is SOMEWHAT LIKELY (nearish to Dense Centre)


#       CONFIDENCE INTERVAL:   (Using PERCENTILES)
#  INSTEAD, can Report an INTERVAL which COVERS '95% of Values' (instead of FULL)
print(np.percentile(null_outcomes, [2.5, 97.5]))   
     #Have BETWEEN 37 TO 63 Purchases
#SO -"we are '95% CONFIDENT' that for '10% CHANCE' of Visitor Purchasing, RANDOM SAMPLE of 500 gives BETWEEN '37-63' Purchases!"
#Side Note: Want 95% CI to be MIDDLE PART of Distribution, so 97.5-2.5 = 95%. 5% is OUTSIDE OF THIS CI - so 2.5% is BELOW 2.5th percentile, 2.5% is ABOVE 95th percentile! 
#  So IF an Observed value falls OUTSIDE OF CI - REJECT NULL HYPOTHESIS!!! 
#  Could do SAME THING for 90% CI:
null_90CI = np.percentile(null_outcomes,[5,95])  
print(null_90CI)      #Between 39 TO 61 Purchases! NARROWER RANGE, AS EXPECTED! 


#         CALCULATING P-VALUE for Binomial Test
#ONE-SIDED p-value (Probability Purchase Rate LESS THAN 10%)
#i.e. PROPORTION OF VALUES 'LESS THAN OR EQUAL TO 41' (recorded value) 
null_outcomes=np.array(null_outcomes)
p_value1 = np.sum(null_outcomes<=41)/len(null_outcomes)
print(p_value1)       #p_value of Roughly '0.10' AS EXPECTED!!

#TWO-SIDED p-value (PROBABILITY of Purchase Rate LESS THAN 10%)
#i.e. PROPORTION of Values 'LESS THAN OR EQUAL TO 41' OR 'GREATER THAN OR EQUAL TO 59'  (since 50-41=9. So ALSO want 9 ABOVE 50)
null_outcomes = np.array(null_outcomes)
p_value2 = np.sum((null_outcomes<=41) | (null_outcomes>=59))/len(null_outcomes)
print(p_value2)   #p-value = 0.20  (TWICE the One-Sided p-value - AS EXPECTED!)
#NOTE: '|' is 'OR' LOGICAL OPERATOR, Just like in R Language!


#USING 'binom_test()' to DO ALL ABOVE in ONE STEP:
from scipy.stats import binom_test
p_value = binom_test(41, 500, 0.1, alternative='less')
#Arguments: 'OBSERVED_Successes', 'Size of Sample', 'EXPECTED (NULL) Probability of Success'
#ALSO have 'alternative' which SPECIFIES if 'less than' or 'greater than' for ONE-SIDED!
#   (NOT INCLUDING 'alternative' goes to DEFAULT 2-SIDED)

# MAIN QUESTION: 
#"IS OBSERVED '41' Purchases in Sample 'SIGNIFICANTLY DIFFERENT' from EXPECTED 10% (Null Probability) Purchase Rate?
#(so Purchase Rate was 'DIFFERENT FROM EXPECTATION' this week)"


#%%   SIGNIFICANCE THRESHOLDS CONTINUED...

# RECAP:
#USE Significance Threshold so MAKE DECISION FROM p-value (YES/NO Question)
#   e.g. For a Quiz Question- WANT '70% Chance' of getting RIGHT
#   SAMPLE of 100 people DO the quiz. '60 people' get it CORRECT. 
#   "IS THIS SIGNIFICANTLY DIFFERENT FROM Target of 70% ?" 

#COMMON Threshold is '0.050'. <0.05 is Significant (REJECT NULL), >0.05 is NOT Significant (NULL=TRUE).
#  For QUIZ Example - REMOVE the Question IF is SIGNIFICANTLY DIFFERENT From '70%'
#Run Binomial Test (Null = Probability of getting question correct IS 70%, Alternative = "Probabilty of getting question correct is NOT 70%")
#Significance Threshold of '0.05' is used
#p-value 'LESS THAN 0.05' = 'SIGNIFICANT' (Reject Null and REMOVE this Question), p-value GREATER than 0.05 = NOT SIGNIFICANT so can keep the Question)


#         ERRORS for SIGNIFICANCE THRESHOLDS:

#  TYPE I ERROR (FALSE POSITIVE)  
# = Null Hypothesis is TRUE (Positive), BUT p-value is SIGNIFICANT (less than threshold)!
#e.g. 70% is TRUE Probability, BUT if we RUN Test, get a 'SIGNIFICANT' p-value! p-value is FALSELY SIGNIFICANT! Remove Question Even Though we DONT NEED TO!!!!
#  TYPE II ERROR (FALSE NEGATIVE)
# = Null Hypothesis is NOT TRUE (Negative), BUT p-value is NOT SIGNIFICANT! 
#i.e. KEEP Question EVEN THOUGH we SHOULD REMOVE IT!!!

#Another Example - Average Score for a Test is 50 points. Does an Ergonomic Chair lead to SIGNIFICANTLY DIFFERENT Scores?
#NULL = Average Score IN Ergonomic Chair IS 50 Points
#But Say 'TRUE' PROBABILITY of those who USED Ergonomic Chair, gives Average Score='52 points' (So REJECT NULL)
#Get p-value = 0.07 > Threshold of 0.05 - NOT SIGNIFICANT.
#'TRUE' Probability is 'NOT NULL' though, yet p-value implies to ACCEPT Null! (52 points NOT equal to 50!) 
#This is 'FALSE NEGATIVE' - TYPE II ERROR


#      TYPE I ERROR 'RATE' (False Positives)
#     Type I Error RATE '= Significance Threshold' 
#SIMULATING 'Quiz' to PROVE THIS:  (Null = True, yet p-value=Significant)
false_positives = 0
sig_threshold = 0.05
for i in range(1000):    #USUAL BINOMIAL TEST!!!
    sim_sample = np.random.choice(['correct', 'incorrect'], size=100, p=[0.7,0.3])
    num_correct = np.sum(sim_sample =='correct')   #oscilates AROUND 70%    
    p_val = binom_test(num_correct, 100, 0.7) 
    if p_val < sig_threshold:  #IF SIGNIFICANT (reject Null, EVENTHOUGH Null is True)
        false_positives += 1
print(false_positives/1000)   #'0.049' CLOSE TO SIG-THRESHOLD!
#Just COUNT UP Number of FALSE POSITIVES we HAVE, for each repeat,
#THEN DIVIDE by '1000' to get PROPORTION of False Positive Tests. 
#Is CLOSE to SIGNIFICANCE THRESHOLD of 0.05! CONFIRMS that Type I Error Rate = Significance Threshold (approximately)

#PROBLEM with MULTIPLE Hypothesis Tests?
#MORE TESTS = GREATER Proability of Type I Error!!!
# = HARDER to use Significance Threshold to CONTROL FALSE POSITIVE RATE
# FOR EXAMPLE - Quiz has 10 DIFFERENT QUESTIONS. 
# Want to know If Probability of Correct is DIFFERENT From 70%
# NOW WANT THIS 'FOR EACH QUESTION'!!!!! 10 Hypothesis Tests Needed (One for EACH)!         

#Say NULL=TRUE for EACH Hypothesis Test: (Prob Correct IS 70%)
#SINGLE Question - '95% Chance is Correct' (p-value > 0.05), 5% Chance of Type I Error
#TWO Questions - 'Only 90% Chance Correct FOR BOTH' (0.95*0.95 = 0.90), 10% Chance of AT LEAST ONE Type I Error 
#ALL 10 Questions - '60% Chance' of CORRECT (0.95 ^ 10 = 0.60), left with 40% Chance of AT LEAST ONE Type I Error! 

#What should we do??? 
# - PLAN and DECIDE HOW MANY Hypothesis Tests are NEEDED, Use SMALLER Significance Threshold for EACH (REDUCES Probability of Type I Error)

num_tests = np.array(range(50))   #Increasing Number of Hypothesis Tests from 1 up to 50
type_1_probabilities = 1 - ((1-sig_threshold)**num_tests) #(e.g. '1-((1-0.95)**3)' if 3 Tests done)
plt.plot(num_tests, type_1_probabilities)
plt.title("Type I Error Rate for Multiple Tests")
plt.ylabel("Probability of At Least One Type I Error")
plt.xlabel('Number of Hypothesis Tests')       
plt.show()


#%%      SUMMARY EXAMPLE for Inferential Statistics

#     Binomial Example - EXPECT that 10% of Suscribers READ an Article
#Random Sample of 100 Suscribers find that '15 people' read it - Random Chance or Significant?
#NULL = Suscribers who Read Article 'IS 10%'
#Alternative = Suscribers who Read Article is 'MORE THAN 10%' 
#Assume that after Binomial Test, get 'p-value = 0.04' 

#INTERPRETING: "If EACH Suscriber has 10% Chance of Reading (EXPECTED),
# Then there is '4% CHANCE' that '15 or More' will Read it, in a RANDOM SAMPLE of 100"          
#(Depending on Significance Threshold, will accept or reject Null based on this...)
#(expected probability of outcome is '10%', Observed Frequency of Outcome "15 people")

#     Binomial Example 2 - Sample of 100 Shoppers Visit Website.
# 23 People BOUGHT something   (23/100 = 23% of Shoppers!)
# Null = '20%' of Shoppers who Visit BUY Something
# Alternative = 'MORE THAN 20% of Shoppers BUY Something                           
#FIND 'p-value by doing ONE-SIDED BINOMAIL TEST:
from scipy.stats import binom_test
p_value = binom_test(23,100,0.20,alternative = 'greater')
print(p_value)   #p-value = 0.261   LARGE p-value!
# 0.261 > 0.05 Significance Threshold, NOT SIGNIFICANT
#So ACCEPT NULL - IS a 20% chance of Shoppers buying, was simply ABOVE 20% in Sample by RANDOM CHANCE!
 



#FINAL PROJECT - 'heart' dataset on Patients Evaluated for Heart Disease 
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
from scipy.stats import binom_test
heart = pd.read_csv('heart_disease.csv')
print(heart.head())
#First, will SPLIT into Two SUBSETS 'yes_hd' and 'no_hd'.
#Will Investigate for 'chol' (CHOLESTEROL) and 'fbs' (Fasting Blood Sugar)
# "Whether 'chol' and 'fbs' are GREATER THAN '120' OR NOT"
yes_hd = heart[heart['heart_disease']=='presence']   #FILTER ROWS AS USUAL!
no_hd = heart[heart['heart_disease']=='absence']

#1. Those WITH Heart Disease:
#TOLD that Total Cholesterol OVER 240mg/dl is 'HIGH'
chol_hd = yes_hd['chol']    #extracts the 'cholesterol' column
chol_mean = np.mean(chol_hd) #TOTAL MEAN Cholesterol = 251.5mg/dl
#'251.5' is GREATER THAN '240mg/dl' which is VERY HIGH Cholesterol for those WITH Heart Disease!

#Use 'ONE SAMPLE T-TEST' to see IF HIGH Cholesterol ON AVERAGE:
#(Null = Those WITH Heart Disease have Cholesterol EQUAL TO 240mg/dl)
#(Alternative = Have Cholesterol GREATER THAN 240mg/dl)
tstat, pval = ttest_1samp(chol_hd, 240, alternative='greater')
print(pval)  #ONE SIDED p-value = 0.00354 - VERY SMALL!
#0.00354 is MUCH SMALLER THAN Significance Threshold of 0.05
#REJECT NULL Hyothesis - Average Cholesterol is NOT Simply Greater than 240mg/dl by RANDOM CHANCE! 
# = SIGNIFICANTLY DIFFERENT

#2.  WITHOUT Heart Disease (REPEAT!):
chol_nohd = no_hd['chol']
tstat, pval = ttest_1samp(chol_nohd, 240)
print(pval/2)  #One Sided p-value = 0.264 > Significance Threshold
#'NOT SIGNIFICANTLY' different from 240mg/dl!   EASY!!!
#So ACCEPT NULL Hypothesis (Those WITHOUT Heart Disease have cholesterol EQUAL TO 240mg/dl)    


#     Now Analyse FASTING BLOOD SUGAR (fbs) Levels
# = Binary Categorical Variable (Either ABOVE 120mg/dl or NOT)
# '1' = GREATER THAN 120mg/dl, '0' = LESS THAN/EQUAL TO 120mg/dl
num_patients = len(heart)   #SAMPLE SIZE = 303
high_fbs_patients = np.sum(heart['fbs'] == 1)  
print(high_fbs_patients)   # = '45 Patients with HIGH fbs' 

#COMPARE THIS to KNOWN Population:
#About '8%' of Population HAD Diabetes in 1988 (=NULL Probability)
#ASSUME that 'fbs > 120mg/dl' INDICATES DIABETES 
#ALSO ASSUME 'Sample is REPRESENTATIVE OF this Population'
expected_diabetes = 0.08 * 303    #EASY! 
# = 24 Patients who HAVE Diabetes. QUITE DIFFERENT to OBSERVED '45' Patients!!
#OBSERVED was 45 - 45/303 = 15% (which is what would be EXPECTED From Population!) 

#BINOMIAL TEST:
#"Is Sample FROM Population where fbs > 120 is EQUIVALENT to 8% of the Population OR NOT??"
p_value = binom_test(high_fbs_patients, num_patients, 0.08, alternative='greater')
print(p_value)   #VERY SMALL p-value = 4.69e-05 

#SIMPLY Compared 'FREQUENCY of OUTCOME' (fbs>120) WITH the 'EXPECTED PROBABILITY of the Outcome'!


#%%             LINEAR ALGEBRA - THEORY

#Vectors, Matrices, Linear Systems of Equations 
#In Data Science, we ORGANIZE and MANIPULATE LARGE AMOUNTS of DATA which is FED INTO Learning Models
#MACHINE LEARNING and Deep Learning ALL Involves MATRIX and VECTOR Multiplication and Addition!

#Linear Algebra = MATHS involving Linear Operations and Solving Linear Equations
# 'MULTI-DIMENSIONAL' Data, using VECTORS and MATRICES
#e.g. Number of Cars Along Roads Between Intersections - INTO Intersection = OUT of Intersection
#Becomes COMPLEX when MANY Linear Equations!!! So? Use LINEAR ALGEBRA to SOLVE THESE Complex Problems!

#So? Will learn 2 IMPORTANT Linear ALgebra DATA STRUCTURES and will learn to perform Operations BETWEEN Structures. 

#Will study 'LINEAR TRANSFORMATIONS' and 'LINEAR REGRESSION'


#     VECTORS:
# =Building Blocks with 'DIRECTION' AND 'MAGNITUDE'
# Vector has DIMENTIONALITY = Number of Elements IN that Vector
# 'SCALAR' = JUST Magnitude (e.g. Car with Speed 40mph)
#Example of 2D Vector - Car AT 40mph AND 'to the EAST' (Direction in x AND y directions)
#Vectors/Arrays can ONLY have Elements of SAME TYPE (i.e. Data Type)

# MAGNITUDE/Length of Vector '||v||' 
# = SQUARE ROOT of SUM of EACH Vector Component SQUARED
#Example - Ball travelling through air has Velocities:
y = 8
z=-2
x=-12
magnitude = np.sqrt(y**2 + z**2 + x**2)   # = 14.56
#So TOTAL SPEED of Ball = MAGNITUDE of Vector

#VECTOR OPERATIONS:
# MULTIPLYING Vector by SCALAR (constant) = EACH ELEMENT * Scalar
# ADDITION and SUBTRACTION of Vectors with SAME Dimension = NEW Vector of SAME Dimension
#  (Note: Vector Addition is 'Commutative' - ORDER DOES NOT MATTER)

#Vector DOT PRODUCTS (Multiplying Vectors):
#Multiply 2 EQUAL DIMENSION Vectors and RETURN SUM of PRODUCTS of Vector's CORRESPONDING Components - SIMPLE!
# (Notice how 'Magnitude' is simply Dot Product of Vector WITH ITSELF)
#ANGLE BETWEEN 2 VECTORS = arccos(a*b / ||a||*||b||)


#      MATRICES:
#MATRIX = with 'm' ROWS-by-'n' COLUMNS of Data
#Vectors are simply 'SINGLE COLUMN' Matrices!
#MULTIPLE VECTORS can be COMBINED to make MATRIX
#USEFUL! - Can Represent LARGE SYSTEMS of Equations IN a SINGLE Matrix!

#Matrix has Shape of 'm Rows by n Columns' (Denoted by CAPITAL Letter. Also can include SUBSCRIPT to Indicate LOCATION of a Matrix Element IN the Matrix 'm,n')

#MATRIX OPERATIONS
#Multiplication by Scalar, Addition and Subtraction of EQUAL SHAPED Matrices is SAME AS FOR VECTORS!

#MATRIX 'DOT' MULTIPLICATION:  (e.g. Matrix A . Matrix B)
# = Dot Product Between 'EACH ROW of FIRST' Matrix WITH 'EACH COLUMN of SECOND' Matrix
# IMPORTANT POINT: Number of COLUMNS in Matrix A MUST EQUAL Number of ROWS in Matrix B (vice versa)
#e.g. If 'A' has 3 Columns, 'B' has 3 Rows - is POSSIBLE!
#Matrix DOT 'PRODUCT' will have 'SHAPE of Rows of A x Columns of B'
#'NOT COMMUTATIVE' - ORDER 'DOES' Matter!! ('AB' is NOT EQUAL to 'BA')

#      SPECIAL MATRICES:
#IDENTITY Matrix (I) = SQUARE Matrix, with Diagonal of '1's, All Surrounding Elements of '0's
#                  (Any Matrix * Identity Matrix = ITSELF!)
#TRANSPOSE Matrix = SWAP ROWS and COLUMNS of Matrix (Superscript 'T' denotes this)
#PERMUTAION Matrix (P) = SQUARE Matrix, SIMILAR to Identity Matrix, Except 1 Element in EACH row EQUALS '1', Rest are '0'
#                        SWAPS AROUND Rows OR Columns of an Existing Matrix WHEN MULTIPLIED BY it. Denoted by 'P'
#         ORDER MATTERS! -  P*A = FLIP 'ROWS'. A*P = Flip COLUMNS Around      


#  USING Matrices to SOLVE 'SYSTEMS OF LINEAR EQUATIONS':
#Can Represent System of Linear Equations Using VECTORS
# x, y and z values are SCALARS Each respectively MULTIPLIED By a VECTOR of the COEFFICIENTS (a1, a2, a3 for x.....)    
# Change to Form 'Ax = b' -  " MATRICES of Coefficients on LHS of Equation * Vector of x,y and z = Vector of 'd' Coefficients "
#Can ALSO Write in AUGMENTED MATRIX form = '[A|b]'

#  GAUSS-JORDAN ELIMINATION:
# = way to 'SOLVE FOR UNKNOWN' Variables
# 1. PUT Augmented Matrix INTO 'ROW ECHELON FORM' = Diagonal of '1's ACROSS and where ALL Elements BELOW DIAGONAL are 'EQUAL to 0'
# 2. REWRITE ORIGINAL Equation IN this 'Row Echelon Form'
# 3. JUST SOLVE Using Simple Algebra now!!! COOL!

# How to GET Row Echelon Form? 
#Done by ADDING/SUTRACTING Rows From EACH OTHER and/or SWAPPING ROWS around!
#Note: System of Linear Equations is NOT ALWAYS SOLVABLE, some cases will be Impossible to solve!

# INVERSE MATRICES (A-1):
# Where: A*A-1 = A-1*a = 'I'  'Matrix * Its INVERSE = IDENTITY MATRIX'!
#USE this to SOLVE Linear Equations!
#e.g.    xA = BC    Multiply BOTH SIDES BY INVERSE of A to SOLVE
#        xA*A-1 = BC*A-1     so get 'x=BC*A-1'

#Can CALCULATE the INVERSE MATRIX USING 'Gauss-Jordan Elimination'!
# 1. START with [A|I] Augmented Matrix (A=matrix we want inverse of)
# 2. CONVERT LHS to 'ROW ECHELON' Form, so LHS = IDENTIFY MATRIX 
# 3. Normalize and Now should be '[I|A-1]'  - RHS = INVERSE we WANT!

#  SEE NOTES - P.g. 107-112 for Examples of this MANUAL Method to do this WITHOUT Python.


#%%             LINEAR ALGEBRA - IN PYTHON
import numpy as np
#Will use Numpy 'ARRAYS' to REPRESENT Vectors (1D Arrays) and Matrices (2D Arrays)
vector = np.array([1,2,3,4,5,6])  #1D Array Vector
matrix = np.array([[1,2], [3,4],[4,2]])   #'NESTED' List for MULTI-DIMENTIONAL ARRAYS
print(matrix)  #'EACH Sub-List' in the Array '= ROW' of MATRIX
#               (3 Rows by 2 Column Matrix)   

# 'np.column_stack()' COMBINES 'VECTORS' to Form Matrix:
v = np.array([-2,-2,-2,-2])
u = np.array([0,0,0,0])
w = np.array([3,3,3,3])
A = np.column_stack((v,u,w))   #Argument put INSIDE 'TUPLE' ()
print(A)     #'EACH VECTOR' = 'COLUMN' of the Matrix - EASY!

#Use '.shape' ATTRIBUTE, If we want to know 'SHAPE' of Matrix/Vector:
B = np.array([[1,2],[3,4]])
print(B.shape)   #prints '(2,2)' - 2 Rows, 2 Columns

#INDIVIDUAL Elements are ACCESSED using [Row INDEX, Column Index]:
print(B[0,1])   #1st ROW, 2nd Column = '2' - SIMPLE! 

#SUBSET or Entire DIMENSION Selected using COLON (Like Slicing!):
print(B[:,1])   # Selects 'ALL ROWS' of '2nd Column'  
#NOTE: Output given as  '[2 4]' - IS a Column, but will always OUTPUT HORIZONTALLY (for some reason!)

#Another Example for Practice:
vector_1 = np.array([-2,-6,2,3])
vector_2 = np.array([4,1,-3,8]) 
vector_3 = np.array([5, -7, 9, 0]) 
matrix = np.column_stack( (vector_1, vector_2, vector_3) )  
print(matrix)
print(matrix.shape)   #4 Rows, 3 Columns
print(matrix[2,:])    #ALL of 3rd Row    


#    LINEAR ALGEBRA 'OPERATIONS' IN Python:
#Multiplication of Scalar (done just using '*'):
A = np.array([[1,2],[3,4]])
print(4 * A)   #Each Element is Multiplied
#ADDING Equal-Sized VECTORS/Matrices:
B = np.array([[-4,-3],[-2,-1]])
print(f"Vector Addition: {A + B}")

#   Vector DOT PRODUCTS - use 'np.dot()' Function:
v = np.array([-1,-2,-3])
u = np.array([2,2,2])
print(np.dot(v,u))      #Dot Product = '-12'
 
#'MATRIX' (DOT) Multiplication - use 'np.matmul()' OR '@' shorthand:
print(np.matmul(A,B))   #Or could do A@B for SAME ANSWER!

#Another Example for Practice:
A = np.array([[2,3,-4], [-2,1,-3]])         
B = np.array([[1,-1,4], [3,-3,3]])          
C = np.array([[1,2], [3,4], [5,6]])         
print(4*A - 2*B)      
E = np.matmul(A,C)
print(E)         
print(C@A)       #Just Shows is NOT COMMUTATIVE since are Different!    


#     SPECIAL MATRICIES:
#IDENTITY matrix - use 'np.eye()' Function: (Takes Integer 'n' and RETURNS a SQUARE Identity Matrix 'n-by-n')
print(np.eye(4))   
#Matrix of ALL ZEROS - use 'np.zeros()' Function:  (Tuple Argument of SHAPE of Matrix)
zero_matrix = np.zeros((3,4))
print(zero_matrix)         
#TRANSPOSE MATRIX - use '.T' ATTRIBUTE:
A = np.array([[1,2],[3,4]])
print(A.T)     #simply transposes existing matrix!

#Example Demonstrating that 'A-1*A = A*A-1 =  I':
A = np.array([[1,-1,1],[0,1,0],[-1,2,1]])
B = np.array([[0.5,1.5,-0.5],[0,1,0],[0.5,-0.5,0.5]])
print(np.matmul(A,B))    #Shows that A and B are Inverses of Each Other!  
print(np.matmul(B,A))    #BOTH Give Identity Matrix AS EXPECTED!


#         OTHER LA OPERATIONS:
# (Will use 'numpy.linalg' SUBLIBRARY to DO these):

# 'LENGTH/MAGNITUDE of Vector' found with 'np.linalg.norm()':
v = np.array([2, -4, 1])
v_norm = np.linalg.norm(v)
print(v_norm)    # Magnitude of Vector = 4.58         
# 'INVERSE of EXISTING Square Matrix' with 'np.linalg.inv()':
A = np.array([[1,2],[3,4]])
print(np.linalg.inv(A))  #Gives us INVERSE (A-1) 


# SOLVING for UNKNOWN VARIABLES "System of Linear Equations"
# 'Ax=b' form - Use np.linalg.solve(), inputs 'A' and 'b' :
#Example:                
#  x + 4y - z = -1    
#  -x + 3y + 2z = 2
#  2x - y - 2z = -2
A = np.array([[1,4,-1],[-1,-3,2],[2,-1,-2]])  
print(A.shape)        #As usual, EACH Sub-list = ROW of Augmented Matrix
b = np.array([-1,2,-2])          #Augmented form Variables defined
x,y,z = np.linalg.solve(A,b)
print((x,y,z))    #gives 'x=0, y=0, z=1'  - SIMPLE! 

#%%     CREATING and TRANSFORMING IMAGES (using LINEAR ALGEBRA)
import numpy as np
import matplotlib.pyplot as plt

#First CREATE a 7x7 Matrix of a Heart-Shaped Image:
heart_img = np.array([[255,0,0,255,0,0,255],
              [0,255/2,255/2,0,255/2,255/2,0],
          [0,255/2,255/2,255/2,255/2,255/2,0],
          [0,255/2,255/2,255/2,255/2,255/2,0],
              [255,0,255/2,255/2,255/2,0,255],
                  [255,255,0,255/2,0,255,255],
                  [255,255,255,0,255,255,255]])
#Now Create Function to SHOW the IMAGE (as Plot of sorts!):
def show_image(image, name_identifier):
    plt.imshow(image, cmap = 'gray')   #pyplot function
    plt.title(name_identifier)  
    plt.show()     #Note: For 'cmap' CAN choose OTHER COLOR-SCALES IF we WANT!               
     
show_image(heart_img, "Heart Image")
#Used 'GRAYSCALE' where Ranges from '0=Black', '255=White', 255/2 = Gray color (HALF)

#Now can do IMAGE TRANSFORMATIONS using Linear Algebra:
#INVERTING COLORS of Matrix:
inverted_heart_img = 255 - heart_img  #Is 255 - EACH ELEMENT OF MATRIX - COOL! Very Handy!
show_image(inverted_heart_img, "Inverted Heart Image")
# (Notice that 'Gray' Pixels stay SAME - since 255 - 255/2 = 255/2 !)

#ROTATING Image Matrix (TRANSPOSE/SWAP Rows and Columns):
rotated_heart_img = heart_img.T
show_image(rotated_heart_img, "Rotated Heart Image")

#PLOTTING RANDOM Image:
random_img = np.random.randint(0,255, (7,7))  #7-by-7 Matrix    
show_image(random_img, "Random Image")  #Random Pixels!

#Could then SOLVE 'heart_img' AS SYSTEM OF Linear EQUATIONS:
# (Calculate UNKNOWN Variables, Multiply BY 'random_img' to GET 'heart_img')
#  'random_img * x = heart_img'    'Ax = b FORM'
x = np.linalg.solve(random_img, heart_img)
show_image(x, 'x (Unknown Variables')
#Now SOLVE for 'heart_img' and SHOULD get Heart Image!
solved_heart_image = random_img@x
show_image(solved_heart_image, "Solved Heart Image") #BACK TO HEART IMAGE!
#Just Showing How Linear Equations can be Solved SO EASILY!


#%%       DIFFERENTIAL CALCULUS

# = Study of 'RATE OF CHANGE' - HOW Variables RELATE to EACH OTHER
# = HOW a 'RESPONSE Variable' (profit, revenue...) CHANGES WITH a Related 'INDEPENDENT Variable' (e.g. gas prices, TIME...)
#  e.g. What is OPTIMAL TIME to Spend on Making a Routine More Efficient, before you SPEND MORE Time THAN you SAVE.

# Data Scientists must make DATA-DRIVEN DECISIONS, with RECOMMENDATIONS to Best Solve Problems!
#  e.g.2 Data Scientist for t-shirt Company - Investigate HOW 'CHANGE' IN 'Marketing' BUDGET could Impact 'TOTAL PROFIT'
#       If TOO LITTLE Spent on Marketing, Product will be LESS KNOWN! 
#       Spend TOO MUCH and Could reach 'POINT OF DIMINISHING RETURNS' - Money Could be BETTER SPENT ELSEWHERE!
#       WHAT is 'OPTIMAL SIZE' of Marketing Budget?

#This mathematical 'OPTIMIZATION' is Linked with CALCULUS
# - Want to find BEST SOLUTION to a Problem by assessing ALL POSSIBLE OPTIONS    
#  Use 'Statistical Models' to BEST FIT our Data - MAXIMIZE Budget WITHOUT Spending TOO Much!
#  SIMPLE and LOGICAL Concept!

#Calculus involves 'INFINITESIMAL Analysis' - Think of VERY SMALLL Quantities
#USE the 'Small' Quantities to learn HOW 'INSTANTANEOUS Rates of Change' Occur AT SPECIFIC, 'EXACT Points in Time'!  


#         LIMITS
# = shows when (y) VALUES of FUNCTION as APPROACHES a GIVEN POINT
#  e.g.  lim f(x) = L (as x TENDS TOWARDS '6')
# Take Points CLOSER and CLOSER to '6' and FROM this Evaluate WHERE Function is HEADING 
# 'ONE-SIDED Limit':
# f(5), f(5.9), f(5.999999)...  
# - Here, x approaches '6' from 'ONE DIRECTION' - LHS (-ve) 
# OR, From RHS (+ve) - f(6.1), f(6.01), f(6.0005)...

# KEY CONCEPT: 
# Limit (as x approaches 6) ONLY EXISTS 'IF LHS Limit = RHS Limit'
#  (SEE NOTES - pg. 121 for Example for Example of This)
# i.e. LIMIT is 'y value' AT the x-value. The y value WHERE LHS and RHS INTERSECT MUST be EQUAL


# LIMIT DEFINITION 'OF a DERIVATIVE):
#Example - Want to Measure RUNNERS 'INSTANTANEOUS' Speed (i.e. AT EXACT MOMENT in TIME)
#  (t) - Runners DISTANCE FROM START Time (Time=t)
#  What is 'Instantaneous Speed' '1 Second AFTER Race?
#  We CAN find 'AVERAGE' Speed over a Time Interval, but NOT INSTANTANEOUS (EXACT) Speed 
#Take Measurement at 't=1.0000000001 seconds' - TINY INCREMENT
#Take 'AVERAGE SPEED AT' THIS 'TINY INCREMENT' - APPROACHES INSTANTANEOUS Speed now!

# average speed = 'f(x+h)-f(x) / h'    h = TIME the SECOND Measurement is Taken (makes sense!)
#Make 'h' REALLY TINY so APPROACHES '0' = INSTANTANEOUS Rate 


#    DERIVATIVE FUNCTION
#'DERIVATIVE AT A POINT' = 'SLOPE' OF FUNCTION AT the SPECIFIC POINT 
# (i.e. = INSTANTANEOUS Rate of Change)
#'Derivative' is ALSO given as 'SLOPE OF 'TANGENT' LINE AT POINT
#Rather than using limit to Calculate Derivative:
#Use COMMON FUNCTIONS, denoted as:  f'(x)  which is HOW FUNCTION CHANGES AT a Point, 'WITH TIME' usually!

#IMPORTANT PROPERTIES of Derivative Functions:
# f'(x) > 0  = Function f(x) is INCREASING  at a point
# f'(x) < 0  = FUnction f(x) is DECREASING
# f'(x) = 0  = Function NOT CHANGING
#    IF = 0, could mean Function is AT:
#    1. 'LOCAL MAXIMUM' (maxima) = x value where Derivative goes from +ve to 0 to -ve (TOP OF UPWARD CURVE)
#    2. 'LOCAL MINIMUM' (minima) = x value where Derivative goes from -ve to 0 to +ve (BOTTOM of BOTTOM CURVE)
#    3. 'INFLECTION POINT' = CHANGES Direction in Curvature (Need 2nd Derivative Test) - CURVE goes from DOWN to UP
#Note: 'GLOBAL' Maxima/Minima = LARGEST/SMALLEST Points Over ENTIRE RANGE, RATHER than LOCALISED Points


#CALCULATING DERIVATIVES (A-LEVEL Stuff!!)
# e.g.   3x^4 - 2x^2 + 4   has derivative of '12x^3 - 4x'   SUPER EASY!

# PRODUCT RULE:  f(x) = u(x)*v(x)       (When Functions are Multiplied Together)
#           so:  f'(x) = u(x)*v'(x) + v(x)*u'(x)
#e.g. f(x) = x^2 * log(x)  -  f'(x) = x^2 * d(log(x)) + log(x)*2x

#SPECIFIC Defined Derivatives:   
#  d(ln(x)) = 1/x , d(e^x) = e^x , d(sin(x)) = cos(x) , d(cos(x)) = -sin(x)
  
#%%    CALCULATING DERIVATIVE 'IN PYTHON'

#Example - Create ARRAY for a Function:   f(x) = x^2 + 3
import numpy as np
from math import pow     #(just Nicer way to do Powers than '**' !)

dx = 0.05   #'dx' = STEP Between EACH 'x' Value
def f(x):
    return pow(x,2) + 3   #calculates 'y-values' OF the Function!

f_array_x = [np.round(x,2) for x in np.arange(0,4,dx)]
print(f_array_x)    # = Simply 'x values' GOING UP in 'INCREMENTS of dx'
# NOTE: 'np.arange' (ARRAY-RANGE) = RANGE STORED IN 1D ARRAY


f_array_y = [np.round(f(x),2) for x in np.arange(0,4,dx)]
print(f_array_y)   # =ASSOCIATED y Values WITH these x Values

#NOW, CALCULATE DERIVATIVE with 'np.gradient()':
f_array_derivative = np.gradient(f_array_y, dx)
print(f_array_derivative) #Gives us 'DERIVATIVE' FOR the ARRAY - AT EACH INCREMENT (i.e. BETWEEN CONSECUTIVE POINTS)

#Example 2 - Have FUNCTION:   f(x) = sin(x)
from math import sin
import matplotlib.pyplot as plt

dx = 0.01    #AS Discussed, SMALLER INCREMENTS = BETTER! CLOSER to 'INSTANTANEOUS VALUE'
def f(x):
    return sin(x)
sin_x = [x for x in np.arange(0,20,dx)]
sin_y = [f(x) for x in np.arange(0,20,dx)]
sin_deriv = np.gradient(sin_y, dx)   #SAME PROCESS!!!
#  This Lets us See CHANGE IN DERIVATIVES AT DIFFERENT POINTS! Smaller dx means is MORE PRECISE!

#PLOTTING the Sin Curve:
plt.plot(sin_x, sin_y)
plt.plot(sin_x, sin_deriv)  #DERIVATIVE OF 'SIN' = 'COS' CURVE!
plt.show()    #Blue Plot = sin(x) , Orange = Derivative (cos(x))
#Note: if we INCREASED 'dx', would give LESS ACCURATE GRAPH!! NOT a NICE CURVE!


#     LIMIT DEFINITION of DERIVATIVE:
#Instead of 'np.gradient', could CREATE a FUNCTION to CALCULATE 'LIMIT' Derivative:
#  DEFINITION:   Limit of 'f(x+h)-f(x) / h' AS h TENDS to '0' 
from math import sin, cos, log, pi

def limit_derivative(f,x,h):
    return ((f(x+h) - f(x))/h)    
# 'f' = FUNCTION TO BE DIFFERENTIATED.
def f1(x):
    return sin(x)      #Function can be ANYTHING WE WANT!
def f2(x):
    return pow(x,4)
def f3(x):
    return pow(x,2)*log(x)

#Finding 'Limit Derivative' of 'f3', AT x=1, for 3 h values:
print(limit_derivative(f3,x=1, h=2))
print(limit_derivative(f3,x=1, h=0.1)) 
print(limit_derivative(f3,x=1, h=0.0000001))  #SMALLER h values EACH TIME
#AS SHOWN, SMALLER h = Limit Derivative APPROACHES '1' AT x=1

#PLOTTING 'TRUE DERIVATIVE' of 'f1' (cos(x))  
x_vals = np.linspace(1,10,200)        # = x-axis values
y_vals = [cos(val) for val in x_vals]  
plt.figure(1)
plt.plot(x_vals, y_vals, label="True Derivative", linewidth=3)

#NOW, will Plot 'APPROXIMATED DERIVATIVES' on SAME Figure:
def plot_approx_deriv(f):
    x_vals = np.linspace(1, 10 , 200)
    h_vals = [10, 1, 0.25, 0.01]  #For INCREASINGLY SMALLER 'h' Values!
    for h in h_vals:
        derivative_values = [ ]
        for x in x_vals:
            derivative_values.append(limit_derivative(f,x,h))
        plt.plot(x_vals, derivative_values, linestyle='--', label=F'h = {h}')
        plt.legend()
        plt.title("Convergence to Derivative by VARYING h")
    plt.show()      #INDENTATION IS IMPORTANT!
#Simply PLOTS 'APPROXIMATE' DERIVATIVES at 4 DIFFERENT 'h' Values

plot_approx_deriv(f1)   
#Plots show how AS 'h' becomes CLOSER to '0', Plot of Approximate Derivative is CLOSER TO 'cos(x)' TRUE Derivative!

#REPEATING for 'DERIVATIVE of f2':
y_vals2 = [4*pow(val,3) for val in x_vals] #Derivative OF 'f2'
plt.figure(2)
plt.plot(x_vals, y_vals2, label="True Derivative", linewidth = 3)
plot_approx_deriv(f2)   #AGAIN - AS 'h' TENDS to '0' = CLOSER TO 'TRUE' Derivative Curve!













