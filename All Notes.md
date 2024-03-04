All Notes
--------------------------------------------------------------------------------------------------------------------------------------------
# Lecture 2: Variables and Lists

## Variables and Data
- use type(variable) to identify type of obect
- code blocks only show result of last line of code → use print method to show multiple outputs
- size of string is number of characters

## Basic Operations
- multiplication: *
- add: +
- subtract: -
- divide: /    ##python counts remainder; unlike Java
- exponent: **

- can use variables in calculation (hard coded)

- concatenate strings using +

## Lists
- denoted by […] with elements separated by commas
- can mix variable types
- can use lists within lists

## Sublist
- list_mixed = [”red”, 1, “yellow”, 4.5, 3.5]
- another_list = [list_mixed, 3, “h”]
    - result: [ [”red”, 1, “yellow”, 4.5, 3.5], 3, “h”]
- another_list[0][2] —> “yellow”      #first [ ] indicate which sublist, second [ ] indicate element within dictated sublist

## Histogram for Categorical Data
- add to list without sublist:
    
    list_list = list_colors + [”red”] + [”green”]
- plt.hist(x = list_list)

## Scatterplot for Continuous Data
- plt.scatter(x = list_numbers, y = list_numbers_sqr)
- plt.xlabel(”Number”)
- plt.ylabel(”Squared Value”)
- plt.show()
--------------------------------------------------------------------------------------------------------------------------------------------

# Lecture 3: Mathematical Operations and Random Numbers

import numpy as np     #mathematical operations

## Numbers and Functions

```python
np.pi for pi
np.log(x) computes the logarithm with base "e" (Euler constant)
np.exp(x) compute the Euler constant raised to the power of "x"
np.sin(x) computes the sine of x
np.cos(x) computes the cosine of x
```

## Vector Arrays
- create array by converting list to numeric object
- array is subfunction of library numPy

```python
vec_a = np.array([1,2,3])
vec_a = np.array([0,1,0])

#operations with elements of an array
vec_a * 2 --> 2,4,6

#operations element-by-element between 2 arrays
vec_a + vec_b --> 1,3,3
```

## Summary Statistics of Array

```python
np.mean(vec_a)
np.std(vec_a)
np.min(vec_a)
np.median(vec_a)
np.max(vec_a)
```

## Random Numbers
- normal distribution with mean "loc" (location) and standard deviation "scale”
- number of distinct variables is “size”

```python
randomvar_a = np.random.normal(loc=0, scale=1, size-10)

#can get same random numbers by setting seed
np.random.seed(10393)
```

## Histogram with Random Results
- edit number of bins with plt.hist(x = randomvar_a)

```python
plt.hist(x = randomvar_x)
plt.xlabel("Variable a")
plt.ylabel("Frequency")
```

## Matrix Operations
- create matrix by stacking different rows/columns
- denoted by capital letters

```python
X = np.array([ [1,2,3], [0,4,5], [0,0,6] ]) #stack rows
Y = np.column_stack([ [1,0,1], [2,1,0] ])

#can transpose matrix (row/column switch)
np.matrix.transpose(Y)

#Matrix multiplication
np.dot(X,Y)
np.matmul(X,Y)
```
--------------------------------------------------------------------------------------------------------------------------------------------

# Lecture 4: Boolean and if/else

## Testing Expressions with Text —> Boolean
- want to know if an expression is True or False
- test equality using “**==**”

```python
#test two data
"Expression same?" == "Expression same?"

value_x = "yes"
print(value_x == "yes")
```

## Testing for Keyword in Sentence
- “**in**” command to check if a word is contained in a sentence

```python
keyword = "economic"
sentence = "The FED forecasts economic outcomes"

keyword in sentence                #True
keyword in "Major: Biology"        #False
```

## Test is Word part of List

```python
current_month = "January"
list_summer_months = ["June","July","August"]

current_month in list_summer_months
"July" in list_summer_months
```

## Testing Expressions with Numbers —> Inequalities

- strictly less than: <
- less than or equal: <=
- equal: ==
- strictly more than: >
- more than or equal: >=

```python
x = 10

print( x < 5 )
print( x <= 5 )
print( x == 5 )
print( x >= 5 )
print( x > 5 )
```

## Validate Data Type

- isinstance (variable, test_type) → returns true/false
- type(variable) == test_type

```python
y = 10

isinstance(y, int)
type(y) == int
```

- notes: Equality of Vectors is done element-by-element
    - will return array of true/false

## Testing Multiple Expressions

- **not** condition

```python
age = 23

not (age < 18)    #true -> can vote
```

- **&** condition: both condition A AND B must be satisfied

```python
age = 23
#age between 20 to 30
(age >= 20) & (age <= 30)
```

- **|** condition:  condition A OR B satisfied

```python
student_status = "freshman"
#student 1st or 2nd year?
(student_status == "freshman" | student_status == "sophomore")
```

## Flow Control

```python
color = "red"

if color == "red":
	print("Red")
elif color == "white":
	print("White")
else:
	print("Other")
```

## Sum() function with Arrays
```python
#sum() adds up individual elements
vec_c = np.array([1,2,3])
vec_c.sum()  #6
```
--------------------------------------------------------------------------------------------------------------------------------------------

# Lecture 5: Loops

## Manipulating Lists
- lists with null values (None)
- can add real values later

```python
list_answers = [None,None,None]
```

- assign values using index

```python
list_answers[0] = "Red"
```

- or create an empty list [ ] and append( ) values

```python
new_list = []
new_list.append("Atlanta")
new_list.append("New York")

new_list      #["Atlanta", "New York"]
```

- repeating values

```python
# Repeat a single value 30 times
list_two_rep = [7] * 30

# Repeat a list 4 times
list_answers_rep = list_answers * 4 

# Repeat of 8 null values
list_none_rep = [0] * 8 
list_none_rep2 = ["NONE"] * 8
```

- PITFALL: multiplying lists vs arrays

```
#multiply list = repeat entire list
list_a = [1,2,3]
list_a * 2                 #[1,2,3,1,2,3]

#multiply array = multiply each element
vec_a = np.array(list_a)
vec_a * 2                   #[2,4,6]
```

Counting Length of Vectors

- len() count the number of elements

```python
len(list_a)         #3
```

## For Loops

```python
list_ids = ["KIA", "Ferrari", "Ford", "Tesla"]

#id is the local variable (intermediate)
for id in list_ids:
    print("I have a " + id)

#numbering using str(index)
index = 1
for id in list_ids:
	print("The " + str(index) + " car is a " + id)
	index += 1

for id in range(len(list_ids)):
	print("The " + str(id+1) + " car is a " + list_ids[id])
```

## Plots for Multiple Variables

- rather than writing similar code for different variables, can use for-loop

```python
carfeatures = pd.read_csv("data/features.csv")
list_vars = ["acceleration","weight"]

for variable_name in list_vars:
	plt.scatter(x = carfeatures[variable_name], y = carfeatures["mpg"])
	plt.ylabel("mpg")
	plt.xlabel(variable_name)
	plt.show()
```

- plus numbering for title

```python
carfeatures = pd.read_csv("data/features.csv")
list_vars = ["acceleration","weight"]

index = 1
for variable_name in list_vars:
	plt.scatter(x = carfeatures[variable_name], y = carfeatures["mpg"])
	plt.ylabel("mpg")
	plt.xlabel(variable_name)
	plt.title("Figure" + str(index))
	plt.show()
	index = index + 1
```

- filling values with numbering

```python
#y = x^2 +2x
list_x = [1,2,4,5,6,7,8,9,10]
list_y = [None] * len(list_x)

# Create an index 
index = 0
for x in list_x:
    list_y[index] = list_x[index]**2 + 2*list_x[index]
    index = index + 1

# Display results visually
plt.scatter(list_x, list_y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
```

- appending list using math operations
```python
#y = x^2 +2x
list_x = [1,2,4,5,6,7,8,9,10]
list_y = []

# Create an index 
for x in list_x:
    y = x**2 + 2*x
    list_y.append(y)

# Display results visually
plt.scatter(list_x, list_y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
```
--------------------------------------------------------------------------------------------------------------------------------------------

# Lecture 6: Simulation Studies

Visualizing Random Variables

- sample with n observations
- stimulating from different distributions

```python
# Set Sample size 
# A normal with "loc" and standard deviation "5"
# A chi-square with "df" degrees of freedom
# A uniform with values between -3 and 5

n = 10000

vec_normal  = np.random.normal(loc = 7, scale = 5, size = n)
vec_chisqr  = np.random.chisquare(df = 1, size = n)
vec_unif    = np.random.uniform(low = -3,high = 5, size = n)
```

Multiple Plots in a Row (subplot)

- normal is bell shape
- uniforms is rectangular

```python
#------------------ Setting up subplots-------------------------#
# Create a plot with 1 row, 2 columns
# You will create a list of subfigures "list_subfig"
# You can choose whichever name you like
# The option "figsize" indicates the (width,height) of the graph
fig, list_subfig = plt.subplots(1, 2,figsize = (6,3))

# tight_layout fpr axes not overlapping
plt.tight_layout()

# First Figure
list_subfig[0].hist(x = vec_normal)
list_subfig[0].set_title("Normal Distribution")
list_subfig[0].set_xlabel("Value")
list_subfig[0].set_ylabel("Frequency")

# Second Figure
list_subfig[1].hist(x = vec_unif)
list_subfig[1].set_title("Uniform Distribution")
list_subfig[1].set_xlabel("Value")
list_subfig[1].set_ylabel("Frequency")
```

Single Loops

- simple sequence: [0,1,2,…,n-1]

```python
# Use "list(range(n))" to create a list from 0 to (n-1).

n= 10
list_zero_ten = list(range(n))
```

Drawing Different Samples

```python
vec_xbar = [None, None, None,None]
sample_size = 100

vec_unif  = np.random.uniform(low = -2, high=2, size = sample_size)
vec_xbar[0] = vec_unif.mean()

vec_unif  = np.random.uniform(low = -2, high=2, size = sample_size)
vec_xbar[1] = vec_unif.mean()

vec_unif  = np.random.uniform(low = -2, high=2, size = sample_size)
vec_xbar[2] = vec_unif.mean()

vec_unif  = np.random.uniform(low = -2, high=2, size = sample_size)
vec_xbar[3] = vec_unif.mean()

vec_xbar
```

```python
# draw random sample "num_simulations" times
# Each time create a random vector of size "sample_size"
# generate values from a uniform between -2 and 2

num_simulations = 2000
sample_size     = 100

vec_xbar = [None] * num_simulations

for iteration in range(num_simulations):
    vec_unif  = np.random.uniform(low = -2, high=2, size = sample_size)
    vec_xbar[iteration] = vec_unif.mean()

plt.hist(vec_xbar)
plt.title("Distribution of Xbar with different simulated samples")
plt.ylabel("Frequency")
plt.xlabel("Values of Xbar")
plt.show()
```

### Nested Loops

- mean with different n observations
- Central Limit Theorem: distribution will have a bell shape with higher n

```python
#Can do with repeated code chunks with different sample sizes
#Each time, generating new data from scratch.

num_simulations = 2000

# Simulate with sample size one
sample_size = 1
vec_xbar = [None] * num_simulations
for iteration in range(num_simulations):
    vec_unif  = np.random.uniform(low = -2, high=2, size = sample_size)
    vec_xbar[iteration] = vec_unif.mean()
plt.hist(vec_xbar)
plt.title("Distribution of Xbar with size 1")
plt.ylabel("Frequency")
plt.xlabel("Values of Xbar")
plt.show()

# Simulate with sample size 10
sample_size = 10
vec_xbar = [None] * num_simulations
for iteration in range(num_simulations):
    vec_unif  = np.random.uniform(low = -2, high=2, size = sample_size)
    vec_xbar[iteration] = vec_unif.mean()
plt.hist(vec_xbar)
plt.title("Distribution of Xbar with size 10")
plt.ylabel("Frequency")
plt.xlabel("Values of Xbar")
plt.show()

# Simulate with sample size 50
sample_size = 50
vec_xbar = [None] * num_simulations
for iteration in range(num_simulations):
    vec_unif  = np.random.uniform(low = -2, high=2, size = sample_size)
    vec_xbar[iteration] = vec_unif.mean()
plt.hist(vec_xbar)
plt.title("Distribution of Xbar with size 50")
plt.ylabel("Frequency")
plt.xlabel("Values of Xbar")
plt.show()
```

Nested Loops

```python
# evaluate different sample size
#write a for-loop within another for-loop

num_simulations = 2000
sample_size_list = [1,10,50,100,200]

for sample_size in sample_size_list:

    # The following command a vector null values, of length "num_simulations"
    vec_xbar = [None] * num_simulations
    
    for iteration in range(num_simulations):
            vec_unif  = np.random.uniform(low = -2, high=2, size = sample_size)
            vec_xbar[iteration] = vec_unif.mean()
    plt.hist(vec_xbar)
    plt.title("Distribution of Xbar when n is " + str(sample_size))
    plt.ylabel("Frequency")
    plt.xlabel("Values of Xbar")
    plt.show()
```

- ex with Chi-Square distribution with (df=1)

```python
num_simulations = 2000
sample_size_list = [1,10,50,100,200]

for sample_size in sample_size_list:

    # The following command a vector null values, of length "num_simulations"
    vec_xbar = [None] * num_simulations
    
    for iteration in range(num_simulations):
            vec_chisqr  = np.random.chisquare(df = 1, size = sample_size)
            vec_xbar[iteration] = vec_chisqr.mean()
    plt.hist(vec_xbar)
    plt.title("Distribution of Xbar when n is " + str(sample_size))
    plt.ylabel("Frequency")
    plt.xlabel("Values of Xbar")
    plt.show()
```

Code to Put all Figures in Same Row

```python
# To evaluate different sample size which just have to write a for-loop within 
# another for-loop

num_simulations = 2000
sample_size_list = [1,10,50,100,200]

fig, list_subfig = plt.subplots(1,len(sample_size_list),figsize = [12,3])
plt.tight_layout()

index = 0
for sample_size in sample_size_list:

    # The following command a vector null values, of length "num_simulations"
    vec_xbar = [None] * num_simulations
    
    for iteration in range(num_simulations):
            vec_chisqr  = np.random.chisquare(df = 1, size = sample_size)
            vec_xbar[iteration] = vec_chisqr.mean()
    list_subfig[index].hist(vec_xbar)
    list_subfig[index].set_title("Distribution of Xbar when n is " + str(sample_size))
    list_subfig[index].set_ylabel("Frequency")
    list_subfig[index].set_xlabel("Values of Xbar")
    index = index + 1
```

Loops + if/else

- proportion of True statements in boolean list

```python
list_boolean = [True,False,True,False,False]
np.mean(list_boolean)
```

- sample_stdv as sample standard deviation of X

```python
# Parameters of a normal random variable
n                 = 10000
population_mean   = 2
population_stdv   = 5

# Create random variable and produce summary statistics
X           = np.random.normal(loc = 2,scale = 5,size = n)
Xbar        = X.mean()
sample_stdv = X.std()

# Check that the sample and standard deviation are close to their
# population values
print(Xbar)
print(sample_stdv)
```

- 95% normal confidence interval defined by…
```python
lower_bound = Xbar - 1.96*(sample_stdv / np.sqrt(n))
upper_bound = Xbar + 1.96*(sample_stdv / np.sqrt(n))
```
--------------------------------------------------------------------------------------------------------------------------------------------

# Lecture 7: User-Defined Functions

**Intro to Functions**

- function- block of reusable code (to perform a specific task)
    - helps avoid repetition
    - make large codes manageable
- built-in functions include: print(), type(), round(), abs(), len()
- arguments= values of inputs
- return= output

```python
#Ex:
# First Argument:   np.pi     (a numeric value)
# Second Argument:  6         (the number of decimals)
# Return:  Round the first argument, given the number of decimals in the second argument

round(np.pi,  10)
```

- enter arguments by assigning parameters (what the argument is labeled as within function)

```python
# Here "df" and "size" are both parameters
# They get assigned the arguments "2" and "20", respectively
# The return is a vector of random variables

vec_x = np.random.chisquare(df = 2, size = 20)
vec_x2 = np.random.chisquare(2,20)
#same
```

**Custom Functions**

- General Format:
    
    def function_name(param1, param2):
    
    body code
    
    return expression
    

```python
#Function to calculate compound interest
# V = P(1+ r/n)^(nt)

def function_v(P,r,n,t):
	v = P*(1+(r/n))**(n*t)
	return v
```

```python
#Fuction: pass if grade >= 55, otherwise fail
def f_num(numeric_grade):
    if numeric_grade>=50:
        return 'pass'
    else:
        return 'fail'
 #or
def f_num_one(numeric_grade):
    if numeric_grade >= 50:
        status = 'pass'
    else:
        status = 'fail'
    return status
```

Lambda Functions (defined in one line)

- General Format:
    
    my_function = lamba parameters: expression
    
- generally use for one line of code
- returns result of expression → limits expression capability

```python
#Compound Interest Function
fn_v = lambda P, r, n, t: P*(1+(r/n))**(n*t)
```

**Functions for Visualization**

- returning value of expression is not always necesary
- could create a function for customized plot
```python
#defines histogram that uses red instead of default
def red_histogram(vec_x,title):
    plt.hist(x = vec_x, color = "red")
    plt.title(title)
    plt.ylabel("Frequency")
    plt.show()
```
--------------------------------------------------------------------------------------------------------------------------------------------

# Lecture 8: Local/Global and Apply

**Global Variables**

- stored in working environment
- can be referenced in other parts of notebook
- can be referenced inside functions:
    - but can lead to mistakes
    - preferably, include all inputs as parameters (local)

- note: if not all arguments of function are given, the function will search for missing variables in global environment
    - may be using wrong variable
    - may cause issues if function changes value of global variable

**Local Variables**

- variables defined inside functions are “local”
- stored “temporarily” while running
- includes parameters + intermediate variables
- supercede global variables
- not stored in working environment

```python
def fn_square(x):
    y = x**2
    return(y)
```

- to permanently modify the global variable

```python
def modify_x():
    global x
    x = x + 5

x = 1
# Now, running the function wil permanently increase x by 5.
modify_x()
```

**Operations over Data Frames(apply/map)**

- create empty data frame

```python
data = pd.DataFrame()

#add variables
data["age"] = [18,29,15,32,6]
data["num_underage_siblings"] = [0,0,1,1,0]
data["num_adult_siblings"] = [1,0,0,1,0]

#define functions
# The first two functions return T/F depending on age constraints
# The third function returns the sum of two numbers
# The fourth function returns a string with the age bracket

fn_iseligible_vote = lambda age: age >= 18
fn_istwenties = lambda age: (age >= 20) & (age < 30)
fn_sum = lambda x,y: x + y

def fn_agebracket(age):
    if (age >= 18):
        status = "Adult"
    elif (age >= 10) & (age < 18):
        status = "Adolescent"
    else:
        status = "Child"
    return(status)
```

**Applying Functions with One Argument**

- apply(myfunction)
- takes a dataframe series (column vector) as input
- computes function separately for each individual

```python
# The fucntion "apply" will extract each element and return the function value
# It is similar to running a "for-loop" over each element

#creating a new column within data set

data["can_vote"] = data["age"].apply(fn_iseligible_vote)
data["in_twenties"] = data["age"].apply(fn_istwenties)
data["age_bracket"] = data["age"].apply(fn_agebracket)

# NOTE: The following code also works:
# data["can_vote"]    = data["age"].apply(lambda age: age >= 18)
# data["in_twenties"] = data["age"].apply(lambda age: (age >= 20) & (age < 30))
```

**Dropping Existing Variable from Dataset**

```python
data = data.drop(columns=['age_bracket'])
```

**Mapping Functions with One or More Arguments**

- map( ) function executes a specified function for each item in iterable (list, array, etc.)
- ex: list(map(my_function, list1, list2,…)
    - use list to output?

```python
# In this example, there are more than two arguments
# function goes over each element of data set (row by row)

data["num_siblings"] = list(map(fn_sum,
                                data["num_underage_siblings"],
                                data["num_adult_siblings"]))
```

**External Scripts**

“ipynb” files

- interactive python notebook
- markdown + python code
- great for interactive output

“.py” files

- python (only) scripts
- used for specific tasks
- Why? split code into smaller, more manageable files
    - ex: file with functions
- import functions into working environment from file

```python
import scripts.example_functions as ef
#has function fn_quadratic(x) return(x**2)

x = 1
print(ef.fn_quadratic(1)) #doesn't affect global x
print(ef.fn_quadratic(5))

ef.message_hello("Juan")
```

- files with variables
    - storing values/settings
    - variables are global (can be referenced later)
    - can import and reference variables
    
    ```python
    import scripts.example_variables as ev
    # When we run this program
    # # the value of alpha will be overwritten
    
    alpha = 1
    print(alpha)
    print(ev.alpha)
    ```
    
- can use **from** and ***** to import variables directly into working environment

```python
from scripts.example_variables import *

print(alpha)
print(beta)
print(gamma)
print(delta)
```

****Notes: always want to store modified dataset as new dataset to preserve old one**
```python
#import
carfeatures = pd.read_csv("data_raw/features.csv")
#export
carfeatures.to_csv("data_clean/features.csv")
```
--------------------------------------------------------------------------------------------------------------------------------------------

# Lecture 10: Subsetting Data

- pandas library for manipulating datasets
- mathplotlib.pyplot for creating plots

### Basic Data Display

Extracting Columns

- extract using ‘datasetName.columns.values’ to identify variables

```python
car_colnames = carfeatures.columns.values
#mpg, cylinders, displacement, horsepower, weight, acceleration
```

Subsetting Columns → actual column value (name)

- data[list_names]

```python
#extract first variable: mpg
carfeatures[car_columns[0]]
carfeatures['mpg']

#extract multiple columns & in certain order
# 1) subset using list variable with all wanted columns
list_subsetcols = ['weight', 'mpg']
carfeatures[list_subsetcols]
# 2) double brackets -> data[ [list] ]
carfeatures[['weight', 'mpg']]
```

Subsetting by row/column position → iloc

- sort by column

```python
carsorted <- carfeatures.sort_values(by = "mpg", ascending = False)
```

- subset rows
    
    data.iloc[ row_int, : ]
    
    data.iloc[ list_rows, : ]
    
    - use ‘:’ to indicate ‘all columns’
    - or just include bracket for rows, default all columns

```python
#rows 0,1,2 with all columns
# since carsorted, will show car with top mpg
carsorted.iloc[ [0,1,2] ]

#block of rows

```

- blocks of rows

```python
#use lower:upper  (not including upper)
#or leave one-side blank for all values before/after

#rows 0-4
carsorted.iloc[ 0.5, : ]

#rows below 8 (not including 8)
carsorted.iloc[ :8, : ] 
```

- similar for subsetting columns by position
    - one column: data.iloc[ : , col_integer ]
    - multiple columns: data.iloc[ : , list_cols]
        
                                        data.iloc[ : , lower:upper]
        

- subset row+column: data.iloc[ list_rows, list_cols]

### Filtering Dataframes Based on Logical Expressions

Filtering using ‘pandas.query( )’

- data.query(”logical expression”)
- extracts based on condition

```python
#extract all cars with mpg >= 25
carfeatures.query("mpg >= 25")

#multiple conditions using "and" / "or"
carfeatures.query(" (acceleration >= 10) and (acceleration < 18) ")

#invoke global variables into query using @variablename
#  in don't use @, python look for column
threshold = 25
carfeatures.query("mpg >= @threshold")
carfeatures[carfeatures['mpg'] >= threshold]
```

- if column_name has a space, must denote `variable name` to use in expressions

```python
carfeatures["new variable"] = carfeatures["mpg"]
carfeatures.query("`new variable` >= 25")
```

### Visualization for Subsets of Data

List of unique categories

- use “pd.unique()” to extract list of unique elements within column

```python
pd.unique(carfeatures["cylinders"])
#outputs: 8, 4, 6, 3, 5
```

Compute Two Overlapping Plots

- call plt.scatter() twice and [plt.show](http://plt.show) once at end
- creates stacked graphs
- if want separate graphs, use [plt.show](http://plt.show) in between plots

```python
df_8 = carfeatures.query("cylinders == 8")
df_4 = carfeatures.query("cylinders == 4")

plt.scatter(x = df_8["weight"],y = df_8["acceleration"])
plt.scatter(x = df_4["weight"],y = df_4["acceleration"])
plt.legend(labels = ["8","4"],
           title  = "Cylinders")

plt.show()
```

Compute Plots by All Categories

```python
#unique categories
list_unique_cylinders = pd.unique(carfeatures["cylinders"])

#for loop to plot scatter (weight vs acceleration) for each unique cylinder
#each plot with different color
for category in list_unique_cylinders:
	df = carfeatures.query("cylinders == @category")
	plt.scatter( x = df["weight"], y = df["acceleration"])

#Show after stacking all plots
plt.xlabel("Weight")
plt.ylabel("Acceleration")
plt.legend(labels = list_unique_cylinders,
					 title = "Cylinders")
plt.show()
```

### Alt Method: Filtering Dataframes Based on Logical Expressions

- using square bracket operator
- Extracting cars (rows) with mpg greater than or equal to 25

```python
carfeatures['mpg'] >= 25
#will return list of true/false values

carfeatures[carfeatures['mpg'] >= 25]
#extract rows where the inner list equals TRUE

#extract based on two conditions
carfeatures[ (carfeatures['acceleration'] >= 10) &
						 (carfeatures['acceleration'] < 18) ]
```

- when comparing “pandas” series, need to use bitwise operators for “or” ( | ) and “and” ( & )
    - bc need to do element-to-element boolean comparison of a Series

Comparison: [Why](https://stackoverflow.com/questions/67341369/pandas-why-query-instead-of-bracket-operator) use “pandas.query()” instead of square brackets?

--------------------------------------------------------------------------------------------------------------------------------------------

# Lecture 11: Linear Regression

**statsmodels library**

- statsmodels.formula.api for functions to estimate models
- statsmodels.api for general-use statistical options

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
```

- line of best fit: p = intercept + slope(x)

```python
#creating a column within "dataset" for equation for line of best fit
b0 = 1  #intercept
b1 = 2  #slope
dataset["x"] = np.random.normal(loc = 0,scale = 1, size = n) 
dataset["p"] = b0 + b1*dataset["x"]
```

- plotting data

```python
#plot points
plt.scatter(x = dataset["x"], y = dataset["y"])
#plot line of best fit against x-values
plt.plot(dataset["x"], dataset["p"], color = 'green')

plt.xlabel("X Variable")
plot.ylabel("Y Variable")
plt.legend(labels = ["Data points", "Best fit line"])
plt.show()
```

- Exercise:

```python
#storing sample mean of "y" as "ybar"
ybar = dataset["y"].mean()

#storing standard deviation of "y" as "stdv_sample"
stdv_sample = dataset["y"].std()

#subset observations where "abs(y-ybar) <= stdv_sample"
datset.query("abs(y - @ybar) <= @stdv_sample")
```

- Find Best Fit Line
    - have data on (y,x) but don't know (b0,b1,e)
    - use subfunction “ols()” from smf library
        - 1st argument: formula string with format “outcome ~ independent_vars”
        - 2nd argument: dataset
    - “.fit( )” fits model with standard errors “cov”
    
    ```python
    model = smf.ols(formula = 'y ~ x', data = datset)
    results = model.fit()
    
    #one line
    results = smf.ols(formula = 'y ~ ', data=dataset).fit(cov = "HC1")
    ```
    
    - compute estimated best fit line
        - use “.params” to get attribute “parameters from the results”
    
    ```python
    b_list = results.params
    print(b_list)
    
    #extract intercept and slope from b_list
    #to compute estimated best fit lines
    dataset["p_estimated"] = b_list[0] + b_list[1] * datset["x"]
    
    #estimators for "b0" and "b1" are close to the values used to generate the data
    ```
    
    - plot best fit line
        - use scatter twice, each with different “y” inputs
        - “legend” command to create box with color labels
    ```python
    plt.scatter(x = dataset["x"], y = dataset["y"])
    plt.plot(dataset["x"], dataset["p_estimated"], color = 'green')
    plt.legend(labels = ["Data points", "Estimated Predicted Model"])
    plt.show()
    ```



