import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures  #perform a polynomial transform on multiple features
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler   #normaliser les caractéristiques
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

file_path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"

df=pd.read_csv(file_path)

#print(df.head(5))

#Simple Linear Regression : y target / response variable ; x predictor variable

Lm=LinearRegression()
#we want to look at how highway-mpg can help us predict car price

x=df[['highway-mpg']]
y=df[['price']]

Lm.fit(x,y)
    
yhat=Lm.predict(x)
#print(yhat[0:5] )

#rint(Lm.intercept_)  #b0 intercept
#rint(Lm.coef_)       #b1 slope
"""
# Afficher le scatter plot des données réelles
plt.scatter(x, y, label='Données réelles')

# Tracer la droite de régression
plt.plot(x, yhat, color='red', label='Ligne de régression')
"""
# Utiliser Seaborn pour créer un regplot
#ns.regplot(x=x, y=y, data=df, scatter_kws={"color": "blue"}, line_kws={"color": "red"})

#Multiple Linear Regression

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
"""
Lm.fit(Z,df[['price']])
Y_hat = Lm.predict(Z)
print(Lm.intercept_)  
print(Lm.coef_) #b1,b2,b3,b4

#Model Evaluation Using Visualization

#when it comes to simple regression , we use regression plot using regplot 

#residual plot :A residual plot is a graph that shows the residuals on the vertical y-axis and the independent variable on the horizontal x-axis.
#residual : The difference between the observed value (y) and the predicted value (Yhat) 
#If the points in a residual plot are <b>randomly spread out around the x-axis then a linear model is appropriate for the data.
sns.residplot(x=df['highway-mpg'], y=df['price'])
plt.show()  #We can see from this residual plot that the residuals are not randomly spread around the x-axis, leading us to believe that maybe a non-linear model is more appropriate for this data.

#for multiple linear regression 
# distribution plot. We can look at the distribution of the fitted values that result from the model and compare it to the distribution of the actual values.

ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
# the fitted values are reasonably close to the actual values 
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()
"""
# Polynomial Regression and Pipelines
#polynomial regression is a particular case of the general linear regression model or multiple linear regression models.
#We get non-linear relationships by squaring or setting higher-order terms of the predictor variables.

x = df['highway-mpg']
y = df['price']

# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3) # function to fit an 11th degree polynomial to the data points represented by x (independent variables) and y (dependent variables)
p = np.poly1d(f)   #This line converts the array of polynomial coefficients f into a polynomial object p using np.poly1d
print(p)

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()
    
PlotPolly(p, x, y, 'highway-mpg')
#We can already see from plotting that this polynomial model performs better than the linear model. This is because the generated polynomial function "hits" more of the data points.

pr=PolynomialFeatures(degree=2)

Z_pr=pr.fit_transform(Z)

print(Z.shape) #In the original data, there are 201 samples and 4 features.

print(Z_pr.shape)  #After the transformation, there are 201 samples and 15 features.

#Data Pipelines simplify the steps of processing the data.

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

pipe=Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z,y)
ypipe=pipe.predict(Z)

print(ypipe[0:4])

#. Measures for In-Sample Evaluation

#R^2 / R-squared : also known as the coefficient of determination, is a measure to indicate how close the data is to the fitted regression line

#Mean Squared Error (MSE) measures the average of the squares of errors. That is, the difference between actual value (y) and the estimated value (ŷ).
x = df[['highway-mpg']]
y = df[['price']]
#SLR
Lm.fit(x, y)
# Find the R^2
print('The R-square is: ', Lm.score(x, y))

Yhat=Lm.predict(x)

mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)
#MLR
Lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', Lm.score(Z, df['price']))

Y_predict_multifit = Lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))

#Polynomial Fit
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)

mean_squared_error(df['price'], p(x))

#Prediction and Decision Making : how do we determine a good model fit?
"""
What is a good R-squared value?
When comparing models, the model with the higher R-squared value is a better fit for the data.

What is a good MSE?
When comparing models, the model with the smallest MSE value is a better fit for the data.

Let's take a look at the values for the different models.
Simple Linear Regression: Using Highway-mpg as a Predictor Variable of Price.

R-squared: 0.49659118843391759
MSE: 3.16 x10^7
Multiple Linear Regression: Using Horsepower, Curb-weight, Engine-size, and Highway-mpg as Predictor Variables of Price.

R-squared: 0.80896354913783497
MSE: 1.2 x10^7
Polynomial Fit: Using Highway-mpg as a Predictor Variable of Price.

R-squared: 0.6741946663906514
MSE: 2.05 x 10^7

Comparing these three models, we conclude that the MLR model is the best model to be able to predict price from our dataset(highest r2 and smallest MSE). This result makes sense since we have 27 variables in total and we know that more than one of those variables are potential predictors of the final car price.























