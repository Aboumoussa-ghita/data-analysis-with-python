import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from ipywidgets import interact
from sklearn.linear_model import Ridge
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV

file_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'

df=pd.read_csv(file_url)
#print(df.head(5))

df=df._get_numeric_data()
#print(df.head(5))
#Functions for Plotting
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()
    
def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()    
#et's remove the columns 'Unnamed:0.1' and 'Unnamed:0' since they do not provide any value to the models
df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)

#Part 1: Training and Testing

y_data=df[['price']]
x_data=df.drop('price',axis=1)  # toutes les autres colonnes du DataFrame original df à l'exception de la colonne 'price'

#Now, we randomly split our data into training and testing data using the function train_test_split.

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.1,random_state=1) #10% de la data à tester

#print("number of test samples :", x_test.shape[0])
#print("number of training samples:",x_train.shape[0])

lr=LinearRegression()

lr.fit(x_train[['horsepower']],y_train)

#print(lr.score(x_test[['horsepower']], y_test))
#print(lr.score(x_train[['horsepower']], y_train))

#Cross-Validation Score

cross=cross_val_score(lr,x_data[['horsepower']],y_data,cv=4)

#print("The mean of the folds are", cross.mean(), "and the standard deviation is" , cross.std())

yhat = cross_val_predict(lr,x_data[['horsepower']], y_data,cv=4)
#print(yhat[0:5])

#Overfitting, Underfitting and Model Selection

lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

#predicting using training data
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
#print(yhat_train[0:5])

Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
#DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

#predicting using test data 
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
#print(yhat_test[0:5])

Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
#DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)

#Let's see if polynomial regression also exhibits a drop in the prediction accuracy when analysing the test dataset.

#Let's use 55 percent of the data for training and the rest for testing:
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
#We will perform a degree 5 polynomial transformation on the feature 'horsepower'.
polpr=PolynomialFeatures(degree=5)

x_train_pr=polpr.fit_transform(x_train[['horsepower']])
x_test_pr=polpr.fit_transform(x_test[['horsepower']])

lr.fit(x_train_pr, y_train)
yhat = lr.predict(x_test_pr)

#print("Predicted values:", yhat[0:4])
#print("True values:", y_test[0:4].values)

PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, lr,polpr)
#A polynomial regression model where red dots represent training data, green dots represent test data, and the blue line represents the model prediction.
#We see that the estimated function appears to track the data but around 200 horsepower, the function begins to diverge from the data points.

#R^2 for train data
#print("R^2 for the training data is ",lr.score(x_train_pr, y_train))
#R^2 for test data
#print("the R^2 on the test data",lr.score(x_test_pr, y_test))
#We see the R^2 for the training data is 0.5567 while the R^2 on the test data was -29.87. The lower the R^2, the worse the model. A negative R^2 is a sign of overfitting.

#Let's see how the R^2 changes on the test data for different order polynomials and then plot the results:
Rsqu_test=[]
order=[1,2,3,4]

for n in order : 
    pr=PolynomialFeatures(degree=n)
    x_train_pr=pr.fit_transform(x_train[['horsepower']])
    x_test_pr=pr.fit_transform(x_test[['horsepower']])
    lr.fit(x_train_pr,y_train)
    
    Rsqu_test.append(lr.score(x_test_pr,y_test))

"""plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ') """   
#We see the R^2 gradually increases until an order three polynomial is used. Then, the R^2 dramatically decreases at an order four polynomial.
def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly,pr)
    
#interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))

#Ridge Regression
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

RidgeModel=Ridge(alpha=1)
RidgeModel.fit(x_train_pr,y_train)
yhat=RidgeModel.predict(x_test_pr)

#Let's compare the first five predicted samples to our test set:

print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)


# Initialize lists to store the R^2 scores for test and train datasets
Rsqu_test = []
Rsqu_train = []

# Generate an array of alpha values (regularization strengths) multiplied by 10,
# ranging from 0 to 9990 (1000 values)
Alpha = 10 * np.array(range(0,1000))

# Create a progress bar using tqdm to monitor the loop over alpha values
pbar = tqdm(Alpha)

# Iterate over each alpha value in the Alpha array
for alpha in pbar:
    # Create a Ridge regression model with the current alpha value
    RigeModel = Ridge(alpha=alpha)
    
    # Fit the Ridge model to the polynomially transformed training data
    RigeModel.fit(x_train_pr, y_train)
    
    # Calculate and retrieve the R^2 score for both test and training datasets
    test_score = RigeModel.score(x_test_pr, y_test)
    train_score = RigeModel.score(x_train_pr, y_train)
    
    # Update the progress bar with the current test and training R^2 scores
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})
    
    # Append the current test and training R^2 scores to their respective lists
    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

# Note: The loop progresses through each alpha value, updating the model each time,
# fitting it, and storing the performance metrics, allowing for evaluation of how
# the regularization strength impacts model performance.
#We can plot out the value of R^2 for different alphas:
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()

#Grid Search
#The term alpha is a hyperparameter. Sklearn has the class <b>GridSearchCV</b> to make the process of finding the best hyperparameter simpler.

parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]

rr=Ridge()
grid1=GridSearchCV(rr,parameters1,cv=4)

grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)

BestRR=grid1.best_estimator_
print(BestRR)

print(BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test))
#it gives a good score of R2 : 0.841








