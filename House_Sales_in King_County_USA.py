import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

# Load the data
file_name = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(file_name)

# Display data types and descriptions
print(df.dtypes)
print(df.describe())

# Convert date from string to datetime, if necessary, and drop the unnecessary columns
df.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)
print(df.describe())

# Handle non-numeric data
# Assuming 'date' might be a column based on the error, convert it or drop if not needed
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert to datetime or set errors to NaN
    df.drop(['date'], axis=1, inplace=True)  # Drop if not needed

print(df.describe())

# Check and fill NaN values
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

mean_bedrooms = df['bedrooms'].mean()
df['bedrooms'].replace(np.nan, mean_bedrooms, inplace=True)
mean_bathrooms = df['bathrooms'].mean()
df['bathrooms'].replace(np.nan, mean_bathrooms, inplace=True)

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

# Visualizations
sns.boxplot(x='waterfront', y='price', data=df)
plt.show()

sns.regplot(x='sqft_above', y='price', data=df)
plt.show()

# Correlation matrix after ensuring all data is numeric
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[numeric_cols].corr()['price'].sort_values())

# Linear regression examples
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
print("score",lm.score(X, Y))

x = df[['sqft_living']]
y = df['price']
lm = LinearRegression()
lm.fit(x, y)
print(lm.score(x, y))

# Multiple Regression
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]
X = df[features]
lm.fit(X, df['price'])
print("the score is :",lm.score(X, df['price']))

# Pipeline for polynomial and linear regression
pipe = make_pipeline(StandardScaler(), PolynomialFeatures(include_bias=False), LinearRegression())
pipe.fit(X, Y)
print(pipe.score(X, Y))

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
print("number of test samples:", x_test.shape[0])
print("number of training samples:", x_train.shape[0])

# Ridge Regression
RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
print(RidgeModel.score(x_test, y_test))

# Polynomial and Ridge Regression Pipeline
model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=0.1))
model.fit(x_train, y_train)
r_squared = model.score(x_test, y_test)
print(f'R² score on the test data: {r_squared}')





























