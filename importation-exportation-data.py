import pandas as pd 
import numpy as np

url="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

df=pd.read_csv(url , header=None)

#print("The first 5 rows of the dataframe")
#print ( df.head(5))

#print("The last 5 rows of the dataframe")
#print ( df.tail(5))

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df.columns = headers
#print ( df.head(20))

#we need to replace the "?" symbol with NaN so the dropna() can remove the missing values
df1=df.replace('?',np.NaN)

#drop missing values along the column "price" 
df=df1.dropna(subset=["price"],axis=0)
#print( df.head(20))

#print(df.columns)

#df.to_csv("automobile.csv", index=False)

#df.dtypes  #type de tous les elements de la df
#df.describe() 
#df.describe(include="all")
#df[['length', 'compression-ratio']].describe()  #to discribe just the 2 columns 