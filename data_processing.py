import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 


file_path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df=pd.read_csv(file_path,names=headers)

#print ( df.head(5))

# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
#print ( df.head(5))

#missing_data = df.isnull()
#for column in missing_data.columns.values.tolist():
    #Calcul et impression du nombre de valeurs True et False pour chaque colonne
    #print (missing_data[column].value_counts())
    #print("")  
    
#print(df.dtypes)

#Calculate the mean value for the "normalized-losses" column 
avg_norm_losses = df["normalized-losses"].astype("float").mean(axis=0)
#Replace "NaN" with mean value
df["normalized-losses"].replace(np.nan,avg_norm_losses,inplace=True)

#Calculate the mean value for the "stroke" column 
avg_stroke = df["stroke"].astype("float").mean(axis=0)
#Replace "NaN" with mean value
df["stroke"].replace(np.nan,avg_stroke,inplace=True)

#Calculate the mean value for the "bore" column 
avg_bore = df["bore"].astype("float").mean(axis=0)
#Replace "NaN" with mean value
df["bore"].replace(np.nan,avg_bore,inplace=True)

#Calculate the mean value for the "horsepower" column 
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
#Replace "NaN" with mean value
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)


#cela renvoie une série des valeurs uniques et de leurs comptages.
df['num-of-doors'].value_counts()
#idxmax indique la valeur la plus fréquente dans la colonne 'num-of-doors'
#print(df['num-of-doors'].value_counts().idxmax())
df["num-of-doors"].replace(np.nan,"four",inplace=True)

# simply drop whole row with NaN in "price" column
#Reason: You want to predict price. You cannot use any data entry without price data for prediction; therefore any row now without price data is not useful to you
df.dropna(subset=["price"], axis=0, inplace=True)

# réinitialiser l'index du DataFrame 
df.reset_index(drop=True, inplace=True)

#print(df.dtypes)

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

#print(df.dtypes)  #  all data in its proper format.

#Data Standardization
# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-mpg'] = 235/df["city-mpg"]
df.rename(columns={'"city-mpg"':'city-L/100km'}, inplace=True)


# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df["highway-mpg"] = 235/df["highway-mpg"]

# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)


# data normalising : scaling :replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max() 

# show the scaled columns
#print(df[["length","width","height"]].head())

# Convert "horsepower" to numeric, coercing errors to NaN
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# Calculate the mean value for the "horsepower" column
avg_horsepower = df['horsepower'].mean(axis=0)

# Replace "NaN" with mean value
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

# Now safely convert "horsepower" to integer
df['horsepower'] = df['horsepower'].astype(int)

# Continue with your binning code
bins = np.linspace(min(df['horsepower']), max(df['horsepower']), 4)
group_names = ['Low', 'Medium', 'High']
df['horsepower_binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)

# Show the updated DataFrame
#print(df[['horsepower', 'horsepower_binned']].head(20))

# Count the number of instances in each bin
bin_counts = df['horsepower_binned'].value_counts()
#print(bin_counts)
# Sort the counts to align with the group_names order
sorted_bin_counts = bin_counts.reindex(['Low', 'Medium', 'High']).fillna(0)
#print(sorted_bin_counts)

# Creating the bar chart
plt.bar(group_names, sorted_bin_counts)

# Set the labels and title for the plot
plt.xlabel("Horsepower")
plt.ylabel("Count")
plt.title("Horsepower Bins")

# Display the plot
plt.show()

#dummy variables
#print(df.columns)

#pour duel type
dummy_var1=pd.get_dummies(df["fuel-type"])
# Convertir True/False en 0/1
dummy_var1 = dummy_var1.astype(int)
dummy_var1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
#print(dummy_var)

# merge data frame "df" and "dummy_var" 
df = pd.concat([df, dummy_var1], axis=1)
# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)
#print(df['fuel-type-gas'])

#pour aspiration
dummy_var2 = pd.get_dummies(df['aspiration'])
dummy_var2 = dummy_var2.astype(int)

# change column names for clarity
dummy_var2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)
print(dummy_var2)

# merge the new dataframe to the original datafram
df = pd.concat([df, dummy_var2], axis=1)

# drop original column "aspiration" from "df"
df.drop('aspiration', axis = 1, inplace=True)

df.to_csv('clean_df.csv')











