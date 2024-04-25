import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

file_path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"
df=pd.read_csv(file_path)

#print(df.head(5))

#print(df.dtypes)

#print(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr())

#sns.regplot(x="engine-size", y="price", data=df)
#plt.ylim(0,)
#plt.show()
#remarque : As the engine-size goes up, the price goes up: this indicates a positive direct correlation between these two variables

#print(df[["engine-size","price"]].corr()) #=0.87 proche a 1  

"""sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
plt.show()"""
#As highway-mpg goes up, the price goes down: this indicates an inverse/negative relationship between these two variables.

#df[['highway-mpg', 'price']].corr()  #=--0.704
"""
sns.regplot(x="peak-rpm", y="price", data=df)
#weak linear relationship
df[['peak-rpm','price']].corr()   #-0.1

sns.boxplot(x="body-style", y="price", data=df)

sns.boxplot(x="engine-location", y="price", data=df)

sns.boxplot(x="drive-wheels", y="price", data=df)
"""

df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
#print(grouped_test1)

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
#print(grouped_pivot)

grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0

plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

# Extracting the body style names which will be used as column tick labels.
row_labels = grouped_pivot.columns.levels[1]

# Getting the drive-wheel types which will be used as row tick labels.
col_labels = grouped_pivot.index

# Setting the positions of the ticks in the middle of each cell for the x-axis.
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)

# Setting the positions of the ticks in the middle of each cell for the y-axis.
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

# Applying the actual tick labels for the x-axis.
ax.set_xticklabels(row_labels, minor=False)

# Applying the actual tick labels for the y-axis.
ax.set_yticklabels(col_labels, minor=False)

# Rotating the x-axis tick labels by 90 degrees to prevent them from overlapping.
plt.xticks(rotation=90)

# Adding a color bar to the new figure to represent the scale.
fig.colorbar(im)

# Displaying the second heatmap with proper labels and color bar.
plt.show()

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  














