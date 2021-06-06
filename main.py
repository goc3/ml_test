import numpy as np
import pandas as pd
import scipy.stats
from numpy import asarray, float64, remainder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#1. Read the file and display columns.
data = pd.read_csv('menu_info (1).csv', index_col=None)
print(list(data.columns.values))
print(data.head())
print(data.info())

#2. Calculate basic statistics of the data (count, mean, std, etc) and examine data and state your observations.

print(data.shape)
print(data.describe()) #most common descriptions, min max std etc.

range_personnel = ((data['Personnel_needed'].max()) - (data['Personnel_needed'].min()))
print('Personnel needed range is ' + str(range_personnel))
range_consumption = ((data['Consumption_duration'].max()) - (data['Consumption_duration'].min()))
print('Consumption duration range is ' + str(range_consumption))
range_price = ((data['Price'].max()) - (data['Price'].min()))
print('Price range is ' + str(range_price))
### Count == 158 for all columns (0 null values), Serving_duration has very low std (1.79), compared to mean (8.13).
### Spice_density is also interesting, akin to Serving_duration, other columns std's are almost 50% of their mean
### Large variations in range

#3. Select columns that will probably be important to predict “Personnel_needed” size.

data = data.drop(['Preparation_duration', 'Spice_density', 'Restaurant'], axis = 1)
print(data.head())

data.plot(x='Personnel_needed', y='Price', style='o')
plt.title('Price vs Personnel needed')
plt.xlabel('Personnel_needed')
plt.ylabel('Price')
plt.show()  

# Price seems to show linear correlation with Personnel_needed: the price of the dish is a better predictor of personnel needed

#4. If you removed columns, explain why you removed those.

print(data.corr()) ### Personnel_needed higly correlates with Waiting_duration (0.95), Price (0.92), 
### Consumption_duration (0.91) and Serving_duration (0.89). Dropping Spice_density, Restaurant and Preparation_duration.
print(data[data['Personnel_needed'] > 8]['Dish_name'].value_counts().head())
#print(data[data['Personnel_needed'] > 8]['Restaurant'].value_counts().head())
# Using 8 as a filter because of 50% value of 8.15

#5. Use one-hot encoding for categorical features.

dish_names = data['Dish_name']
dish_names_df = pd.DataFrame(dish_names, columns=['Dish_name'])
dum_dish_df = pd.get_dummies(dish_names_df, columns=['Dish_name'], prefix=['Dish_is'])
dish_names_df = dish_names_df.join(dum_dish_df)
data = data.drop(['Dish_name'], axis=1)
data = data.join(dish_names_df)
data = data.drop(['Dish_name'], axis=1)

#6. Create training and testing sets (use 60% of the data for the training and reminder for testing).

data = data.values
X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=25)

#7. Build a machine learning model to predict the ‘Personnel_needed’ size.

clf = LinearRegression()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
expected = y_test
print("RMS: %s" % np.sqrt(np.mean((predicted - expected) ** 2)))

#8. Calculate the Pearson correlation coefficient for the training set and testing datasets.

X = X.astype('float64')
y = y.astype('float64')
X = np.squeeze(X)
y = np.squeeze(y)

#r, p = scipy.stats.pearsonr(X, y)
#print(r)

#9. Describe hyper-parameters in your model and how you would change them to improve the performance of the model.

#10. Print answer to: What is regularization and what is the regularization parameter in your model?

#11. Plot regularization parameter value vs Pearson correlation for the test and training sets, and see whether your model 
# has a bias problem or variance problem.
