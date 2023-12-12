# Heart_Failure_Prediction_Logistic_Regression_Model
This model predicting whether a patient will experience a death event (1) or not (0) based on various features or input variables.

**__General View__**

I sourced the training data for this model from Kaggle and explored on the dataset. I analyzed the correlations among various features and visualized the insights using plots for better observation. Subsequently, I divided the dataset into training and testing groups, constructed a logistic regression model, and trained it using the existing data to collectively predict whether a death event is likely to occur. Finally, I conducted an analysis to evaluate the effectiveness of the model.

**__The code__**

**__#Explore the data__**

import pandas as pd

data = pd.read_csv('E:/Resume/机器学习数据模型/heart_failure.csv')

print(data.head())

print(data.info())

print(data.describe())

print(data['DEATH_EVENT'].unique())

print(data['DEATH_EVENT'].value_counts())

print(data.isnull().sum())

**__#Counting Statistics of Categorical Variables__**

categorical_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']
for col in categorical_cols:
    print(f"col_name: {col}")
    print(data[col].value_counts())
    print()

**__#Distribution of Target Variable__**

target_col = 'DEATH_EVENT'
print(f"Target Variable: {target_col}")
print(data[target_col].value_counts())

**__#Feature Correlation Analysis__**

correlation = data.corr()
print(correlation)

**__#Show the above three in plots__**

import matplotlib.pyplot as plt

target_col = 'DEATH_EVENT'

target_counts = data[target_col].value_counts()

x_ticks = [0, 1]

plt.bar(x_ticks, target_counts.values)
plt.xlabel(target_col)
plt.ylabel('Count')
plt.title('Distribution of Death Event')
plt.xticks(x_ticks)  
plt.show()

