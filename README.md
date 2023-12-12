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

**__#plot1_Death event distribution__**

import matplotlib.pyplot as plt

target_col = 'DEATH_EVENT'

target_counts = data[target_col].value_counts()

x_ticks = [0, 1]

plt.bar(x_ticks, target_counts.values)
plt.xlabel(target_col)
plt.ylabel('Count')
plt.title('Distribution of Death Event')
plt.xticks(x_ticks)  
plt.savefig('plot1.png')
plt.show()

**__#plot2_distribution of Numerical Features__**

numeric_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']

fig, axes = plt.subplots(nrows=len(numeric_cols), ncols=1, figsize=(8, 6 * len(numeric_cols)))

for i, col in enumerate(numeric_cols):
    axes[i].hist(data[col], bins=10)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'Distribution of {col}')

plt.tight_layout()

plt.savefig('plot2.png')

plt.show()

**__#plot3_Feature Correlation Heatmap__**

import seaborn as sns
correlation = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('plot3.png')
plt.show()

**__#Generate Logistic Regression Model__**

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

best_model.fit(X_train_scaled, y_train)

y_pred_scaled = best_model.predict(X_test_scaled)
print("Predicted Labels:", y_pred_scaled)

**__#Evaluate the model__**

print("Accuracy:", accuracy_score(y_test, y_pred_scaled))
print("Precision:", precision_score(y_test, y_pred_scaled))
print("Recall:", recall_score(y_test, y_pred_scaled))
