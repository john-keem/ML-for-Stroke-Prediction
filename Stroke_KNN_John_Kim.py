#Created by JOHN KIM

# Utilize numpy to find the mean of each evaluation list
import numpy as np
# Utilize pandas library to create and manipulate the dataframe containing the stroke dataset
import pandas as pd
# This will be used to help create and initialize the KNN model
from sklearn.neighbors import KNeighborsClassifier
# Use classification report, accuracy score, precision score, recall score for evaluation
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, roc_auc_score
# StratifiedKFold will be used to help with the evaluation process
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns


#Data Preprocessing
# Put the stroke dataset into a pandas dataframe
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Irrelevant column is dropped ('user.id')
df = df.drop('id', axis=1)

# Display first 5 rows of the dataframe
print(df.head())


#Visualize nullity of dataset

# Count non-null values in each column
not_null = strokeData.notnull().sum()

# Plot the bar graph
plt.figure(figsize = (11,9))
not_null.plot(kind='bar', color='pink', edgecolor='black')

# Customize the plot
plt.title('Valid Values per Column')
plt.xlabel('Columns')
plt.ylabel('Count of Valid Values')
plt.xticks(rotation=30, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=1)

# Show the plot
plt.show()

#Show the balance of the dataset
# Count the occurrences of each class
class_counts = strokeData['stroke'].value_counts()

# Plot the bar graph
plt.figure(figsize=(6, 4))
class_counts.plot(kind='bar', color=['pink', 'violet'], edgecolor='black')

# Customize the plot
plt.title('Distribution of Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['No-Stroke', 'Stroke'], fontsize=10, rotation=0)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# KDE plot for "Feature" grouped by "Output"
sns.kdeplot(data=strokeData[strokeData['stroke'] == 0],
            x='age',
            fill=True,
            color='violet',
            alpha=0.75)
sns.kdeplot(data=strokeData[strokeData['stroke'] == 1],
            x='age',
            fill=True,
            color='pink',
            alpha=0.75)

# Customize the plot
plt.title('KDE Plot of Age Grouped by Stroke')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend(['No-Stroke', 'Stroke'],title="Stroke")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# KDE plot for "Feature" grouped by "Output"
sns.kdeplot(data=strokeData[strokeData['stroke'] == 0],
            x='bmi',
            fill=True,
            color='violet',
            alpha=0.75)
sns.kdeplot(data=strokeData[strokeData['stroke'] == 1],
            x='bmi',
            fill=True,
            color='pink',
            alpha=0.75)

# Customize the plot
plt.title('KDE Plot of Bmi Grouped by Stroke')
plt.xlabel('Bmi')
plt.ylabel('Density')
plt.legend(['No-Stroke', 'Stroke'],title="Stroke")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# KDE plot for "avg_glucose_level" grouped by "Stroke"
sns.kdeplot(data=strokeData[strokeData['stroke'] == 0],
            x='avg_glucose_level',
            fill=True,
            color='violet',
            alpha=0.75)
sns.kdeplot(data=strokeData[strokeData['stroke'] == 1],
            x='avg_glucose_level',
            fill=True,
            color='pink',
            alpha=0.75)

# Customize the plot
plt.title('KDE Plot of avg_glucose_level Grouped by Stroke')
plt.xlabel('avg_glucose_level')
plt.ylabel('Density')
plt.legend(['No-Stroke', 'Stroke'],title="Stroke")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# Use pandas 'get_dummies' function to one-hot encode categorical columns
# These columns will be: gender, ever_married, work_type, Residence_type, smoking_status
df = pd.get_dummies(df, columns=['gender','ever_married','work_type','Residence_type','smoking_status'])

# Deal with missing values in the dataset
# Fill missing values in 'bmi' column with the mean values
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Split healthcare-dataset-stroke-data.csv into attributes and labels
# x contains all data except for data in the stroke column (last column)
x = df.drop('stroke', axis=1)
# y contains the target column, which is the 'stroke' column
y = df['stroke']

# Perform undersample of majority class since data is very imbalanced
# Retrieve the indices of the minority class
min_class = df[df['stroke'] == 1] # minority class
maj_class = df[df['stroke'] == 0] # majority class

# Sample by random from majority class the same amount of samples as the minority class contains
under_maj_class = maj_class.sample(n=len(min_class))

# Merge minority class w/ undersampled majority class
under = pd.concat([under_maj_class, min_class])

# Shuffle the merged dataset
under = under.sample(frac=1).reset_index(drop=True)

# Separate newly 'acquired' data into target and features
X_under = under.drop('stroke', axis=1) # contains all data except 'stroke' column
y_under = under['stroke'] # contains only 'stroke' column (target)

# See how many samples are in the dataset
print('\nNumber of Samples:',len(under),'\n')

#KNN Model Implementation
# 'K = 22' is sqrt(n), which is a common heuristic
classifier = KNeighborsClassifier(n_neighbors=22)
# Initialize parameter 'n_neighbors', K will essentially be n_neighbors
# By default, KNN will utilize Euclidean distance unless specified otherwise
# Hence, this program will utilize Euclidean distance for the KNN algorithm


#Model Evaluation
# Set up Stratified 10-fold cross-validation
# Use Stratified instead of regular K-fold so each fold maintains the same class distribution as the original dataset
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=40)
# 'n_splits' splits the data into 10 different train-test sets
# 'shuffle=True' randomly shuffles the data before each fold split, adding randomness for more robust results
# 'random_state=40' ensures that the data shuffling process can be reproduced (40 is an arbitrary choice)

# Lists to store accuracy, precision, recall, AUC, classification reports
accuracy_list = [] # accuracy list
precision_list = [] #precision list
recall_list = [] # recall list
auc_list = [] # AUC (Area Under Curve) list
class_report_list = [] # classification report list

# Perform stratified 10-fold cross-validation
# Loop through each fold in cross-validation
for train_index, test_index in kf.split(X_under, y_under):
    # Split data into training and testing sets for this fold
    X_train, X_test = X_under.iloc[train_index], X_under.iloc[test_index] # split X into train and test
    y_train, y_test = y_under.iloc[train_index], y_under.iloc[test_index] # split y into train and test

    # Fit the model (train the model on the training data, in this case KNN)
    classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = classifier.predict(X_test) # generate predictions for the test set

    # Store accuracy in list for this fold
    accuracy_list.append(accuracy_score(y_test, y_pred))
    # Store precision in list for this fold
    precision_list.append(precision_score(y_test, y_pred))
    # Store recall in list for this fold
    recall_list.append(recall_score(y_test, y_pred))
    # Store AUC in list for this fold
    auc_list.append(roc_auc_score(y_test, y_pred))
    # Store classification report for this fold
    class_report_list.append(classification_report(y_test, y_pred))

# Loop through the classification reports and output each fold's classification report
for i, report in enumerate(class_report_list):
    print(f"Fold {i + 1}:\n{report}")

# Display average accuracy across 10 folds
print(f"Average Accuracy: {np.mean(accuracy_list):.4f}")
# Display average precision across 10 folds
print(f"Average Precision: {np.mean(precision_list):.4f}")
# Display average recall across 10 folds
print(f"Average Recall: {np.mean(recall_list):.4f}")
# Display average Area Under Curve across 10 folds
print(f"Average AUC: {np.mean(auc_list):.4f}")
# Display number of nearest neighbors utilized for this model
print(f"Number of Nearest Neighbors(K):", 22)