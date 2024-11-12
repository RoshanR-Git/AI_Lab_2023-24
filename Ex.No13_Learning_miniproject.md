# Ex.No: 10 Learning – Use Supervised Learning  
### DATE: 11-11-2024                                                                           
### REGISTER NUMBER : 212222060203
### AIM: 
To write a program to train the classifier for Loan Repayment Predictor.
###  Algorithm:
1.Data Collection and Preprocessing: Gather and clean loan data (e.g., credit score, income), handling missing values and encoding categories.

2.Feature Selection and Engineering: Choose key features like debt-to-income ratio to improve prediction accuracy.

3.Split Data into Training and Test Sets: Split data into 80% training and 20% test sets for model validation.

4.Train Decision Tree Classifier: Train the Decision Tree on training data, tuning parameters for accuracy.

5.Evaluate Model Performance: Test and assess the model’s accuracy, precision, and recall on the test set.

6.Deploy Model in Chatbot: Integrate the model in a chatbot to predict loan repayment likelihood based on user inputs.

### Program:
```
#Importing Dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#Reading the CSV File

filename = (r"C:\Users\rrosh\OneDrive\Desktop\ML dataset\Loan repayment_ Dataset.csv")
df = pd.read_csv(filename,sep = ',',header = 0)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

print("Dataset Length : ",len(df))

print("Dataset : ",df.shape)

df.head()

df.isnull()

df.isnull().sum()

#Seperate the target variable

X = df.values[:, 0:4]
Y = df.values[:, 5]


X.shape

Y.shape

import seaborn as sns
import matplotlib.pyplot as plt
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Plot the credit score distribution
sns.histplot(df['Credit Score'],kde = True)
plt.xlabel('Credit Score')
plt.ylabel('Count')
plt.show()


#Split the Dataset into train and test

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 100)

print(X_train.shape)
X_test.shape

print(y_train.shape)
y_test.shape

#Model Implementation

model = DecisionTreeClassifier(criterion = "entropy",random_state = 100,max_depth = 3,min_samples_leaf = 5)
model.fit(X_train,y_train)

#Prediction

y_pred = model.predict(X_test)
print(y_pred)


#Checking Accuracy

print("Accuracy is ",accuracy_score(y_test,y_pred)*100)
```

### Output:

![Screenshot 2024-11-12 222834](https://github.com/user-attachments/assets/512d1090-9e60-45d8-a858-8bc001e31c5b)

### Result:
Thus the system was trained successfully and the prediction was carried out.
