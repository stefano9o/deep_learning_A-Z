# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

script_dir = os.path.dirname(__file__)
abs_file_path = os.path.join(script_dir, 'Churn_Modelling.csv')

# Importing the dataset
dataset = pd.read_csv(abs_file_path)
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras import backend

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100, validation_split=0.1)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Predicting a new customer with the following paramteres as per exercise
#
"""Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000"""

# First metod: create a new csv files and rewxecute all the steps performed previously
abs_file_path = os.path.join(script_dir, 'single_prediction.csv')

# Importing the new dataset containing the single observation
dataset = pd.read_csv(abs_file_path)
X = dataset.iloc[:, 3:13].values

# Encoding categorical data using the same obj used before
X[:, 1] = labelencoder_X_1.transform(X[:, 1])
X[:, 2] = labelencoder_X_2.transform(X[:, 2])

X = onehotencoder.transform(X).toarray()
X = X[:, 1:]

# Feature Scaling using the same obj used before
X_new1 = sc.transform(X)

# Predicting
y_new_pred1 = classifier.predict(X_new1)

# Second method: hardcoding the new observation in a NumPy array

X = np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])

# Feature Scaling using the same obj used before
X_new2 = sc.transform(X)

# Predicting
y_new_pred2 = classifier.predict(X_new2)

y_new_pred = (y_new_pred2 > .5)

backend.clear_session()
