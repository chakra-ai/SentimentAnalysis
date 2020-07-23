# Import necessary packages.
import string
import re
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from PreProcessing import clean_doc, Cleaned_X_Y
from model import define_model

filename = "Roman Urdu DataSet.csv"

X, y = Cleaned_X_Y(filename)

# Count Vectorizer
cv = CountVectorizer(max_features=2500)

#Encode Label 
encoder = LabelEncoder()
onehot = OneHotEncoder(handle_unknown='ignore')

# Fit Transform X and y
X = cv.fit_transform(X).toarray()
print('Shape of the features X : ',X.shape)
y = encoder.fit_transform(y)
y = onehot.fit_transform(y.reshape(-1, 1)).toarray()
print('Shape of the Label y : ',y.shape)

# Train Test Split (0.8, 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

n_words = X_train.shape[1]
# Create Model
model = define_model(n_words)

# Train the model.
model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=100, verbose=2)


# Predict Outcome
pred = model.predict(X_test)

print(pred.shape)
print(type(pred))
y_pred = []
for i in range(pred.shape[0]):
    temp = np.argmax(pred[i])
    y_pred.append(temp)

print(np.array(y_pred))

y_test_n = []
for i in range(y_test.shape[0]):
    temp = np.argmax(y_test[i])
    y_test_n.append(temp)
print(np.array(y_test_n))

print('--------------------The Confusion Matrix---------------------')
print(confusion_matrix(y_test_n, y_pred))
