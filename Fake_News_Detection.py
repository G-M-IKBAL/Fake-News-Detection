# Importing Dependencies
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import nltk
import matplotlib.pyplot as plt

# Stop words will be downloaded (such words that do not add much value to our data set i.e. 'I' 'is' 'being')
nltk.download('stopwords')
# Read dataset csv
dataset = pd.read_csv('train.csv')
print(dataset.isnull().sum())
# For Cleaning the dataset we'll impute the missing values with empty strings
dataset = dataset.fillna('')
print(dataset.isnull().sum())
# Combining dataset columns
dataset['content'] = dataset['author'] + ' ' + dataset['title']
print(dataset['content'])
# We'll perform stemming to reduce words to their root words.
port_stem = PorterStemmer()


# Function of stemming
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# stemming
dataset['content'] = dataset['content'].apply(stemming)
print(dataset['content'])
# Separating content & label
X = dataset['content'].values
Y = dataset['label'].values
# converting the textual data to numerical data using vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
# Splitting dataset to training dataset & testing dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8)
# KNN Classifier for prediction and model training
KNN_Classifier = KNeighborsClassifier(n_neighbors=1)
KNN_Classifier.fit(X_train, Y_train)
X_train_prediction = KNN_Classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data in KNN Classifier : ', training_data_accuracy*100)
X_test_prediction = KNN_Classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data in KNN Classifier : ', test_data_accuracy*100)
# Logistic Regression for prediction and model training
Logistic_Regression = LogisticRegression()
Logistic_Regression.fit(X_train, Y_train)
X_train_prediction = Logistic_Regression.predict(X_train)
training_data_accuracy_L = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data in LogisticRegression : ', training_data_accuracy_L*100)
X_test_prediction = Logistic_Regression.predict(X_test)
test_data_accuracy_L = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data in LogisticRegression : ', test_data_accuracy_L*100)
p1 = test_data_accuracy*100
p2 = test_data_accuracy_L*100
p3 = training_data_accuracy*100
p4 = training_data_accuracy_L*100

# now we'll get new dataset and test the trained model on it.
# Read new dataset csv
new_dataset = pd.read_csv('test.csv')
temp = pd.read_csv('submit.csv', usecols=['label'])
new_dataset['label'] = temp
print(new_dataset.isnull().sum())
# For Cleaning the dataset we'll impute the missing values with empty strings
new_dataset = new_dataset.fillna('')
print(new_dataset.isnull().sum())
# Combining dataset columns
new_dataset['content'] = new_dataset['author'] + ' ' + new_dataset['title']
print(new_dataset['content'])
# We'll perform stemming to reduce words to their root words.
# stemming
new_dataset['content'] = new_dataset['content'].apply(stemming)
print(new_dataset['content'])
# Separating content & label
X = new_dataset['content'].values
Y = new_dataset['label'].values
# converting the textual data to numerical data using vectorizer
X = vectorizer.transform(X)
# Using KNN Classifier to predict
X_test_prediction = KNN_Classifier.predict(X)
test_data_accuracy = accuracy_score(X_test_prediction, Y)
print('Accuracy score of the new dataset in KNN Classifier : ', test_data_accuracy*100)
# Using Logistic Regression to predict
X_test_prediction = Logistic_Regression.predict(X)
test_data_accuracy_L = accuracy_score(X_test_prediction, Y)
print('Accuracy score of the new dataset in Logistic Regression : ', test_data_accuracy_L*100)
# Plotting Accuracy
p5 = test_data_accuracy*100
p6 = test_data_accuracy_L*100
# data to plot
n_groups = 2
LG = (p1, p2)
KNN = (p3, p4)
New_data = (p5, p6)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
plot1 = plt.bar(index, LG, bar_width, alpha=opacity, color='b', label='Test Data')
plot2 = plt.bar(index + bar_width, KNN, bar_width, alpha=opacity, color='g', label='Training Data')
plot3 = plt.bar(index + bar_width + bar_width, New_data, bar_width, alpha=opacity, color='k', label='New Data')
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Classifier Scores')
plt.xticks(index + bar_width, ('KNN Classifier', 'Logistic Regression'))
plt.legend()

plt.tight_layout()
plt.show()
