import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize

import re
import os
from pandas.core.series import Series

import seaborn as sns
import matplotlib as mat
import matplotlib.pyplot as plt


class Model:

    def __init__(self, model_name: str):
        self.bully_data = pd.read_csv("finaldata.csv")
        self.model = self.get_model(model_name)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def get_model(self, model_name: str):

        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(n_jobs=-1, n_estimators=50),
            'Support Vector Machine': SVC(C=1.0, kernel='linear', degree=3, gamma='auto'),
            "NavieBayes": MultinomialNB()
        }
        return classifiers[model_name]

    def process_data(self, tweet_text: str):
        """
        Remove hashtags, mentions, URLs, and stopwords from the input text.

        Parameters:
        - tweet_text (str): The input text from which hashtags, mentions, URLs, and stopwords should be removed.

        Returns:
        str: The processed text with hashtags, mentions, URLs, and stopwords removed.
        """

        tweet_text = re.sub(r'http\S+|www\S+|https\S+', '', tweet_text, flags=re.MULTILINE)
        tweet_text = re.sub(r'@\w+|\#\w+', '', tweet_text)
        tweet_text = re.sub(r'[^a-zA-Z\s]', '', tweet_text)
        tweet_text = tweet_text.lower()
        words = word_tokenize(tweet_text)
        clean_words = [word for word in words if word not in stopwords.words("english")]
        stem = PorterStemmer()
        processed_list = [stem.stem(word) for word in clean_words]
        " ".join(processed_list)
        return tweet_text

    def vectorize_data(self, X: Series, X_train: Series, X_test: Series):
        """
        Convert text data into TF-IDF vectorized format.

        Parameters:
        - X (pandas.Series): The entire text input data.
        - X_train (pandas.Series): The training text data.
        - X_test (pandas.Series): The testing text data.

        Returns:
        - X_train_vecs (scipy.sparse.csr_matrix): TF-IDF vectorized representation of X_train.
        - X_test_vecs (scipy.sparse.csr_matrix): TF-IDF vectorized representation of X_test.
        """
        
        vectorizer = TfidfVectorizer()
        vectorizer.fit(X)
        X_train_vecs = vectorizer.transform(X_train)
        X_test_vecs = vectorizer.transform(X_test)

        return X_train_vecs, X_test_vecs

    def gather_data(self):
        '''
        Initializes values to the the training and testing data
        '''

        X = self.bully_data["clean_tweet_text"]
        y = self.bully_data["cyberbullying_type"]

        X_train_pre, X_test_pre, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        self.X_train, self.X_test = self.vectorize_data(X, X_train_pre, X_test_pre)

    def train_model(self):
        
        self.gather_data()
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        
        return self.model
    
    def get_accuracy_score(self):

        if self.y_test is not None and self.y_pred is not None:
            return str(round(accuracy_score(self.y_test, self.y_pred) * 100, 2)) + " %"
        
        raise("The Model has not been trained yet")
    
    def vectorize_user_data(self, X, data):
        
        
        vectorizer = TfidfVectorizer()
        vectorizer.fit(X)
        vectorized_data = vectorizer.transform([data])

        return vectorized_data
    
    def get_corresponding_prediction(self, prediction):
        data_dictionary = [
            'Not Cyberbullying', 
            'Religion', 
            'Age', 
            'Gender', 
            'Ethnicity', 
            'Other Cyberbullying'
            ]
        return data_dictionary[prediction[0]]

    def predict_data(self, data):
        
        if self.y_test is not None and self.y_pred is not None:
            
            X = self.bully_data["clean_tweet_text"]
            processed_data = self.process_data(data)
            vectorized_data = self.vectorize_user_data(X,processed_data)
            prediction_index = self.model.predict(vectorized_data)

            prediction = self.get_corresponding_prediction(prediction_index)

            return prediction

        
        raise("The Model has not been trained yet")
    
    
    def get_confusion_matrix(self):

        if self.y_test is not None and self.y_pred is not None:
            mat.use('Agg')
            class_labels = {
                'NONE':0,
                'religion':1,
                'age':2,
                'gender':3,
                'race':4,
                'other':5
            }
            conf_matrix = confusion_matrix(self.y_pred, self.y_test)
            fig, ax = plt.subplots(figsize=(4, 4))
            map = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=None, xticklabels=class_labels, yticklabels=class_labels, cbar=False, ).get_figure()
            matrix_path = os.path.join('.','static', 'images', 'matrix.png')
            
            map.savefig(matrix_path)

def process_data(tweet_text: str):
        """
        Remove hashtags, mentions, URLs, and stopwords from the input text.

        Parameters:
        - tweet_text (str): The input text from which hashtags, mentions, URLs, and stopwords should be removed.

        Returns:
        str: The processed text with hashtags, mentions, URLs, and stopwords removed.
        """

        tweet_text = re.sub(r'http\S+|www\S+|https\S+', '', tweet_text, flags=re.MULTILINE)
        tweet_text = re.sub(r'@\w+|\#\w+', '', tweet_text)
        tweet_text = re.sub(r'[^a-zA-Z\s]', '', tweet_text)
        tweet_text = tweet_text.lower()
        words = word_tokenize(tweet_text)
        clean_words = [word for word in words if word not in stopwords.words("english")]
        stem = PorterStemmer()
        processed_list = [stem.stem(word) for word in clean_words]
        " ".join(processed_list)
        return tweet_text

def add_data(message, messagetype):
    data_dictionary = [
            'Not Cyberbullying', 
            'Religion', 
            'Age', 
            'Gender', 
            'Ethnicity', 
            'Other Cyberbullying'
            ]
    
    message = process_data(message)
    messagetype = data_dictionary.index(messagetype)

    mydata = pd.DataFrame({"clean_tweet_text": [message], "cyberbullying_type": [messagetype]})

    cleandata = pd.read_csv('clean_data.csv', index_col=0)
    finaldata = cleandata._append(mydata, ignore_index=True)
    try:
        finaldata.to_csv('finaldata.csv')
    except PermissionError as E:
        raise("Please close the local files")
        
