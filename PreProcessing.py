# Import needed packages
import re
import string
import numpy as np
import pandas as pd

# Urdu Stopwords
stopwords=['ai', 'ayi', 'hy', 'hai', 'main', 'ki', 'tha', 'koi', 
           'ko', 'sy', 'woh', 'bhi', 'aur', 'wo', 'yeh', 'rha', 'hota', 
           'ho', 'ga', 'ka', 'le', 'lye', 'kr', 'kar', 'lye', 'liye', 
           'hotay', 'waisay', 'gya', 'gaya', 'kch', 'ab', 'thy', 'thay', 
           'houn', 'hain', 'han', 'to', 'is', 'hi', 'jo', 'kya', 'thi', 
           'se', 'pe', 'phr', 'wala', 'waisay', 'us', 'na', 'ny', 'hun', 
           'rha', 'raha', 'ja', 'rahay', 'abi', 'uski', 'ne', 'haan', 
           'acha', 'nai', 'sent', 'photo', 'you', 'kafi', 'gai', 'rhy', 
           'kuch', 'jata', 'aye', 'ya', 'dono', 'hoa', 'aese', 'de', 
           'wohi', 'jati', 'jb', 'krta', 'lg', 'rahi', 'hui', 'karna', 
           'krna', 'gi', 'hova', 'yehi', 'jana', 'jye', 'chal', 'mil', 
           'tu', 'hum', 'par', 'hay', 'kis', 'sb', 'gy', 'dain', 'krny', 'tou']

# load document into the memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# Clean the text with stopword removal, lower case conversion, etc. 
# turn a document into clean tokens
def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('',w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    #tokens = [word for word in tokens if not word in stopwords]
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if len(word)>2]
    return tokens


def Cleaned_X_Y(filename):
    # Import data
    df = pd.read_csv(filename, encoding='utf8', header=None)
    df.columns = ['text','target','junk']
    df.drop('junk',axis=1, inplace=True)
    df.dropna(inplace=True)
    data = df[df['target'] != 'Neative']

    #Get X data
    corpus = []
    for i in range(data.shape[0]):
        review = data.iloc[:,0].values[i]
        review = clean_doc(review)
        review=' '.join(review)
        corpus.append(review)

    X = np.array(corpus)

    y = data.iloc[:,1].values

    return X, y


def get_reviews(data, positive=True):
    label = 1 if positive else 0

    reviews = []
    labels = []
    data_senti = data[data['target']=='Positive']['text'] if positive else data[data['target']=='Negative']['text']
    data_senti = np.array(data_senti)
    #print(data_senti)
    for review in data_senti:
        reviews.append(review)

    for review in reviews:
        labels.append(label)
    
    return reviews, labels

def extract_label_data(data):
    positive_reviews, positive_labels = get_reviews(data, positive=True)
    negative_reviews, negative_labels = get_reviews(data, positive=False)

    data = positive_reviews + negative_reviews
    labels = positive_labels + negative_labels

    return labels, data
