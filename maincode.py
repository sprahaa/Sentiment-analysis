import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
import string
from nltk.corpus import stopwords
import os
print(os.listdir("C:/Users/91982/Desktop/NTCC"))
df= pd.read_csv("C:/Users/91982/Desktop/NTCC/Reviews.csv")
df.head()
df.columns
df.describe()
df.info()
data1=pd.DataFrame(df.groupby('Score').size().sort_values(ascending=False).rename('No of Users').reset_index())
data1.head()
data2 = df[['Score' , 'Text' , 'Summary' , 'UserId']]
finaldata=data2.dropna()
finaldata.head()
finaldata['Summary'].fillna("Good", inplace = True)
finaldata.isnull().sum()
finaldata["Sentiment"] = finaldata["Score"].apply(lambda score: "positive" if score > 3 else "negative")
finaldata['Sentiment'] = finaldata['Sentiment'].map({'positive':1, 'negative':0})
finaldata.columns
Senti = finaldata[(finaldata['Sentiment']== 0) | (finaldata['Sentiment']==1)]
y = Senti['Sentiment']
x = Senti['Summary']
print(x)
text = finaldata['Summary']
labels = finaldata['Sentiment']
stop = set(stopwords.words('english'))
def clean_document(doco):
    punctuation = string.punctuation
    punc_replace = ''.join([' ' for s in punctuation])
    doco_link_clean = re.sub(r'http\S+', '', doco)
    doco_clean_and = re.sub(r'&\S+', '', doco_link_clean)
    doco_clean_at = re.sub(r'@\S+', '', doco_clean_and)
    doco_clean = doco_clean_at.replace('-', ' ')
    doco_alphas = re.sub(r'\W +', ' ', doco_clean)
    trans_table = str.maketrans(punctuation, punc_replace)
    doco_clean = ' '.join([word.translate(trans_table) for word in doco_alphas.split(' ')])
    doco_clean = doco_clean.split(' ')
    p = re.compile(r'\s*\b(?=[a-z\d]*([a-z\d])\1{3}|\d+\b)[a-z\d]+', re.IGNORECASE)
    doco_clean = ([p.sub("", x).strip() for x in doco_clean])
    doco_clean = [word.lower() for word in doco_clean if len(word) > 2]
    doco_clean = ([i for i in doco_clean if i not in stop])
    doco_clean = ([p.sub("", x).strip() for x in doco_clean])
    return doco_clean

clean = [clean_document(doc) for doc in text];
sentence = [' '.join(r) for r in clean ]
print(sentence[2])
print(text[2])

def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc) 
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

tokens = x[0].split()
print(tokens)
sample_text = "Hey there! This is a project"
print(text_process(sample_text))
vector = CountVectorizer(analyzer=text_process).fit(x)
len(vector.vocabulary_)
sample=x[20]
v = vector.transform([sample])
X = vector.transform(x)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=101) 
from sklearn.linear_model import LogisticRegression
LRmodel = LogisticRegression()
LRmodel.fit(X_train, y_train)
pred= LRmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, pred)
print(cm)
print('\n')
print(classification_report(y_test, pred))
LRmodel.score(X_train, y_train)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, pred)
print(accuracy)
from sklearn.metrics import recall_score
recall = recall_score(y_test, pred, average=None)
print("Recall :",recall)
from sklearn.metrics import precision_score
precision = precision_score(y_test, pred, average=None) 
print("Precision :",precision)
f1s = 2 * (precision * recall) / (precision + recall)
print("F-1 Score :",f1s)
pred1 = Senti['Summary'][10]
print(pred1)
a = Senti['Sentiment'][10]
print(a)
op = vector.transform([pred1])
sam1= LRmodel.predict(op)[0]
print(sam1)
