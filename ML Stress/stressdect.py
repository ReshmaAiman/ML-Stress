import numpy as np
import pandas as pd

df=pd.read_csv('C:\\Users\\Resh\\stress.csv')
print(df.head())

print(df.describe())

print(df.isnull())

print(df.isnull().sum())

import nltk
import re
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
stemmer=nltk.SnowballStemmer('english')
stopword=set(stopwords.words('english'))
def clean(text):
    text=str(text).lower() #returns a string where all charcters are at lower case.Symbols and Numbers are ignored
    text=re.sub('\[.*?\]',' ',text) #substring and returns a string with replaced values
    text=re.sub('https?://\S+/www\. \S+',' ',text) #whitespace character with pattern
    text=re.sub('<. *?>+',' ',text)#Special chracter enclosed in special square brackets
    text= re.sub(' [%s]' % re.excape(string,punctuation), ' ', text) #eliminate punctuation from string)
    text=re.sub(' \n',' ',text)
    text=re.sub(' \w*\d\w*',' ',text) # word character ASCII punctuation
    text=[word for word in text.split(' ')if word not in stopword] #remove stopwords
    text=" ".join(text)
    text=[stemmer.stem(word) for word in text.split(' ')] #remove morphological affixes from words
    text=" ".join(text)
    return text
    #df[ "text"] =["text"].apply(clean)

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text=" ".join(i for i in df.text)
stopwords=set(STOPWORDS)
word_cloud=WordCloud(stopwords=stopwords,background_color="white").generate(text)
#print(plt.figure(figsize=(15,10)))
#print(plt.imshow(WordCloud,interpolation='bilinear'))
#plt.axis("off")
#print(plt.show())
img = word_cloud.to_image()
img.show()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
x=np.array(df["text"])
y=np.array(df["label"])
cv=CountVectorizer()
X=cv.fit_transform(x)
print(X)
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.33,random_state=42)

from sklearn.naive_bayes import BernoulliNB
model=BernoulliNB()
model.fit(xtrain,ytrain)

user=input("Enter the text\n")
data=cv.transform([user]).toarray()
output=model.predict(data)
print(output)