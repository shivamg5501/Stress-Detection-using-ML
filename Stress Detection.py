#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd


# In[11]:


df=pd.read_csv('D:/ML Project/stress.csv')
df.head()


# In[12]:


df.describe()
#describe function used for summarization of the dataset


# In[13]:


#for checking the null values in my dataset, we use isnull
df.isnull().sum()
# 0 means there are no  null value in data set


# In[14]:


#cleaning of data from dataset, using ragular expresion.
import nltk
import re
from nltk. corpus import stopwords
import string
nltk. download( 'stopwords' )#stopwords to showcase the extracted words from the dataset.
stemmer = nltk. SnowballStemmer("english")
stopword=set (stopwords . words ( 'english' ))

def clean(text):
    text = str(text) . lower()  #returns a string where all characters are lower case. Symbols and Numbers are ignored.
    text = re. sub('\[.*?\]',' ',text)  #substring and returns a string with replaced values.
    text = re. sub('https?://\S+/www\. \S+', ' ', text)#whitespace char with pattern
    text = re. sub('<. *?>+', ' ', text)#special char enclosed in square brackets
    text = re. sub(' [%s]' % re. escape(string. punctuation), ' ', text)#eliminate punctuation from string
    text = re. sub(' \n',' ', text)
    text = re. sub(' \w*\d\w*' ,' ', text)#word character ASCII punctuation
    text = [word for word in text. split(' ') if word not in stopword]  #removing stopwords
    text =" ". join(text)
    text = [stemmer . stem(word) for word in text. split(' ') ]#remove morphological affixes from words
    text = " ". join(text)
    return text
df [ "text"] = df["text"]. apply(clean)


# In[15]:


#using matplotlib, and wordcloud for show casing highlighted words in image 
#according to their frequency of words in dataset.
import matplotlib. pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text = " ". join(i for i in df. text)
stopwords = set (STOPWORDS)
wordcloud = WordCloud( stopwords=stopwords,background_color="white") . generate(text)
plt. figure(figsize=(10, 10) )#size of the shown image
plt. imshow(wordcloud )
plt. axis("off")#doesn't show the x&y axese.
plt. show( )#for showing the generated output


# In[16]:


#countvectorizer converted the textual data to binary vector
#formate according to their occurence which is part of scikit learn library. 
from sklearn. feature_extraction. text import CountVectorizer
from sklearn. model_selection import train_test_split

x = np.array (df["text"])
y = np.array (df["label"])

cv = CountVectorizer ()
X = cv. fit_transform(x)#fit_transform is function which is accepting x value(text array) to fit data calculating the mean & standard deviation for scaling which is stored in X.    
print(X)#mapped with the labeled data.
xtrain, xtest, ytrain, ytest = train_test_split(X, y,test_size=0.33)#splitting the data into 70% & 30% for training & testing respectively.
#can also use hype-rperameter random_state=30, which split data from no.30 onwards,which can impact the performance.


# In[17]:


#distribution of Bernoulli's classify algorithm works on binary values from dataset.

from sklearn.naive_bayes import BernoulliNB
model=BernoulliNB()
model.fit(xtrain,ytrain)
#output - BernoulliNB(), shows that the model is ready.


# In[23]:


user=input("Enter the text")#input from user
data=cv.transform([user]).toarray()#converting the input textual data into array. 
output=model.predict(data)#predict will create new set of data, passing data in predict.The output id depended on the dataset & stored in the output variable
if output>0:
    print("Stressed person")

else:
    print("Chill Dude")

#[output - 1]-shows stressed
#[output - 0]-shows not-stressed


# In[ ]:




