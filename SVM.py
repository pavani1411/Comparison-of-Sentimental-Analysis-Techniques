#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv('D:\Project\Train.csv')
df1 = pd.read_csv('D:\Project\Test.csv')
df.head()


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
train_vectors = vectorizer.fit_transform(df['text'])
test_vectors = vectorizer.transform(df1['text'])


# In[4]:


import time
from sklearn import svm
from sklearn.metrics import classification_report
# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, df['label'])
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(df1['label'], prediction_linear, output_dict=True)
print('positive: ', report['1'])
print('negative: ', report['0'])


# In[5]:


review = """Do not purchase this product. My cell phone blast when I switched the charger"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))


# In[6]:


review = """SUPERB, I AM IN LOVE IN THIS PHONE"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))


# In[7]:


review = """Things are about to change for you. May the overthinking, and the doubt exit your mind right now. May clarity replace confusion. May peace and calmness fill your life. You’ve been strong long enough, it’s time to start receiving your blessings. You deserve it."""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))


# In[9]:


review = """COVID is horrible"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))


# In[ ]:




