
# coding: utf-8

# # @author: Aman Jain

# In[1]:



import pandas as pd
from sklearn import metrics

dataset = pd.read_csv('Tweets.csv')
X = dataset["text"]
y = dataset["airline_sentiment"]


# ### Cleaning the texts
# 

# In[2]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 6918):
    review = re.sub('[^a-zA-Z]', ' ', X[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# ### Creating the Bag of Words model
# 

# In[3]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df = .85, max_features = 1500)
X = cv.fit_transform(X).toarray()


# ### Splitting the dataset into the Training set and Test set
# 

# In[4]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# ### Fitting Multinomial Naive Bayes to the Training set
# 

# In[5]:


from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X_train, y_train)
y_pred = NB.predict(X_test)
print('\nNaive Bayes')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')


# ### K-fold cross validation for Naive Bayes
# 

# In[6]:


from sklearn.model_selection import cross_val_score
accuracies_NB = cross_val_score(estimator = NB, X = X_train, y = y_train, cv = 10)
mean_NB = accuracies_NB.mean()
std_NB = accuracies_NB.std()


# ### Fitting SVM classifier to the Training set
# 

# In[9]:


from sklearn.svm import LinearSVC
SVM = LinearSVC()
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)
print('\nSupport Vector Machine')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')


# ### K-fold cross validation for SVM
# 

# In[10]:


from sklearn.model_selection import cross_val_score
accuracies_SVM = cross_val_score(estimator = SVM, X = X_train, y = y_train, cv = 10)
mean_SVM = accuracies_SVM.mean()
std_SVM = accuracies_SVM.std()


# ### # Fitting Linear Regression model to the Training set
# 

# In[11]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
print('\nLogistic Regression')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')


# ### K-fold cross validation for Logistic Regression

# In[12]:


from sklearn.model_selection import cross_val_score
accuracies_LR = cross_val_score(estimator = LR, X = X_train, y = y_train, cv = 10)
mean_LR = accuracies_LR.mean()
std_LR = accuracies_LR.std()


# ### Fitting K Nearest Neighbor classifier to the Training set
# 

# In[13]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 3)
KNN.fit(X_train, y_train)
y_pred = KNN.predict(X_test)
print('\nK Nearest Neighbors')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')


# ### K-fold cross validation for KNN
# 

# In[14]:


from sklearn.model_selection import cross_val_score
accuracies_KNN = cross_val_score(estimator = KNN, X = X_train, y = y_train, cv = 10)
mean_KNN = accuracies_KNN.mean()
std_KNN = accuracies_KNN.std()


# ### Fitting Decision Tree Classifier to the Training set
# 

# In[15]:


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)
print('\nDecision Tree')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')


# ### K-fold cross validation for Decision Tree
# 

# In[16]:


from sklearn.model_selection import cross_val_score
accuracies_DT = cross_val_score(estimator = DT, X = X_train, y = y_train, cv = 10)
mean_DT = accuracies_DT.mean()
std_DT = accuracies_DT.std()


# ### Fitting Random Forest Classifier to the Training set
# 

# In[17]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = 0)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
print('\nRandom Forest')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')


# ### K-fold cross validation for Random Forest
# 

# In[18]:


from sklearn.model_selection import cross_val_score
accuracies_RF = cross_val_score(estimator = RF, X = X_train, y = y_train, cv = 10)
mean_RF = accuracies_RF.mean()
std_RF = accuracies_RF.std()


# ## Analysing the model
# 

# In[19]:


token_words = cv.get_feature_names()
print('\n Analysis')
print('Number of tokens: ',len(token_words))
counts = NB.feature_count_
df_table = {'Token':token_words,'Negative': counts[0,:],'Positive': counts[1,:]}
tokens = pd.DataFrame(df_table, columns= ['Token','Positive','Negative'])
positives = len(tokens[tokens['Positive']>tokens['Negative']])
print('No. of positive tokens: ',positives)
print('No. of negative tokens: ',len(token_words)-positives)


# ### Check positivity/negativity of specific tokens
# 

# In[20]:


token_search = ['awesome']
print('\nSearch Results for token/s:',token_search)
print(tokens.loc[tokens['Token'].isin(token_search)])


# ### Analyse False Negatives (Actual: 1; Predicted: 0)(Predicted negative review for a positive review) 
# 

# In[21]:


print(X_test[ y_pred < y_test ])


# 
# ### Analyse False Positives (Actual: 0; Predicted: 1)(Predicted positive review for a negative review) 
# 

# In[22]:


print(X_test[ y_pred > y_test ])

