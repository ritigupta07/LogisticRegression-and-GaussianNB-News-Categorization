from sklearn.datasets import fetch_20newsgroups

categories = ['rec.autos', 'rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

news_group = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'), categories=categories,shuffle=True, random_state=42)

# DATA VISUALIZATION
print(news_group.target_names)
print(news_group.target.shape)
print(news_group.DESCR)
print(news_group.filenames)
print(news_group.data[0])
print(news_group.target[0])
print(news_group.data[0].split('\n'))

# DATA PREPROCESSING
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stopWords = set(stopwords.words('english'))

preProcessedNews = []
for news in news_group.data:
  #remove numbers
  news = ''.join([i for i in news if not i.isdigit()])
  
  #conver to lower case
  news = news.lower()
  
  #remove special characters
  news = re.sub(r'[^\w]', ' ', news)
  
  #tokenize
  news_tokens = word_tokenize(str(news))
  
  #remove stop words
  filtered_news_tokens = [] 
  for w in news_tokens: 
    if w not in stopWords: 
        filtered_news_tokens.append(w)
        
  ps = PorterStemmer()
  stemmed_news_tokens = []
  for w in filtered_news_tokens:
    stemmed_news_tokens.append(ps.stem(w))

  filtered_news_str = ""
  for w in stemmed_news_tokens:
    filtered_news_str = filtered_news_str + w + " "
    
  preProcessedNews.append(filtered_news_str)
  news_group.data = preProcessedNews

# SPLITTING THE DATA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(news_group.data, news_group.target, test_size=0.3, random_state=11)

# FEATURE EXTRACTION BASED ON VECTORIZATION
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

# BUILDING OF MULTINOMIAL NAIVE BAYES
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.5).fit(X_train_counts, y_train)

# TESTING THE ACCURACY
X_test_counts = count_vect.transform(X_test)
y_pred_test = clf.predict(X_test_counts)

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' %accuracy_score(y_test, y_pred_test))

from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_pred_test))

# BUILDING OF MULTINOMIAL LOGISTIC REGRESSION
from sklearn import linear_model
reg = linear_model.LogisticRegression()
reg.fit(X_train_counts, y_train)

# TESTING THE ACCURACY
y_pred_test = reg.predict(X_test_counts)
print('Accuracy: %.2f' %accuracy_score(y_test, y_pred_test))

from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_pred_test))

# FEATURE EXTRACTION BASED ON VECTORIZATION
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# BUILDING OF MULTINOMIAL NAIVE BAYES
clf = MultinomialNB(alpha=0.5).fit(X_train_counts, y_train)
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)

# TESTING THE ACCURACY
y_pred_test = clf.predict(X_test_tfidf)

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' %accuracy_score(y_test, y_pred_test))

from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_pred_test))

# BUILDING OF MULTINOMIAL LOGISTIC REGRESSION
from sklearn import linear_model
reg = linear_model.LogisticRegression()
reg.fit(X_train_tfidf, y_train)

# TESTING THE ACCURACY
y_pred_test = reg.predict(X_test_tfidf)
print('Accuracy: %.2f' %accuracy_score(y_test, y_pred_test))
from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_pred_test))
