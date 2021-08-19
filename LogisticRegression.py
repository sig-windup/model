import numpy as np
import pandas as pd
import re

#csv가져오기 (경로 이름 변경)
open_csv = pd.read_csv('C:/Users/X-Note/Desktop/WIND-UP/기사/keyword_LT(20200505~20200731).csv', encoding='cp949')
data_keyword = np.array(open_csv['keyword'].tolist())
data_regression=np.array((open_csv['positive_or_negative'].tolist()))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(
    data_keyword, data_regression, test_size=0.3, random_state=156)
print(X_train.shape, X_test.shape)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

pipline=Pipeline([('cnt_vect', CountVectorizer(stop_words='english', ngram_range=(1,2))),
                  ('lt_clf', LogisticRegression(C=10))])

pipline.fit(X_train, y_train)
pred=pipline.predict(X_test)
pred_prods=pipline.predict_proba(X_test)[:, 1]
print(accuracy_score(y_test, pred), roc_auc_score(y_test, pred_prods))

pipline=Pipeline([('tfidf_vect', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
                  ('lt_clf', LogisticRegression(C=10))])

pipline.fit(X_train, y_train)
pred=pipline.predict(X_test)
pred_prods=pipline.predict_proba(X_test)[:, 1]
print(accuracy_score(y_test, pred), roc_auc_score(y_test, pred_prods))