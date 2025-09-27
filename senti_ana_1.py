import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,recall_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

data=pd.read_csv('nlp/database/comments.csv',encoding='latin1')
df=pd.DataFrame(data)
df.columns=[i for i in range(1,16)]
df.drop(columns=[1,3,4,5,6,7,8,9,12,15],inplace=True)
df.columns=['airline_sentiment','retweet_count','text','tweet_created','tweet_loc']



vectoriser=TfidfVectorizer(stop_words='english')
text_vector=vectoriser.fit_transform(df['text'])
# df['text_vector']=[text_vector[i] for i in range(len(text_vector))]
# x=df[['retweet_count','text_vector','tweet_created','tweet_loc']]
x=text_vector
y=df['airline_sentiment'].map({'negative':2,'neutral':0,'positive':1}).to_numpy()


x_train,x_test,y_train,y_test=train_test_split(x,y)

# mapping={'negative':2,'neutral':0,'positive':1}
# df['target']=df['airline_sentiment'].map(mapping)
# df_train,df_test=train_test_split(df)
# x_train=vectoriser.fit_transform(df_train['text'])
# x_test=vectoriser.transform(df_test['text'])
# y_train=df_train['target']
# y_test=df_test['target']


model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(f"test accuracy : {model.score(x_test,y_test)}")
print(f"train accuracy : {model.score(x_train,y_train)}")
print(f"recall score : {recall_score(y_pred,y_test,average='macro')}")
print(f"f_1 score : {f1_score(y_pred,y_test,average='macro')}")