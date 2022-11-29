#--------------------------------------------about dataset
#I explained how you can build a Spam Mail prediction system using Machine Learning with Python. 
# This is one of the most important Machine Learning Projects. 
#-----------------------------------------------work  flow
#dataset download
#data anlysis
#data separtion
#train model
#model selction
#predict data
#---------------------------------------------import useful labrary
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer #this is used for to convert string data into numerical data 
#--------------------------------------------dataset anlysis
data = pd.read_csv("C:/Users/kunde/all vs code/ml prject/mail_data.csv")
print(data.head(5))
print(data.tail(5))
print(data.describe())
print(data.info())
print(data.isnull().sum())
#if there is null data here u can use following function
data = data.where((pd.notnull(data)), '')
print(data.shape)
print(data.columns)
data.replace({"Category" : {"spam" : 0, "ham" : 1}}, inplace=True)
print(data.head(5))
#--------------------------------------------data separtion
x = data["Message"]
y = data["Category"]
#feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase="True")#min_df is used to get minmum repeation of word is 1, stop_words is used to not get (a, the , or, and)
#x_numerical = feature_extraction.fit_transform(x)
#print(x_numerical.shape)
#-------------------------------------------data train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(x.shape, x_train.shape, x_test.shape)
print(y.shape, y_train.shape, y_test.shape)
#-------------------------------------------convert data into numerical form 
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase="True")#min_df is used to get minmum repeation of word is 1, stop_words is used to not get (a, the , or, and)
x_numerical = feature_extraction.fit_transform(x_train)
x_test_num = feature_extraction.transform(x_test)
#our y value is not numerical we have to convert into numerical 
y_train = y_train.astype("int")
y_test = y_test.astype("int")
#-------------------------------------------data model selection
model = LogisticRegression()
model.fit(x_numerical, y_train)

#-----------------------------------------train data prediction 
y_tr = model.predict(x_numerical)
accur = accuracy_score(y_tr, y_train)
print(accur)

#-----------------------------------------test data prediction
y_te = model.predict(x_test_num)
accur = accuracy_score(y_te, y_test)
print(accur)

#------------------------------------------single data prediction
x = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times."]
#ham
x_num = feature_extraction.transform(x)
y_pred = model.predict(x_num)
print(y_pred, "this is our prediction", "true value is ham")
