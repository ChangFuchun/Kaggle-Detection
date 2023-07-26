import pandas as pd
import code
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



sd = pd.read_csv("data_train.csv", encoding="utf8")
sd_predit = pd.read_csv("data_test.csv", encoding="utf8")

sd = sd.drop(["Unnamed: 0"], axis=1)
sd_predit = sd_predit.drop(["Unnamed: 0"], axis=1)


#確認缺失值
s = sd.isna().sum()
p = sd_predit.isna().sum()


train = func.index(sd)
test = func.index(sd_predit)


x_train, y_train = func.split(train)
x_test, y_test = func.split(test)

#準備一個轉化器
vec = CountVectorizer()
x_train_count = vec.fit_transform(x_train)
x_test_count = vec.transform(x_test)


clf = MultinomialNB(alpha=0.2)
clf.fit(x_train_count, y_train)

pre = clf.predict(x_test_count)
print(accuracy_score(y_test, pre))

mat = confusion_matrix(y_test, pre)
print(mat)