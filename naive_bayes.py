import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

datas = pd.read_csv("Suicide_Detection.csv", encoding="utf8")
datas.drop(["Unnamed: 0"], axis=1)


#將資料轉化為list
z = datas["class"].value_counts().index
#準備字典替換
z = {i:p for p, i in enumerate(z)}
y = datas["class"].replace(z)
x = datas["text"]
#資料拆分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#準備一個轉化器
vec = CountVectorizer()
#轉化成稀疏矩陣
x_train_count = vec.fit_transform(x_train)
x_test_count = vec.transform(x_test)
clf = MultinomialNB(alpha=0.2)
clf.fit(x_train_count, y_train)

pre = clf.predict(x_test_count)
print(accuracy_score(y_test, pre))



mat = confusion_matrix(y_test, pre)
#創建一個DataFrame
print(pd.DataFrame(mat,
             columns=["{}(預測)".format(p) for p in z],
             index=["{}(正確)".format(p) for p in z]
             ))