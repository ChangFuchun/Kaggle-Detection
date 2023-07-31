import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost import DMatrix



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


# 建立 XGBClassifier 模型
xgboostModel = XGBClassifier(n_estimators=100, learning_rate= 0.3)

# 使用訓練資料訓練模型
xgboostModel.fit(x_train_count, y_train)
# 使用訓練資料預測分類
predicted = xgboostModel.predict(x_test_count)

print(accuracy_score(y_test, predicted))

mat = confusion_matrix(y_test, predicted)
#創建一個DataFrame
print(pd.DataFrame(mat,
             columns=["{}(預測)".format(p) for p in z],
             index=["{}(正確)".format(p) for p in z]
             ))