import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost import DMatrix
from sklearn.model_selection import cross_val_score



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

# n_estimators: 總共迭代的次數，即決策樹的個數。預設值為100。
# max_depth: 樹的最大深度，默認值為6。
# booster: gbtree 樹模型(預設) / gbliner 線性模型
# learning_rate: 學習速率，預設0.3。
# gamma: 懲罰項係數，指定節點分裂所需的最小損失函數下降值。
# tree_method:{'approx', 'auto', 'exact', 'gpu_hist', 'hist'}

# 建立 XGBClassifier 模型
xgboostModel = XGBClassifier(n_estimators=300, learning_rate= 0.3, tree_method="hist")

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

scores = cross_val_score(xgboostModel, x_test_count, y_test, cv=10, n_jobs=-1 )
print("十次分數:",scores)
print("average:", np.average(scores))