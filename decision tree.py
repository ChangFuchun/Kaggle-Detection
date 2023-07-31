import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

datas = pd.read_csv("Suicide_Detection.csv", encoding="utf8")
datas.drop(["Unnamed: 0"], axis=1)
x = datas["text"]
y = datas["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#創建詞袋模式
vec = CountVectorizer()
#轉化成稀疏矩陣
x_train_count = vec.fit_transform(x_train)
x_test_count = vec.transform(x_test)

# n=21
# while n<30:    # 創建決策樹分類器並進行訓練
#     clf = DecisionTreeClassifier(max_depth=n)
#     clf.fit(x_train_count, y_train)

#     # 繪製決策樹
#     plt.figure(figsize=(12, 12))
#     plot_tree(clf, filled=True)
#     # plt.show()

#     #將測試資料丟入模型
#     pre = clf.predict(x_test_count)
#     print("Max_depth:", n, accuracy_score(y_test, pre))
#     n += 1

#k-fold驗證
clf = DecisionTreeClassifier(max_depth=30)

scores = cross_val_score(clf, x_train_count, y_train, cv=10, n_jobs=-1 )
print("十次分數:",scores)
print("average:", np.average(scores))