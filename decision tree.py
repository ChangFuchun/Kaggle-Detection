import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# 其他程式碼

# 丟掉不需要的資料
def drop_data(df):
    df = df.drop(["Unnamed: 0"], axis=1)
    return df

# 資料轉換
def index(df):
    mapping = {"suicide": 0, "non-suicide": 1}
    df["class"] = df["class"].replace(mapping)
    return df

# 資料分割
def split(sp):
    x = sp.iloc[:, 0]  
    y = sp.iloc[:, 1:] 
    return x, y

# 其他程式碼

# 載入資料
train = pd.read_csv("data_train.csv", encoding="utf8")
test = pd.read_csv("data_test.csv", encoding="utf8")

train = drop_data(train)
test = drop_data(test)

train = index(train)
test = index(test)

x_train, y_train = split(train)
x_test, y_test = split(test)

#創建詞袋模式
vec = CountVectorizer()
x_train_count = vec.fit_transform(x_train)
x_test_count = vec.transform(x_test)

n=2
while n<21:    # 創建決策樹分類器並進行訓練
    clf = DecisionTreeClassifier(max_depth=n)
    clf.fit(x_train_count, y_train)

    # 繪製決策樹
    plt.figure(figsize=(12, 12))
    plot_tree(clf, filled=True)
    # plt.show()

    #將測試資料丟入模型
    pre = clf.predict(x_test_count)
    print("Max_depth:", n, accuracy_score(y_test, pre))
    n += 1

#k-fold驗證
clf = DecisionTreeClassifier(max_depth=10)

scores = cross_val_score(clf, x_train_count, y_train, cv=10, n_jobs=-1 )
print("十次分數:",scores)
print("average:", np.average(scores))