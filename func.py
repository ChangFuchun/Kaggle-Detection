import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#丟掉不需要的資料
def drop_data(self):
    self = self.drop(["Unnamed: 0"], axis=1)
    return self

#資料轉換
def index(df):
    mapping = {"suicide":0, "non-suicide":1}
    df["class"] = df["class"].replace(mapping)
    return df

#資料分割
def split(sp):
    x = sp.iloc[:, 0]  
    y = sp.iloc[:, 1:] 
    return x,y