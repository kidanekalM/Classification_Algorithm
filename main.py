import pandas as pd
data = pd.read_csv("filename.csv")
x = data.iloc[:,0:13]
y = data.iloc[:,13]