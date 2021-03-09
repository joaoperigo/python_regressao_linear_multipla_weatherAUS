import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import precision_score

dataset = pd.read_csv('weatherAUS.csv')
print (dataset)


#independent ==== features
registro = dataset.iloc[:, [12, 14, 16, 18, 20, 22]].dropna()
print(registro.shape)

independent = registro.iloc[:, :-1]
dependent = pd.get_dummies(registro['RainTomorrow'])  

x = independent.values.reshape(-1, 5)
y = dependent.values.reshape(-1, 2)

ind_train, ind_test, dep_train, dep_test = train_test_split (x, y, test_size = 0.2, random_state = 0)

#obtencao dos coeficientes desejados
#no pdf, na listagem 4.8.1 faltou esta linha
linearRegression = LinearRegression()
linearRegression.fit(ind_train, dep_train)

dep_pred = linearRegression.predict(ind_test)

previsao = np.argmax(dep_pred, axis=1)
teste = np.argmax(dep_test, axis=1)

print(precision_score(teste, previsao, average='micro'))

print(linearRegression.coef_)


