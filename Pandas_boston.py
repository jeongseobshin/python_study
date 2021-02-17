

# 보스턴 집값 예측
# https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv


import tensorflow as tf
import pandas as pd

path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(path)

print(boston.columns)
boston.head()


독립 = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'b', 'lstat']]
종속 = boston[['medv']]
print(독립.shape ,종속.shape)


X = tf.keras.layers.Input(shape=[13])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X,Y)
model.compile(loss='mse')



model.fit(독립, 종속, epochs = 10, verbose=0)
model.fit(독립, 종속, epochs = 10)


model.predict(독립[0:5])

종속[0:5]

#모델의 수식 확인
model.get_weights()

