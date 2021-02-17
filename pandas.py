# import tensorflow as tf
# import pandas as pd

# #   파일에서 데이터 읽어오기
# path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
# data = pd.read_csv(path)
# data.head()

# 독립 = data[['온도']]
# 종속 = data[['판매량']]
# print(독립.shape, 종속.shape)

# X = tf.keras.layers.Input(shape=[1])
# Y = tf.keras.layers.Dense(1)(X)
# model = tf.keras.models.Model(X, Y)
# model.compile(loss='mse')

# model.fit(독립, 종속, epochs=10, verbose=0) # verbose 화면출력안하기

# model.predict([[15]])

# #보스턴 집값 예측
