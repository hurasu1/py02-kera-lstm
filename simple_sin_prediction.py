# ライブラリのインポート
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import random
from matplotlib import pyplot as plt


# sinベクトルの学習・評価用データを生成する関数
def create_sin_dataset(time_sample_size=100, period=33, data_size=1000):
    input_data = []
    predict_data = []
    for i in np.arange(0, data_size):
        input_time = np.arange(i, i + time_sample_size, 1)
        predict_time = i + time_sample_size
        input_sin_vector = np.sin(2 * np.pi * input_time / period)
        output_sin_vector = np.sin(2 * np.pi * predict_time / period)

        # 結果を格納
        input_data.append(input_sin_vector)
        predict_data.append(output_sin_vector)

    # ndarrayに形式変換
    numpy_input_data = np.array(input_data).reshape(data_size, time_sample_size, 1)
    numpy_predict_data = np.array(predict_data).reshape(data_size, 1)
    return numpy_input_data, numpy_predict_data


# 学習データの生成

TIME_SAMPLE_DATA = 100
train_input, train_output = create_sin_dataset(TIME_SAMPLE_DATA)

# モデルの生成
model = Sequential()
model.add(LSTM(128, batch_input_shape=(None, TIME_SAMPLE_DATA, 1)))
model.add(Dense(1, activation="linear"))
model.compile(Adam(), loss="mean_squared_error")

early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
model.fit(train_input, train_output,
          batch_size=300,
          epochs=100,
          validation_split=0.1,
          callbacks=[early_stopping]
          )


# 実際にsinを評価してみる
temp = train_input[0, :, ]
predicted_result = temp
input_data = np.array(temp).reshape(1, temp.shape[0], temp.shape[1])
for i in np.arange(0, 12 * TIME_SAMPLE_DATA):
    print(i)
    predicted = model.predict(input_data)
    predicted_result = np.append(predicted_result, predicted[0])

    input_data = np.append(input_data, predicted[0])
    input_data = np.delete(input_data, 0)
    input_data = np.array(input_data).reshape(1, temp.shape[0], temp.shape[1])
    # plt.plot(input_data[0, :, ])


correct_data, _ = create_sin_dataset(12 * TIME_SAMPLE_DATA)
correct_data = correct_data[0, ]
plt.plot(correct_data)
plt.plot(predicted_result)

# error = np.array(correct_data).reshape(200,1) - np.array(predicted_result).reshape(200,1)
# plt.plot(error)
