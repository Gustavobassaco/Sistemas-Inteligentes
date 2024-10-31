# Baixar dados para um ativo específico
data = yf.download('TSLA', start='2010-01-01', end='2012-01-01')
data = data['Close'].values.reshape(-1, 1)

plt.plot(data, linestyle='--',  color='b')
data = scaler.fit_transform(data)

train_size = int(len(data) * 0.70)
test_size  = len(data) - train_size
train, test = data[0:train_size,:], data[train_size:len(data),:]

trainX, trainY = create_dataset(train, janela_previsao)
np.column_stack((trainX, trainY))
testX, testY  = create_dataset(test, janela_previsao)
np.column_stack((testX, testY))

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(units=4, input_shape=(1, janela_previsao)))
model.add(Dense(1))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=["MeanSquaredError", "RootMeanSquaredError", "MeanAbsoluteError"]
)
history = model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)



plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['mean_absolute_error'])
plt.xlabel('epoch')
plt.legend(['MSE', 'RMSE', 'MAE'], loc='upper right')
plt.show()


trainPredict = model.predict(trainX)
testPredict  = model.predict(testX)
np.column_stack((testY, testPredict))

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[janela_previsao:len(trainPredict)+janela_previsao, :] = trainPredict

testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(janela_previsao*2)+1:len(data)-1, :] = testPredict

plt.plot(trainPredictPlot,  linestyle="--")
plt.plot(testPredictPlot,  linestyle="--")
plt.plot(scaler.inverse_transform(data))
plt.xlabel('instances')
plt.legend(['train', 'test', 'original'], loc='upper left')
plt.show()
















