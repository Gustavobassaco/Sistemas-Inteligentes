import arff
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def preprocess(data):
    # Extrair informações do cabeçalho
    attributes = data['attributes']
    data_instances = data['data']
    # Converter atributos categóricos em numéricos
    le = LabelEncoder()
    for attr_index, attr in enumerate(attributes):
        attr_type = attr[1]
        if isinstance(attr_type, list):
            # Atributo categórico
            attr_values = [instance[attr_index] for instance in data_instances]
            encoded_values = le.fit_transform(attr_values)
            for instance_index, instance in enumerate(data_instances):
                instance[attr_index] = encoded_values[instance_index]
    return data_instances

def knn_regression_literatura(dataset, query, res, k):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(dataset, res)
    y_pred = knn.predict([query])
    return y_pred[0]

def knn_regression(dataset, query, res, k):
    # Calcular a distância de query para cada padrão do dataset
    distance = np.zeros(dataset.shape[0])
    for i in range(dataset.shape[0]):
        # Distância Euclidiana
        # distance[i] = np.sqrt(np.sum((dataset[i] - query)**2))
        # Distância Manhattan
        distance[i] = np.sum(np.abs(dataset[i] - query))

    # Retornar os k mais próximos
    idx = np.argsort(distance)[:k]
    k_nearest_res = res[idx]

    # Calcular a média dos valores dos k vizinhos mais próximos
    mean_value = np.mean(k_nearest_res)
    return mean_value


file_path = 'C:/Users/gusta/OneDrive/Documents/Python Scripts/Sistemas Inteligentes 1/Trabalho 3 - KNN e DWNN/dataset.arff'
with open(file_path) as file:
    data = arff.load(file)

data_instances = preprocess(data)


# Dividir em conjunto de treino e teste
X = np.array([instance[:-1] for instance in data_instances])
y = np.array([instance[-1] for instance in data_instances])

test_sizes = range(10, 50, 10)
ks = range(1, 9, 2)
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

mse_avg = []
interaction = 1
contk = 0
for k in ks:
    mse_sum = np.zeros(len(test_sizes))
    for l in range(interaction):
        print(l)
        mse = []
        for i, test_size in enumerate(test_sizes):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
            y_pred = []
            for j in range(len(X_test)):
                prediction = knn_regression(X_train, X_test[j], y_train, k)
                y_pred.append(prediction)
            mse.append(mean_squared_error(y_test, y_pred))
            print('TestSize = ' + str(round(test_size, 3)) + '%, k = ' + str(k) + ', MSE = ' +
                   str(round(mean_squared_error(y_test, y_pred), 3)))
        mse_sum += np.array(mse)
    plt.plot(test_sizes, mse_sum / interaction, color=colors[contk], label='k = ' + str(k))
    contk += 1
plt.xlabel('Tamanho do Conjunto de teste (%)')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.title('MSE médio variando K e o conjunto de teste')
plt.legend()
plt.grid(True)
plt.show()