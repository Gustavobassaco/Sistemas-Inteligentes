import numpy as np
import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def data_normalizer(dataset):
    # Crie uma instância do MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Aplique a normalização nas colunas do dataset
    dataset = scaler.fit_transform(dataset)
    return dataset


def data_imputer(data):
    # Filtrar os elementos com valor 1 na última coluna
    data_ones = data[data[:, -1] == 1]

    # Calcular a média de cada coluna (exceto a última)
    column_means = np.mean(data_ones[:, :-1], axis=0)

    # Criar novos dados com base nas médias das colunas
    new_data = np.tile(column_means, (232, 1))

    # Definir a última coluna dos novos dados como 1 (representando a classe)
    new_data = np.hstack((new_data, np.ones((232, 1))))

    # Concatenar os novos dados com os dados existentes
    data = np.concatenate((data, new_data), axis=0)
    return data

def custom_weights(distances, sigma=1):
    weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    return weights

def dwnn_discrete_literatura(X_train, query, y_train, k=3, sigma=1):
    # Calcular as distâncias
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', metric='euclidean')
    nbrs.fit(X_train)
    distances, indices = nbrs.kneighbors([query])

    # Calcular os pesos personalizados
    weights = custom_weights(distances, sigma)

    # Calcular a soma ponderada das classes vizinhas
    weighted_sum = np.sum(weights * y_train[indices])

    # Calcular a classe predita
    prediction = round(weighted_sum / np.sum(weights))
    return prediction

def dwnn_discrete(dataset, query, res, k=3, sigma=1):
    # calcular a distância de query para cada padrão do dataset
    distance = np.zeros(dataset.shape[0])
    for i in range(dataset.shape[0]):
        # Distancia Euclidiana
        #distance[i] = np.sqrt(np.sum((dataset[i] - query)**2))
        # Distancia Manhattan
        distance[i] = np.sum(np.abs(dataset[i] - query))
    
    # calcula os pesos para cada instância usando uma função de base radial
    weights = np.exp(-(distance ** 2) / (2 * sigma ** 2))
    
    # retornar os k mais próximos
    idx = np.argsort(distance)[:k]
    candidates = res[idx]
    w = weights[idx]
    
    # calcular as distâncias para cada classe
    unique_classes = np.unique(res)
    total = np.zeros(len(unique_classes))
    
    # somar os pesos de cada instância para cada classe
    for i in range(len(candidates)):
        index = int(candidates[i])
        total[index] += w[i]
    
    # retornar o índice da classe com maior soma
    id_class = np.argmax(total)
    return unique_classes[id_class]


file_path = 'C:/Users/gusta/OneDrive/Documents/Python Scripts/Sistemas Inteligentes 1/Trabalho 3 - KNN e DWNN/dataset_37_diabetes.arff'
with open(file_path) as file:
    data = arff.load(file)
    
data_instances = data['data']
data_instances = np.array([list(instance[:-1]) + [0 if instance[-1] == 'tested_negative' else 1] for instance in data_instances])

# Normaliza os dados
data_instances = data_normalizer(data_instances)

# Imputa dados
data_instances = data_imputer(data_instances)


# Dividir em conjunto de treino e teste
X = np.array([instance[:-1] for instance in data_instances])
y = np.array([instance[-1] for instance in data_instances])

test_sizes = range(10, 70, 10)
ks = range(1, 13, 2)
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

accuracy_avg = []
interaction = 10
contk = 0
for k in ks:
    accuracy_sum = np.zeros(len(test_sizes))
    for l in range(interaction):
        print(l)
        accuracy = []
        for i, test_size in enumerate(test_sizes):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100)
            cont = 0
            for j in range(len(X_test)):
                ad = dwnn_discrete(X_train, X_test[j], y_train, k)
                if ad == y_test[j]:
                    cont += 1
            accuracy.append(cont / len(X_test) * 100)
            print('TestSize = ' + str(round(test_size, 3)) + '%, k = ' + str(k) + ', Porcentagem = ' +
                   str(round((cont / len(X_test)) * 100, 3)) + '%')
        accuracy_sum += np.array(accuracy)
    plt.plot(test_sizes, accuracy_sum / interaction, color=colors[contk], label='k = ' + str(k))
    contk += 1
plt.xlabel('Tamanho do Conjunto de teste (%)')
plt.ylabel('Porcentagem de acerto')
plt.title('Porcentagem de acerto média em 10 iterações\nvariando K e o conjunto de teste')
plt.legend()
plt.grid(True)
plt.show()