import numpy as np
import matplotlib.pyplot as plt  


class AdalineSGD:
    def __init__(self, n_iter=100, learning_rate=0.0001, classification=True):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.classification = classification

    def train(self, dataset):
        # Preparação dos dados
        X = np.insert(dataset, 0, 1, axis=1)
        class_id = X.shape[1] - 1
        y = X[:, class_id].astype(np.float64)
        X = X[:, :-1]

        # Inicialização dos pesos aleatoriamente entre -0.5 e 0.5
        w = np.random.uniform(-0.5, 0.5, size=X.shape[1])

        # Inicialização de vetores para rastrear o custo e erro por época
        error = np.zeros(self.n_iter)
        misc = np.zeros(self.n_iter)

        # Loop por cada época
        for n in range(self.n_iter):
            print("* Epoch:", n)

            # Loop por cada ponto de dados em ordem aleatória
            indices = list(range(X.shape[0]))
            for i in indices:

                z = np.dot(w, X[i, :])

                k = w
                for m in range(len(w)):
                    k[m] = w[m] + (self.learning_rate*(y[m] - z)*X[i][m])
                w = k

            # Cálculo da função de custo da época
            error[n] = np.sum((y - np.dot(X, w)) ** 2) / X.shape[0]
            print(" - error:", error[n], " - misc:", misc[n])
            
        # Retorna o modelo
        model = {
            "epochs": self.n_iter,
            "misc": misc,
            "learning_rate": self.learning_rate,
            "error": error,
            "weights": w
        }

        return model

    @staticmethod
    def predict(example, weights):
        example = np.insert(example, 0, 1)
        z = np.dot(example, weights)
        y = 1 if z >= 0 else -1
        return y

    @staticmethod
    def regression(example, weights):
        example = np.insert(example, 0, 1)
        z = np.dot(example, weights)
        return z
    
# Crie uma instância do AdalineSGD
adaline = AdalineSGD(n_iter=1000, learning_rate=0.01, classification=False)

# Defina os dados de entrada
entrada = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
entrada = np.array(entrada)

# Separe as características (X) das classes (y)
X = entrada[:, :-1]
y = entrada[:, -1]

# Treine o modelo Adaline com gradiente descendente estocástico
modelo = adaline.train(entrada)

# Realize previsões usando o modelo treinado
for i in range(X.shape[0]):
    exemplo = X[i, :]
    previsao = adaline.predict(exemplo, modelo['weights'])
    print(f"Exemplo: {exemplo}, Previsão: {previsao}, Classe Real: {y[i]}")

# Plotar os pontos de entrada
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', marker='o', label='Classe 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='x', label='Classe 1')

# Traçar a reta gerada pelo Adaline
x_line = np.linspace(-0.1, 1.1, 100)
y_line = (-modelo['weights'][0] - modelo['weights'][1] * x_line) / modelo['weights'][2]
plt.plot(x_line, y_line, color='green', label='Reta de decisão')

# Configurar o gráfico
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')

# Mostrar o gráfico
plt.show()