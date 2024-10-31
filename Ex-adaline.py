import numpy as np
import matplotlib.pyplot as plt

# Entradas e saídas desejadas
entrada = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
entrada = np.array(entrada)

X = entrada[:, :-1]
y = entrada[:, -1]

# Normalização das características (padronização)

# Inicialização dos parâmetros
np.random.seed(0)
weights = np.random.random(X.shape[1] + 1)
learning_rate = 0.01
n_epochs = 100

# Função de ativação (identidade)
def activation(x):
    return x

# Lista para armazenar os erros quadráticos médios da época
epoch_errors = []

# Treinamento do Adaline
for epoch in range(n_epochs):
    output = np.dot(X, weights[1:]) + weights[0]
    errors = (y - output)
    weights[1:] += learning_rate * X.T.dot(errors)
    weights[0] += learning_rate * errors.sum()
    
    # Cálculo do erro quadrático médio da época
    epoch_error = (errors ** 2).mean()
    epoch_errors.append(epoch_error)
    
    print(f"Época {epoch+1}/{n_epochs} - Erro Quadrático Médio: {epoch_error:.4f}")

# Função para previsão
def predict(x):
    return 1 if activation(np.dot(x, weights[1:]) + weights[0]) >= 0.5 else 0

# Teste das previsões
for i in range(len(X)):
    prediction = predict(X[i])
    print(f"Entrada: {X[i]}, Saída desejada: {y[i]}, Saída prevista: {prediction}")

# Plotar o hiperplano de decisão e o erro quadrático médio por época
plt.figure(figsize=(6, 6))

# Plotar o hiperplano de decisão

plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', marker='o', label='Classe 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='x', label='Classe 1')

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = np.array([predict([xi, yi]) for xi, yi in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.title('Hiperplano de Decisão do Adaline')

plt.tight_layout()
plt.show()
