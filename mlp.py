import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf


class MLP(object):
  # Uma rede neural(Perceptron-multilayer) de 3 camadas
  def __init__(self,
               entrada: int,
               oculta: int,
               saida: int,
               taxaDeAprendizado: float = 0.1) -> None:
    #Entrada : número de entradas, de neurônios ocultos e saidas.
    #Podendo também variar a taxa de aprendizado

    self.entrada = entrada
    self.oculta = oculta
    self.saida = entrada
    self.taxaDeAprendizado = taxaDeAprendizado

    #matriz de pesos para entradas na camada oculta e bias
    self.Wh = np.random.random_sample((self.oculta, self.entrada + 1)) - 0.5
    #matriz de pesos para entradas na camada de saída e bias
    self.Wo = np.random.random_sample((self.saida, self.oculta + 1)) - 0.5

  def getTaxaDeAprendizado(self):
    return self.taxaDeAprendizado

  def setTaxaDeAprendizado(self, taxa):
    self.taxaDeAprendizado = taxa

  def getWeights(self):
    return np.concatenate((self.Wh.flatten(), self.Wo.flatten()))

  def setWeights(self, weights):
    hidden_weights_size = (self.entrada + 1) * self.oculta
    self.Wh = weights[:hidden_weights_size].reshape(self.oculta, self.entrada + 1)
    self.Wo = weights[hidden_weights_size:].reshape(self.saida, self.oculta + 1)

  def ativacaoSigmoidal(self, valor):
    #Função ativadora Sigmoidal = 1 / (1 + e ^ - valor)
    #Entrada : Valor a ser aplicado na função
    #Retorno : Resultado da aplicação
    return 1 / (1 + np.exp(-valor))

  def derivadaAtivacaoSigmoidal(self, valor):
    #Derivada da função ativadora Sigmoidal , dSigmoidal / dValor = Sigmoidal *(1 - Sigmoidal)
    #Entrada : Valor(Resultante da aplicação à sigmoidal) a ser aplicado na função
    #Retorno : Resultado da aplicação
    return self.ativacaoSigmoidal(valor) * (1 - self.ativacaoSigmoidal(valor))

  def erroQuadraticoMedio(self, esperado, valor):
      """
      Cálculo do erro
      Entrada : O target e o valor deduzido
      Retorno : Erro calculado dadas as entradas
      """
      return np.mean(np.square(esperado - valor))


  def feedForward(self, dados):
    #Recebe as entradas e faz a classificação
    #Entrada : As N entradas(float) definidas no __init__
    #Retorno : Nenhum
    if len(dados) != self.entrada:
      print("Numero de entrada errado!")
      exit(1)

    dados = np.array(dados)

    saidas_oculta = np.array([])
    saidas_final = np.array([])

    dados = np.append(dados, 1)
    for i in range(self.oculta):
      y = self.ativacaoSigmoidal(np.matmul(dados, self.Wh[i]))
      saidas_oculta = np.append(saidas_oculta, y)

    saidas_oculta = np.append(saidas_oculta, 1)
    for i in range(self.saida):
      y = self.ativacaoSigmoidal(np.matmul(saidas_oculta, self.Wo[i]))
      saidas_final = np.append(saidas_final, y)

    return saidas_oculta, saidas_final

  def backPropagation(self, dados, esperado, saidas_oculta, saidas_final):
    #Pondera as classificações e faz as correções aos pesos
    #Entrada : Targets(float)
    #Retorno : Nenhum

    dados = np.append(dados, 1)
    valores_esperados = np.zeros(self.saida)
    valores_esperados[esperado] = 1
    
    #Cálculo dos gradientes
    grad_saida = np.array([])
    for i in range(self.saida):
        grad_saida = np.append(grad_saida, (valores_esperados[i] - saidas_final[i]) * self.derivadaAtivacaoSigmoidal(saidas_final[i]))

    # grad_oculta = []
    # for j in range(self.oculta):
    #     grad_oculta.append(self.derivadaAtivacaoSigmoidal(saidas_oculta[j]) * np.sum(grad_saida[:, np.newaxis] * self.Wo[:,j]))
    grad_oculta = []
    for j in range(self.oculta):
        soma_ponderada = np.sum(grad_saida * self.Wo[:,j])
        grad_oculta.append(self.derivadaAtivacaoSigmoidal(saidas_oculta[j]) * soma_ponderada)

    #Atualização dos pesos da camada de saída
    for i in range(self.saida):
      for j in range(self.oculta + 1):
        self.Wo[i,j] += self.getTaxaDeAprendizado() * grad_saida[i] * saidas_oculta[j]

    #Atualização dos pesos da camada oculta
    for i in range(self.oculta):
      for j in range(self.entrada + 1):
        self.Wh[i,j] += self.getTaxaDeAprendizado() * grad_oculta[i] * dados[j]

  def treinamento(self, dados, esperado, epocas):
    erro = []
    erro_epoca = []
    for epoca in range(epocas):
      print(f'\nEPOCA {epoca+1}')
      for i, exemplo in enumerate(dados):
        saidas_oculta, saidas_final = self.feedForward(exemplo)
        if self.saida > 1:
            valores_esperados = np.zeros(self.saida)
            valores_esperados[esperado[i]] = 1
        else:
           valores_esperados = esperado[i]
        
        erro.append(self.erroQuadraticoMedio(exemplo,saidas_final))
        self.backPropagation(exemplo, esperado[i], saidas_oculta, saidas_final)
      erro_epoca.append(np.mean(erro))
      print(f': Erro: {erro_epoca[epoca]}\n')

    return erro_epoca
  
  def predict(self, dado):
    _, y = self.feedForward(dado)
    return np.argmax(y)
    


if __name__ == '__main__':  
  X, y = load_iris(return_X_y = True)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

  lr = 0.01
  epocas = 1000

  m = MLP(4, 2, 4, taxaDeAprendizado=lr)
  erro_epoca = m.treinamento(X_train, y_train, epocas=epocas)
  plt.plot(range(epocas), erro_epoca)
  plt.show()

  y_pred = []
  for amostra in X_test:
      y_pred.append(m.predict(amostra))
  y_pred = np.array(y_pred)
  print(f'\n\n\n{y_pred}\n\n\n')
  print(f'\n\n{y_test}\n\n')
  print(accuracy_score(y_test, y_pred))

  # dados = np.array([
  #   [0,0,1],
  #   [0,1,0],
  #   [1,0,0],
  #   [1,1,1]
  # ])

  # X = np.delete(dados, 2, 1)
  # y = dados[:, -1]

  # m = MLP(4, 2, 4, taxaDeAprendizado=0.1)
  # erro_epoca = m.treinamento(X, y, 10)
  # plt.plot(range(10), erro_epoca)
  # plt.show()
  # y = m.predict([1,1])
  # print(f'Saida: {y}')
