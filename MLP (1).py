import numpy as np

class MLP(object):
    """
    Uma rede neural (Perceptron-multilayer) de 3 camadas
    """

    def __init__(self, entrada, oculta, saida, taxaDeAprendizado=0.1):
        """
        Entrada : número de entradas, de neurônios ocultos e saídas. Podendo também variar a taxa de aprendizado
        """
        self.entrada = entrada
        self.oculta = oculta
        self.saida = saida
        self.taxaDeAprendizado = taxaDeAprendizado

        # Inicialize os pesos e bias aleatoriamente
        self.pesos_input_oculta = np.random.rand(self.entrada, self.oculta)
        self.bias_oculta = np.random.rand(self.oculta)
        self.pesos_oculta_saida = np.random.rand(self.oculta, self.saida)
        self.bias_saida = np.random.rand(self.saida)

    def getTaxaDeAprendizado(self):
        return self.taxaDeAprendizado

    def setTaxaDeAprendizado(self, taxa):
        self.taxaDeAprendizado = taxa

    def ativacaoSigmoidal(self, valor):
        """
        Função ativadora Sigmoidal = 1 / (1 + e ^ - valor)
        Entrada : Valor a ser aplicado na função
        Retorno : Resultado da aplicação
        """
        return 1 / (1 + np.exp(-valor))

    def derivadaAtivacaoSigmoidal(self, valor):
        """
        Derivada da função ativadora Sigmoidal , dSigmoidal / dValor = Sigmoidal *(1 - Sigmoidal)
        Entrada : Valor(Resultante da aplicação à sigmoidal) a ser aplicado na função
        Retorno : Resultado da aplicação
        """
        return valor * (1 - valor)

    def erroQuadraticoMedio(self, esperado, valor):
        """
        Cálculo do erro
        Entrada : O target e o valor deduzido
        Retorno : Erro calculado dadas as entradas
        """
        return np.mean(np.square(esperado - valor))

    def feedForward(self, entrada):
        """
        Recebe as entradas e faz a classificação
        Entrada : As N entradas(float) definidas no __init__
        Retorno : Nenhum
        """
        # Camada oculta
        self.saida_oculta = self.ativacaoSigmoidal(np.dot(entrada, self.pesos_input_oculta) + self.bias_oculta)

        # Camada de saída
        self.saida_rede = self.ativacaoSigmoidal(np.dot(self.saida_oculta, self.pesos_oculta_saida) + self.bias_saida)

        return self.saida_rede

    def backPropagation(self, entrada, esperado):
        """
        Pondera as classificações e faz as correções aos pesos
        Entrada : Targets(float)
        Retorno : Nenhum
        """
        # Cálculo do erro na camada de saída
        erro_saida = esperado - self.saida_rede
        delta_saida = erro_saida * self.derivadaAtivacaoSigmoidal(self.saida_rede)

        # Cálculo do erro na camada oculta
        erro_oculta = delta_saida.dot(self.pesos_oculta_saida.T)
        delta_oculta = erro_oculta * self.derivadaAtivacaoSigmoidal(self.saida_oculta)

        # Atualização dos pesos e bias
        self.pesos_oculta_saida += self.saida_oculta.T.dot(delta_saida) * self.taxaDeAprendizado
        self.bias_saida += np.sum(delta_saida) * self.taxaDeAprendizado
        self.pesos_input_oculta += entrada.T.dot(delta_oculta) * self.taxaDeAprendizado
        self.bias_oculta += np.sum(delta_oculta) * self.taxaDeAprendizado

    def treinamento(self, entrada, esperado, epocas):
        for _ in range(epocas):
            for i in range(len(entrada)):
                entrada_atual = entrada[i]
                esperado_atual = esperado[i]

                # Feedforward
                saida = self.feedForward(entrada_atual)

                # Backpropagation
                self.backPropagation(entrada_atual, esperado_atual)

                # Cálculo do erro
                erro = self.erroQuadraticoMedio(esperado_atual, saida)
                print(f'Época {_}, Amostra {i}, Erro: {erro}')
