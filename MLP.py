from Demo import FlappyBird_Human
import numpy as np

class MLP(object):
	"""
	Uma rede neural(Perceptron-multilayer) de 3 camadas
	"""
	def __init__(self,entrada = 4, oculta=5, saida=1, taxaDeAprendizado = 0.1, fb = None, pesos_entrada_oculta = [], pesos_oculta_saida = [], bias_oculta = [], bias_saida = []):
		"""
		Entrada : número de entradas, de neurônios ocultos e saidas. Podendo também variar a taxa de aprendizado
		"""
		self.taxaDeAprendizado = taxaDeAprendizado
		self.tam_entrada = entrada
		self.tam_oculta = oculta
		self.tam_saida = saida

		if pesos_entrada_oculta is []:
			self.pesos_entrada_oculta = np.random.uniform(size=(self.tam_entrada, self.tam_oculta))
			self.pesos_oculta_saida = np.random.uniform(size=(self.tam_oculta, self.tam_saida))
			self.bias_oculta = np.zeros((1, self.tam_oculta))
			self.bias_saida = np.zeros((1, self.tam_saida))

		else:
			self.pesos_entrada_oculta = pesos_entrada_oculta
			self.pesos_oculta_saida = pesos_oculta_saida
			self.bias_oculta = bias_oculta
			self.bias_saida = bias_saida
		self.fit = 0

		self.fb = fb if fb else FlappyBird_Human()
	
	def getTaxaDeAprendizado(self):
		return self.taxaDeAprendizado	
		
	def setTaxaDeAprendizado(self, taxa = 0.1):
		self.taxaDeAprendizado = taxa	

	def ativacaoSigmoidal(self, valor):
		"""
		Função ativadora Sigmoidal = 1 / (1 + e ^ - valor)
		Entrada : Valor a ser aplicado na função
		Retorno : Resultado da aplicação
		"""
		return 1 / (1+ np.exp(-valor))
			

	def derivadaAtivacaoSigmoidal(self, valor):
		"""
		Derivada da função ativadora Sigmoidal , dSigmoidal / dValor = Sigmoidal *(1 - Sigmoidal)
		Entrada : Valor(Resultante da aplicação à sigmoidal) a ser aplicado na função
		Retorno : Resultado da aplicação
		"""
		return valor * (1 - valor)	

	def erroQuadraticoMedio(esperado, valor):
		"""
		Calculo do erro
		Entrada : O target e o valor deduzido
		Retorno : Erro calculado dadas as entradas
		"""		
		return (np.subtract(valor, esperado)**2).mean()	


	def feedForward(self, X):
		"""
		Recebe as entradas e faz a classificação
		Entrada : As N entradas(float) definidas no __init__
		Retorno : Nenhum
		"""
        # Camada oculta
		self.hidden_input = np.dot(X, self.pesos_entrada_oculta)
		self.hidden_input += self.bias_oculta
		self.hidden_output = self.ativacaoSigmoidal(self.hidden_input)

        # Camada de saída
		self.output_input = np.dot(self.hidden_output, self.pesos_oculta_saida)
		self.output_input += self.bias_saida
		self.final_output = self.ativacaoSigmoidal(self.output_input)



	def backPropagation(self, esperado =[]):
		"""
		Pondera as classificações e faz as correções aos pesos
		Entrada : Targets(float)
		Retorno : Nenhum
		"""
		Xq = self.fb.get_distances()
		X = np.array(Xq).reshape(1, -1)
		Y = self.fb.shouldJump()

		# Calcula o erro na camada de saída
		output_error = Y - self.final_output
		output_delta = output_error * self.derivadaAtivacaoSigmoidal(self.final_output)

        # Calcula o erro na camada oculta
		hidden_error = output_delta.dot(self.pesos_oculta_saida.T)
		hidden_delta = hidden_error * self.derivadaAtivacaoSigmoidal(self.hidden_output)

        # Atualiza os pesos e biases
		self.pesos_oculta_saida += self.hidden_output.T.dot(output_delta) * self.taxaDeAprendizado
		self.bias_saida += np.sum(output_delta, axis=0, keepdims=True) * self.taxaDeAprendizado

		self.pesos_entrada_oculta += X.T.dot(hidden_delta) * self.taxaDeAprendizado
		self.bias_oculta += np.sum(hidden_delta, axis=0, keepdims=True) * self.taxaDeAprendizado

		

	
	def treinamento(self):
		alive = True
		while alive:
			Xa = self.fb.get_distances()
			X = np.array(Xa).reshape(1, -1)

			self.feedForward(X)
			#print('-' * 20)
			#print(self.final_output, end='  ')
			#print('record: '+ str(self.fb.max) + '  mortes: ' + str(self.fb.mortes) + '   atual: ' + str(self.fb.counter))
			a = False
			if self.final_output >= 0.5: 
				a = True
			self.fb.run_step(a)
			
			# é possivel deixar o backpropagation somente para quando ainda n fez 10 pontos
			# ou tambem desativar para que os pesos sejam inteiramente pelo AG
			if not self.fb.dead :#and self.fb.counter < 10:
				#self.backPropagation()
				pass

			# criteiros de parada
			if self.fb.dead or self.fb.counter > 100 or self.fb.dead2:
				alive = False
				self.fit = self.fb.fit
				print('fitness: ' + f"{self.fit:.3f}" + '  max: ' + str(self.fb.max) + '  rodada: ' + str(self.fb.counter))
				self.fb.restart()
		
		return self.fit, self.pesos_entrada_oculta, self.pesos_oculta_saida, self.bias_oculta, self.bias_saida
	
	def jogo(self):
		alive = True
		while alive:
			Xa = self.fb.get_distances()
			X = np.array(Xa).reshape(1, -1)

			self.feedForward(X)
			a = False
			if self.final_output >= 0.5: 
				a = True
			self.fb.run_stepG(a)
			
			if self.fb.dead or self.fb.dead2:
				self.fit = self.fb.fit
				print('fitness: ' + f"{self.fit:.3f}" + '  max: ' + str(self.fb.max) + '  rodada: ' + str(self.fb.counter))
				self.fb.restart()
			
		return self.fit, self.pesos_entrada_oculta, self.pesos_oculta_saida, self.bias_oculta, self.bias_saida

if __name__ == "__main__":
    mlp = MLP(2, 5, 1, 0.01)
    mlp.treinamento()