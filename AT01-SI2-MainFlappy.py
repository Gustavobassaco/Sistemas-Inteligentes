from GeneticAlgorithm import GeneticAlgorithm
from MLP import MLP
import matplotlib.pyplot as plt

# mlp
# é necessário definir no arquivo Demo a função get_distances, quantas entradas a rede possui
nEntrada = 2
nOculta = 3
nSaida = 1
taxaDeAprendizado = 0.1

# GA
nGen = 500
size = 30
mutation = 0.05
elitismo = 2

# Definir cores e legendas antes do loop
colors = ['red', 'green', 'blue']  # Lista de cores para cada conjunto de pontos
labels = ['Configuração 1', 'Configuração 2', 'Configuração 3']  # Lista de legendas para cada conjunto de pontos

bestTamG = []
bestFitGen = []

for i in range(1):
    bestFitnessList = []
    GenList = []
    for j in range(1):
        nOculta = i+10
        ga = GeneticAlgorithm(entrada=nEntrada, oculta=nOculta, 
                            saida=nSaida, taxaDeAprendizado=taxaDeAprendizado,
                            numberOfGenerations=nGen, populationSize = size,
                            mutationRate=mutation, elitism=elitismo)
        bestFitness, bestIndividual, nGen = ga.execut()
        bestFitnessList.append(bestFitness)
        GenList.append(nGen)

    meanFit = sum(bestFitnessList)/len(bestFitnessList)
    bestFitGen.append(meanFit)
    meanGen = sum(GenList) / len(GenList) 
    bestTamG.append(meanGen)

mlp = MLP(entrada=nEntrada, oculta=nOculta, saida=nSaida,
                        taxaDeAprendizado=taxaDeAprendizado, 
                        pesos_entrada_oculta=bestIndividual.pesos_entrada_oculta,
                        pesos_oculta_saida=bestIndividual.pesos_oculta_saida,
                        bias_oculta=bestIndividual.bias_oculta,
                        bias_saida=bestIndividual.bias_saida)

mlp.jogo()

# Plot do melhor fitness ao longo das gerações
for i, (x, y, color, label) in enumerate(zip(bestTamG, bestFitGen, colors, labels)):
    plt.scatter(x, y, color=color, alpha=0.5, label=label)
    plt.annotate(f'Config {i+1}', (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

plt.title('Número Médio de Gerações para Diferentes Quantidades de Neurônios')
plt.xlabel('Número Médio de Gerações')
plt.ylabel('Fitness Médio')
plt.show()