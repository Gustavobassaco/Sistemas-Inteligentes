# -------------------------------------------------------------------------------------------------
# import required packages/libraries
# -------------------------------------------------------------------------------------------------
from Demo import FlappyBird_Human
from MLP import MLP
import numpy as np

# -------------------------------------------------------------------------------------------------
# A class for a Individual
# -------------------------------------------------------------------------------------------------

class Individuo:
    def __init__(self, pesos_entrada_oculta, pesos_oculta_saida, bias_oculta, bias_saida):
        self.pesos_entrada_oculta = pesos_entrada_oculta
        self.pesos_oculta_saida = pesos_oculta_saida
        self.bias_oculta = bias_oculta
        self.bias_saida = bias_saida

# -------------------------------------------------------------------------------------------------
# A class for a Genetic Algorithm 
# -------------------------------------------------------------------------------------------------

class GeneticAlgorithm:
    
    # constructor
    def __init__(self, entrada = 5, oculta= 10, saida = 1, taxaDeAprendizado = 0.1, numberOfGenerations=100, populationSize=30, 
                 mutationRate=0.01, elitism=6):

        self.taxaDeAprendizado = taxaDeAprendizado
        self.tam_entrada = entrada
        self.tam_oculta = oculta
        self.tam_saida = saida
        self.fb = FlappyBird_Human()

        self.population = []
        self.numberOfGenerations = numberOfGenerations
        self.stopEarlyCriteria = 10
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.bestIndividual = []
        self.bestFitness = []
        self.elitism = elitism


        
    
    # generate the initial population
    def generateInitialPopulation(self):
        self.individuos = []
        for _ in range(self.populationSize):
            pesos_entrada_oculta = np.random.uniform(size=(self.tam_entrada, self.tam_oculta))
            pesos_oculta_saida = np.random.uniform(size=(self.tam_oculta, self.tam_saida))
            bias_oculta = np.zeros((1, self.tam_oculta))
            bias_saida = np.zeros((1, self.tam_saida))

            individuo = Individuo(pesos_entrada_oculta, pesos_oculta_saida, bias_oculta, bias_saida)
            self.individuos.append(individuo)

    def selectParents(self, fitness):
        # Seleção de pais proporcional ao desempenho (Roleta)
        total_fitness = np.sum(fitness)
        probabilities = fitness / total_fitness

        # Escolha aleatória com base nas probabilidades
        selected_parents = np.random.choice(self.individuos, size=2, p=probabilities.flatten())

        return selected_parents
    
    # given two parents, generate two children recombining them
    def generateChildren(self, parents):
        parent1, parent2 = parents

        # Crossover (recombinação) em um ponto para cada matriz
        crossover_point1 = np.random.randint(self.tam_entrada)
        crossover_point2 = np.random.randint(self.tam_oculta)
        crossover_point3 = np.random.randint(self.tam_oculta)
        crossover_point4 = np.random.randint(self.tam_saida)
        child1 = Individuo(
            np.concatenate([parent1.pesos_entrada_oculta[:crossover_point1], parent2.pesos_entrada_oculta[crossover_point1:]]),
            np.concatenate([parent1.pesos_oculta_saida[:crossover_point2], parent2.pesos_oculta_saida[crossover_point2:]]),
            np.concatenate([parent1.bias_oculta[:crossover_point3], parent2.bias_oculta[crossover_point3:]]),
            np.concatenate([parent1.bias_saida[:crossover_point4], parent2.bias_saida[crossover_point4:]])
        )

        child2 = Individuo(
            np.concatenate([parent2.pesos_entrada_oculta[:crossover_point1], parent1.pesos_entrada_oculta[crossover_point1:]]),
            np.concatenate([parent2.pesos_oculta_saida[:crossover_point2], parent1.pesos_oculta_saida[crossover_point2:]]),
            np.concatenate([parent2.bias_oculta[:crossover_point3], parent1.bias_oculta[crossover_point3:]]),
            np.concatenate([parent2.bias_saida[:crossover_point4], parent1.bias_saida[crossover_point4:]])
        )
        return child1, child2 
    

    # selects an individual and apply a mutation
    def mutationOperator(self, individual):
        for i in range(len(individual.pesos_entrada_oculta.flatten())):
            if np.random.rand() < self.mutationRate:
                individual.pesos_entrada_oculta.flat[i] += np.random.uniform(-0.5, 0.5)

        for i in range(len(individual.pesos_oculta_saida.flatten())):
            if np.random.rand() < self.mutationRate:
                individual.pesos_oculta_saida.flat[i] += np.random.uniform(-0.5, 0.5)

        for i in range(len(individual.bias_oculta.flatten())):
            if np.random.rand() < self.mutationRate:
                individual.bias_oculta.flat[i] += np.random.uniform(-0.5, 0.5)

        for i in range(len(individual.bias_saida.flatten())):
            if np.random.rand() < self.mutationRate:
                individual.bias_saida.flat[i] += np.random.uniform(-0.5, 0.5)

        return individual
    

    # run GA
    def execut(self):
        self.generateInitialPopulation()
        # cada geração
        gen = 0
        genaux = 0
        while genaux < self.numberOfGenerations:
        #for i in range(self.numberOfGenerations):
            gen += 1
            genaux += 1
            #print('-'*30)
            # cria uma mlp com os pesos
            fitness = []
            for j in range(self.populationSize):
                print('Geracao ' +str(genaux) + '  mlp ' + str(j) + '  ', end='')
                mlp = MLP(entrada = self.tam_entrada, oculta = self.tam_oculta, saida = self.tam_saida,
                        taxaDeAprendizado=self.taxaDeAprendizado, fb=self.fb,
                        pesos_entrada_oculta= self.individuos[j].pesos_entrada_oculta,
                        pesos_oculta_saida=self.individuos[j].pesos_oculta_saida,
                        bias_oculta=self.individuos[j].bias_oculta,
                        bias_saida=self.individuos[j].bias_saida)
                retorno = mlp.treinamento()
                fitness.append(retorno[0])
                self.individuos[j].pesos_entrada_oculta = retorno[1]
                self.individuos[j].pesos_oculta_saida = retorno[2]
                self.individuos[j].bias_oculta = retorno[3]
                self.individuos[j].bias_saida = retorno[4]
            
            # Adiciona o melhor fitness à lista
            maximo = max(fitness)
            self.bestFitness.append(maximo)


            # Seleciona os melhores indivíduos com elitismo
            best_indices = np.argsort(fitness)[-self.elitism:]
            best_indice = np.argsort(fitness)[-1]

            best_individuals = [self.individuos[idx] for idx in best_indices]
            self.bestIndividual.append(self.individuos[best_indice])

            # Criterio de parada, atingido quando um individo se aproximo do ideial
            if maximo > 200:
                print('AQQUUII ' + str(maximo))
                genaux = 100000

            # Seleção, crossover e mutação para criar nova população
            new_population = []
            new_population.extend(best_individuals)

            for _ in range((self.populationSize - len(best_individuals)) // 2):
                parents = self.selectParents(fitness)
                children = self.generateChildren(parents)
                children = [self.mutationOperator(child) for child in children]
                new_population.extend(children)

            # Atualiza a população
            self.individuos = np.array(new_population)

        best_indice = np.argsort(self.bestFitness)[-1]
        return self.bestFitness[best_indice], self.bestIndividual[best_indice], gen

            
            
if __name__ == "__main__":
    ga = GeneticAlgorithm(2, 5, 1, 0.1)
    ga.execute()