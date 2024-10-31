import random
from SnakeGame import *
import time

class GeneticAlgorithm:
    population = []
    numberOfGenerations = 70
    populationSize = 100
    mutationRate = 0.001
    bestIndividual = []
    bestFitness = []
    bestScore = []
    score = 0

    game = SnakeGame()
    
    def ini(self):
        pass

    def generateInitialPopulation(self):
        self.population = []
        for _ in range(self.populationSize):
            # Gera um indivíduo (AI da Snake) aleatório
            individual = self.generateRandomIndividual()
            self.population.append(individual)

    def generateRandomIndividual(self):
        # Gera uma sequência aleatória de movimentos (UP, DOWN, LEFT, RIGHT) para a AI da Snake
        moves = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        individual = [random.choice(moves) for _ in range(10)]
        return individual
    
    def fitnessFunction(self, individual):
        # Configura o estado inicial do jogo
        game = SnakeGame()
        game.snake_speed = 0  # Define a velocidade da Snake como 0 para o jogo ser passo a passo
        game.direction = individual[0]  # Define a direção inicial

        # Executa o jogo passo a passo e avalia o valor de aptidão
        fitness = 0
        sair = False
        tempo = 0
        score = game.score
        dist = game.distance_to_food()
        for move in individual[1:]:
            game.change_to = move
            game.run_step()

            if not game.is_heading_towards_food():
                fitness -= 1

            if score != game.score:
                score = game.score
                fitness -= (tempo - dist/10)/2
                tempo = 0
                dist = game.distance_to_food()
            tempo += 1

            if game.snake_position[0] < 0 or game.snake_position[0] > game.window_x - 10 or \
                    game.snake_position[1] < 0 or game.snake_position[1] > game.window_y - 10:
                game.ini
                break  
            for block in game.snake_body[1:]:
                if game.snake_position[0] == block[0] and game.snake_position[1] == block[1]:
                    game.ini
                    sair = True
                    break
            if sair: break

        fitness += len(individual)
        fitness += game.score*100  # Usa o valor do score como valor de aptidão
        self.score = game.score
        game.ini
        return fitness

    def evaluateIndividual(self, individual):
        fitness = self.fitnessFunction(individual)
        return fitness

    def selectParents(self):
        # Seleção por torneio
        tournamentSize = 3
        selectedParents = []
        for _ in range(2):
            tournament = random.sample(self.population, tournamentSize)
            bestIndividual = max(tournament, key=self.evaluateIndividual)
            selectedParents.append(bestIndividual)
        return selectedParents

    def generateChildren(self, parents):
        parent1, parent2 = parents

        # Define os pontos de crossover
        crossoverPoint1 = random.randint(1, min(len(parent1), len(parent2)) - 1)
        crossoverPoint2 = random.randint(crossoverPoint1, min(len(parent1), len(parent2)) - 1)

        child1 = parent1[:crossoverPoint1] + parent2[crossoverPoint1:crossoverPoint2] + parent1[crossoverPoint2:]
        child2 = parent2[:crossoverPoint1] + parent1[crossoverPoint1:crossoverPoint2] + parent2[crossoverPoint2:]

        return child1, child2

    def mutationOperator(self, individual):
        mutatedIndividual = individual.copy()
        for i in range(len(mutatedIndividual)):
            if random.random() < self.mutationRate:
                # Altera aleatoriamente o movimento na posição i
                moves = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                moves.remove(mutatedIndividual[i])  # Remove o movimento atual das possibilidades de movimento
                mutatedIndividual[i] = random.choice(moves)
        return mutatedIndividual
    
    def generateNewPopulation(self, individuos):
        moves = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        for i in range(len(individuos)):
            game = SnakeGame()
            game.snake_speed = 0
            game.direction = individuos[i][0]  

            sair = False
            for move in individuos[i][1:]:
                game.change_to = move
                game.run_step()

                if game.snake_position[0] < 0 or game.snake_position[0] > game.window_x - 10 or \
                        game.snake_position[1] < 0 or game.snake_position[1] > game.window_y - 10:
                    sair = True
                    game.ini
                    break  
                for block in game.snake_body[1:]:
                    if game.snake_position[0] == block[0] and game.snake_position[1] == block[1]:
                        game.ini
                        sair = True
                        break
                if sair: break

            while not sair:
                mov = random.choice(moves) 

                game.change_to = mov
                game.run_step()

                if game.snake_position[0] < 0 or game.snake_position[0] > game.window_x - 10 or \
                        game.snake_position[1] < 0 or game.snake_position[1] > game.window_y - 10:
                    game.ini
                    sair = True
                    break  
                for block in game.snake_body[1:]:
                    if game.snake_position[0] == block[0] and game.snake_position[1] == block[1]:
                        game.ini
                        sair = True
                        break
                if sair: break
                else: individuos[i].append(mov)

            game.ini
        return individuos

    def execute(self):
        self.generateInitialPopulation()
        for generation in range(self.numberOfGenerations):
            newPopulation = self.generateNewPopulation(self.population)
            for _ in range(self.populationSize // 2):
                parents = self.selectParents()
                children = self.generateChildren(parents)
                for child in children:
                    if random.random() < self.mutationRate:
                        child = self.mutationOperator(child)
                    newPopulation.append(child)
            self.population = newPopulation

            bestIndividual = max(self.population, key=self.evaluateIndividual)
            bestFitness = self.evaluateIndividual(bestIndividual)
            self.bestIndividual.append(bestIndividual)
            self.bestFitness.append(bestFitness)
            self.bestScore.append(self.score)

            print('='*50)
            print('Geração: ',generation)
            print('Tamanho: ', len(bestIndividual))
            print('Melhor aptidão: ',bestFitness)
            print('Score: ', self.score)

            if(generation > 100):
                game = SnakeGame()
                game.snake_speed = 0  # Define a velocidade da Snake como 0 para o jogo passo a passo
                game.direction = bestIndividual[0]  # Define a direção inicial
                sair = False
                
                for move in bestIndividual[1:]:
                    time.sleep(0.05)
                    game.change_to = move
                    game.run_stepT()

                    if game.snake_position[0] < 0 or game.snake_position[0] > game.window_x - 10 or \
                        game.snake_position[1] < 0 or game.snake_position[1] > game.window_y - 10:
                        print('Score:', game.score)
                        break  # Interrompe a avaliação se o jogo acabar

                    for block in game.snake_body[1:]:
                        if game.snake_position[0] == block[0] and game.snake_position[1] == block[1]:
                            print('Score:', game.score)
                            sair = True
                            break
                    if sair:
                        break

                if not sair:
                    print('Score:', game.score)
                
                game.ini
        return self.bestIndividual, self.bestFitness, self.bestScore