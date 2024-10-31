import numpy as np

N = 100 # qtde de numeros que queremos separar
POP_SIZE = 100 # tamanho da população
GERACOES = 500 # gerações

def fitness_function(problem, individual):
    # Encontra os índices onde o valor do indivíduo é igual a 0
    ids0 = np.where(individual == 0)[0]
    
    # Encontra os índices onde o valor do indivíduo é igual a 1
    ids1 = np.where(individual == 1)[0]
    
    # Calcula a soma dos valores do problema correspondentes aos índices onde o indivíduo é igual a 0
    sum_ids0 = np.sum(problem[ids0])
    
    # Calcula a soma dos valores do problema correspondentes aos índices onde o indivíduo é igual a 1
    sum_ids1 = np.sum(problem[ids1])
    
    # Calcula a aptidão (fitness) como a diferença absoluta entre as somas das duas listas
    fit = abs(sum_ids0 - sum_ids1)
    
    # Retorna a aptidão (fitness)
    return fit

def tournament_selection(population, fitness_values, k=3):
    # Seleciona aleatoriamente k indivíduos da população como desafiantes (challengers)
    challengers = np.random.choice(np.arange(len(population)), size=k, replace=False)
    
    # Encontra o índice do desafiante com a menor aptidão (fitness)
    id_min = np.argmin(fitness_values[challengers])
    
    # Obtém o melhor indivíduo (com menor aptidão) dos desafiantes
    best = population[challengers[id_min]]
    
    # Retorna o melhor indivíduo
    return best

def one_point_crossover(parentA, parentB):
    # Gera um ponto de corte aleatório para o crossover
    cutoff = np.random.randint(2, len(parentA) - 1)
    
    # Realiza o crossover entre os pais no ponto de corte
    child1 = np.concatenate((parentA[:cutoff], parentB[cutoff:]))
    child2 = np.concatenate((parentB[:cutoff], parentA[cutoff:]))

    # Cria um dicionário com os dois filhos gerados
    children = {"child1": child1, "child2": child2}
    # Retorna os filhos gerados
    return children

def uniform_mutation(children, mut_prob=0.001):
    # Itera sobre os elementos do filho 1
    for i in range(len(children["child1"])):
        # Gera uma probabilidade aleatória entre 1 e 100
        prob = np.random.randint(1, 101) 
        # Verifica se a probabilidade está abaixo do limite de mutação
        if prob <= mut_prob * 100:
            # Realiza a mutação invertendo o valor do elemento
            children["child1"][i] = 1 - children["child1"][i]
    
    # Itera sobre os elementos do filho 2
    for i in range(len(children["child2"])):
        # Gera uma probabilidade aleatória entre 1 e 100
        prob = np.random.randint(1, 101)

        # Verifica se a probabilidade está abaixo do limite de mutação
        if prob <= mut_prob * 100:
            # Realiza a mutação invertendo o valor do elemento
            children["child2"][i] = 1 - children["child2"][i]

    # Retorna os filhos com as mutações aplicadas
    return children


def genetic_algorithm(problem, pop_size=POP_SIZE, generations=GERACOES):
    # Listas para armazenar os resultados
    bests = []                     # Melhores soluções até o momento (por geração)
    average_fitness = []            # Fitness médio (por geração)
    all_populations = []            # Todas as populações geradas pelo algoritmo genético
    all_fitness = []                # Todos os valores de fitness avaliados
    
    # Inicialização da população inicial e valores de fitness
    population = np.random.randint(2, size=(pop_size, N))
    fitness_values = np.zeros(pop_size)
    
    # Avaliação do fitness para cada indivíduo na população inicial
    for i in range(pop_size):
        fitness_values[i] = fitness_function(problem=problem, individual=population[i])
    
    # Adiciona os valores iniciais às listas de resultados
    all_populations.append(population)
    all_fitness.append(fitness_values)
    bests.append(np.min(fitness_values))
    average_fitness.append(np.mean(fitness_values))
    
    # Loop principal para as gerações do algoritmo genético
    for k in range(generations):
        print(" - Geração:", k)
        
        # Cria uma nova população vazia
        new_population = np.zeros((pop_size, N))
        
        # Loop para gerar novos indivíduos da nova população
        for j in range(pop_size // 2):
            # Seleciona pais através do torneio
            parentA = tournament_selection(population, fitness_values, k=3)
            parentB = tournament_selection(population, fitness_values, k=3)
            
            # Realiza o crossover de um ponto para gerar dois filhos
            children = one_point_crossover(parentA, parentB)
            
            # Realiza a mutação uniforme nos filhos
            xmen = uniform_mutation(children, mut_prob=0.01)
            
            # Insere os filhos na nova população
            idx1 = (2 * j) - 1
            idx2 = 2 * j
            new_population[idx1] = xmen["child1"]
            new_population[idx2] = xmen["child2"]
        
        # Atualiza a população com a nova população gerada
        population = new_population
        
        # Avalia o fitness para cada indivíduo na nova população
        for i in range(pop_size):
            fitness_values[i] = fitness_function(problem=problem, individual=population[i])
        
        # Adiciona os valores da nova geração às listas de resultados
        all_populations.append(population)
        all_fitness.append(fitness_values)
        bests.append(np.min(fitness_values))
        average_fitness.append(np.mean(fitness_values))
        
        # Imprime as informações sobre a melhor solução encontrada
        print("best fitness:", np.min(fitness_values), ",   avg fitness:", np.mean(fitness_values))
        if np.min(fitness_values) < 2:
            # Se a melhor solução atender a um critério (no caso, menor que 2), imprime informações adicionais
            indi = population[np.argmin(fitness_values)]
            ids0 = np.where(indi == 0)[0]
            ids1 = np.where(indi == 1)[0]
            print('\n'+'-'*50 + '\nProblema original:')
            print(problem, end='\n\n')
            print('\n'+'-'*50 +'\nLista 0:')
            print(problem[ids0])
            print('Soma Lista 0:', sum(problem[ids0]))
            print('\n'+'-'*50 +'\nLista 1:')
            print(problem[ids1])
            print('Soma Lista 1:', sum(problem[ids1]))
            break

    # Retorna um dicionário contendo todos os resultados
    return {"bests": bests, "average_fitness": average_fitness, "all_populations": all_populations, "all_fitness": all_fitness}
problema = np.random.choice(np.arange(-1000, 1001), size=N)
teste = genetic_algorithm(problem = problema, pop_size = POP_SIZE, 
	generations = GERACOES)