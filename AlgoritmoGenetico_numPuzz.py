import numpy as np
import copy
import random

N = 9 # qtde de numeros que queremos separar
POP_SIZE = 50 # tamanho da população
GERACOES = 500 # gerações

class Estado:
    def __init__(self, vetor = [], pai1 = [], pai2 = [], profundidade = 0, fit = 0):
        self.vetor = vetor
        self.pai1 = pai1
        self.pai2 = pai2
        self.profundidade = profundidade
        self.fit = fit

def vetor_para_matriz(vetor):
    matriz = []
    for i in range(0, len(vetor), 3):
        linha = vetor[i:i+3]
        matriz.append(linha)
    return matriz

def matriz_para_vetor(matriz):
    vetor = []
    for linha in matriz:
        vetor.extend(linha)
    return vetor

def imprime_matriz(matriz):
    for elem in matriz:
        print(elem, end=' ')

def selecao(populacao, fitness):
    selecionados = random.sample(populacao, k=2)
    return selecionados

def crossover(est, final):
    estado = est.vetor
    estados_possiveis = []
    index_vazio = estado.index(0)  # Encontrar o índice do elemento vazio
    
    # Movimento para cima
    if index_vazio - 3 >= 0:
        novo_estado = estado.copy()
        novo_estado[index_vazio], novo_estado[index_vazio - 3] = novo_estado[index_vazio - 3], novo_estado[index_vazio]
        estados_possiveis.append(novo_estado)
    
    # Movimento para baixo
    if index_vazio + 3 < len(estado):
        novo_estado = estado.copy()
        novo_estado[index_vazio], novo_estado[index_vazio + 3] = novo_estado[index_vazio + 3], novo_estado[index_vazio]
        estados_possiveis.append(novo_estado)
    
    # Movimento para a esquerda
    if index_vazio % 3 != 0:
        novo_estado = estado.copy()
        novo_estado[index_vazio], novo_estado[index_vazio - 1] = novo_estado[index_vazio - 1], novo_estado[index_vazio]
        estados_possiveis.append(novo_estado)
    
    # Movimento para a direita
    if index_vazio % 3 != 2:
        novo_estado = estado.copy()
        novo_estado[index_vazio], novo_estado[index_vazio + 1] = novo_estado[index_vazio + 1], novo_estado[index_vazio]
        estados_possiveis.append(novo_estado)

    fit = []
    possiveis = []
    for i in range(len(estados_possiveis)):
        possivel = Estado(estados_possiveis[i])
        possiveis.append(possivel)
        fit.append(fitness(possivel, final, 0))

    # Obter os 2 maiores valores de fitness
    maiores_fit = sorted(fit, reverse=True)[:2]

    # Filtrar os estados correspondentes aos 2 maiores valores de fitness
    estados_melhores_fit = [estados_possiveis[i] for i, f in enumerate(fit) if f in maiores_fit]
    print(len(estados_melhores_fit))
    return estados_melhores_fit[:2]


'''
def crossover(pai1, pai2):
    ponto_corte1 = random.randint(0, 7)
    ponto_corte2 = random.randint(ponto_corte1 + 1, 8)
    
    filho1 = [-1] * 9
    filho2 = [-1] * 9
    
    # Copiar segmento entre os pontos de corte
    filho1[ponto_corte1:ponto_corte2+1] = pai1.vetor[ponto_corte1:ponto_corte2+1]
    filho2[ponto_corte1:ponto_corte2+1] = pai2.vetor[ponto_corte1:ponto_corte2+1]
    
    # Preencher restante dos elementos em filho1
    index1 = (ponto_corte2 + 1) % 9
    index2 = (ponto_corte2 + 1) % 9
    
    for _ in range(8):
        if filho1[index2] == -1:  # Verificar se o elemento já foi preenchido
            while pai2.vetor[index1] in filho1:
                index1 = (index1 + 1) % 9
            filho1[index2] = pai2.vetor[index1]
        
        index1 = (index1 + 1) % 9
        index2 = (index2 + 1) % 9
    
    # Preencher restante dos elementos em filho2
    index1 = (ponto_corte2 + 1) % 9
    index2 = (ponto_corte2 + 1) % 9
    
    for _ in range(8):
        if filho2[index2] == -1:  # Verificar se o elemento já foi preenchido
            while pai1.vetor[index1] in filho2:
                index1 = (index1 + 1) % 9
            filho2[index2] = pai1.vetor[index1]
        
        index1 = (index1 + 1) % 9
        index2 = (index2 + 1) % 9
    
    return filho1, filho2
'''

def mutacao(filho, taxa_mutacao):
    for i in range(9):
        if random.random() < taxa_mutacao:
            j = random.randint(0, 8)
            filho[i], filho[j] = filho[j], filho[i]
    return filho

def fitness(est, objetivo, profundidade):
    estado = vetor_para_matriz(est.vetor)
    objetivo = vetor_para_matriz(objetivo)
    # Calcula a distância de Manhattan para cada peça
    distancia = 0
    for i in range(3):
        for j in range(3):
            if estado[i][j] != 0:
                objetivo_i, objetivo_j = 0, 0
                # Encontra a posição correta da peça na matriz objetivo
                for m in range(3):
                    for n in range(3):
                        if objetivo[m][n] == estado[i][j]:
                            objetivo_i, objetivo_j = m, n
                            break
                distancia -= abs(i - objetivo_i) + abs(j - objetivo_j)
    distancia -= profundidade
    return distancia

def generateFilho(estado, pai1, pai2, objetivo):
    profundidade = 0
    fit = fitness(estado, objetivo, 0)
    return Estado(estado, pai1, pai2, profundidade, fit)

def algoritmoGenetico(matriz, final, pop_size = POP_SIZE, geracoes = GERACOES):
    cont = 0
    final = matriz_para_vetor(final)

    population = []
    fitness_values = np.zeros(pop_size)

    for i in range(pop_size):
        vetor = random.sample(range(9), N)
        pai1 = Estado()
        pai2 = Estado()
        individuo = Estado(vetor, pai1, pai2, final, 0)
        fitness_values[i] = fitness(individuo, final, 0)
        population.append(individuo)

    for k in range(GERACOES):
        print(" - Geração:", k)
        new_population = []
        for m in range(pop_size):
            new_population.append(Estado())

        for j in range(pop_size//2):
            pai1, pai2 = selecao(population, fitness_values)
            filho1, filho2 = crossover(pai1, final)
            xfilho1 = mutacao(filho1, 0.1)
            xfilho2 = mutacao(filho2, 0.1)

            xind1 = Estado(xfilho1, pai1, pai2, final, 0)
            xind2 = Estado(xfilho2, pai1, pai2, final, 0)
            idx1 = (2 * j) - 1
            idx2 = 2 * j

            new_population[idx1] = xind1
            new_population[idx2] = xind2

        population = new_population
            
        for i in range(pop_size):
            fitness_values[i] = fitness(population[i], final, 0)
        if np.min(fitness_values) == 0:
            print(population[np.argmin(fitness_values)])
    print(population[1].vetor)


#profundidade 17 heuristica normal : 5240 iterações, heuristica manhattan: 554 iterações
grafo = [
    [3, 1, 5],
    [2, 4, 6],
    [8, 0, 7]
]

objetivo = [
    [1, 2, 3],
    [8, 0, 4],
    [7, 6, 5]
]

resultado, retorno = algoritmoGenetico(grafo, objetivo)

if resultado != -999:
    print('\n\n' + '-'*15 + ' INICIO ' + '-'*15)
    print(grafo)
    print('-'*14 + ' OBJETIVO ' + '-'*14)
    print(objetivo)
    print('-'*38, end = '\n\n')

    print('Busca Gulosa teve '+ str(resultado) + ' iterações, com profundidade ' + str(retorno[len(retorno)-1].profundidade))
    for i in range(len(retorno) - 1):
        print(f"{'   ' + retorno[i].rot + ' -> '[:15] :<15}", end = '')
        print(retorno[i].matriz, end = '')
        if (i + 1) % 3 == 0:
            print()
    print(f"{'   ' + retorno[len(retorno) -1].rot + ' -> '[:15] :<15}", end='')
    print(retorno[len(retorno) -1].matriz)
else:
    print('Deu ruim GURI')

