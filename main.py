# -------------------------------------------------------------------------------------------------
# # importing all required objects
# -------------------------------------------------------------------------------------------------
from SnakeGame import *
from GeneticAlgorithm import *
import matplotlib.pyplot as plt
import time
import os
# -------------------------------------------------------------------------------------------------
# # main function 
# -------------------------------------------------------------------------------------------------

def save_vectors_to_file(v1, v2, v3, filename = 'resultado'):
    with open(filename, 'w') as file:
        file.write('bestIndividual:/n')
        file.write(','.join(str(x) for x in v1))
        file.write('/n/n')

        file.write('BestFitness:/n')
        file.write(','.join(str(x) for x in v2))
        file.write('/n/n')

        file.write('BestScore:/n')
        file.write(','.join(str(x) for x in v3))
        file.write('/n/n')

    print('Vectors saved to', filename)

def plot(bestIndividual, bestFitness, bestScore):
    generations = range(len(bestIndividual))
    sizes = [len(individual) for individual in bestIndividual]
    fitness = bestFitness
    scores = bestScore

    # Obter diretório atual da execução
    current_dir = os.getcwd()

    # Criar pasta "plots" se ela não existir
    plots_dir = os.path.join(current_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Personalização dos gráficos
    plt.style.use('seaborn-darkgrid')

    # Criar figura e eixos
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotar o tamanho do indivíduo
    ax1.plot(generations, sizes, label='Tamanho do Indivíduo', color='steelblue')
    ax1.set_ylabel('Tamanho', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # Criar segundo eixo y para o fitness
    ax2 = ax1.twinx()
    ax2.plot(generations, fitness, label='Fitness', color='darkorange')
    ax2.set_ylabel('Fitness', fontsize=12, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    # Criar terceiro eixo y para os scores
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(generations, scores, label='Scores', color='forestgreen')
    ax3.set_ylabel('Score', fontsize=12, color='forestgreen')
    ax3.tick_params(axis='y', labelcolor='forestgreen')

    # Ajustar rótulos e título
    ax1.set_xlabel('Gerações', fontsize=12)
    ax1.set_title('Evolução dos Indivíduos, Fitness e Scores', fontsize=14, fontweight='bold')

    # Ajustar cores da legenda
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()

    # Ajustar posição da legenda
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='best')

    # Ajustar espaçamento entre os subplots
    fig.tight_layout()

    # Ajustar espaçamento das linhas do grid
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax3.grid(True, linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax3.set_axisbelow(True)
    # Salvar o gráfico em um arquivo
    plt.savefig('plots/evolucao.svg', format='svg', dpi=1200)



if __name__ == "__main__":
    ga = GeneticAlgorithm()
    bestIndividual, bestFitness, bestScore = ga.execute()
    plot(bestIndividual, bestFitness, bestScore)
    save_vectors_to_file(bestIndividual, bestFitness, bestScore)


    game = SnakeGame()
    game.direction = bestIndividual[len(bestIndividual)-1][0]
    for move in bestIndividual[len(bestIndividual)-1][1:]:
        time.sleep(0.1)
        game.change_to = move
        game.run_stepT()
        if game.snake_position[0] < 0 or game.snake_position[0] > game.window_x - 10 or \
                game.snake_position[1] < 0 or game.snake_position[1] > game.window_y - 10:
            break  
        
        for block in game.snake_body[1:]:
            if game.snake_position[0] == block[0] and game.snake_position[1] == block[1]:
                break
    
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
