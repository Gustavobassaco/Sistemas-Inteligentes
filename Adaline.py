import random

import numpy as np
import matplotlib.pyplot as plt




def adaline(treinamento, n = 0.01, W = [], max_epocas = 1000, seed = 42, tolError = 0.01):
    if(W == []):
        random.seed(seed)
        for i in range(3):
            W.append(random.uniform(-0.5, 0.5))
    
    erro = 100
    n_epocas = 0
    
    # adicionando bias
    N = len(treinamento)
    for i in range(N):
        treinamento[i].insert(0, 1)
        
    classId = len(treinamento[0])  
    classe  = [row[-1] for row in treinamento]   
    
    while(erro > tolError and n_epocas < max_epocas):
        
        print('\n' + '-'*30 + '\nEpoca ' + str(n_epocas)+ ':' )
        n_epocas += 1
        
        for i in range(N):
            exemplo = treinamento[i][0:classId-1]
            print('\nExemplo: ' + str(i+ 1))
            
            v = 0
            for j in range(len(exemplo)):
                v += exemplo[j]*W[j]
            

            k = W
            for m in range(len(W)):
                k[m] = W[m] + (n*(classe[m] - v)*exemplo[m])
            W = k

            
            print('W = ', end = '')
            for m in range(len(W)):
                print("{:.4f}".format(W[m]), end ='')
                if m != 2:
                    print(', ', end = '')    

        erro = 0
        for j in range(len(exemplo)):
            erro += (classe[j] - (exemplo[j]*W[j]))**2
        erro = erro/N
        print("\nErro: "+ str(erro))
    return W           

   
def adaline_predict(W, teste):
    teste.insert(0, 1)
    v = 0
    for j in range(len(teste)):
        v += teste[j] * W[j]
    print(v)
    if v >= 0:
        return 1
    else:
        return 0



entrada = [[0, 0, 0], [0, 1, 0], [1 , 0, 0], [1, 1, 1]]
W = adaline(entrada)
print('\n\n')
res = adaline_predict(W, [2,2])
print('\n'+'-'*30+'\npredição')
print(res)

slope = -(W[0]/W[2])/(W[0]/W[1])
intercept = -W[0]/W[2]

x = np.linspace(-10, 10, 100)  # Você pode ajustar os limites e a quantidade de pontos conforme necessário

# Calcule os valores correspondentes de y usando a equação da reta (y = mx + b)
y = slope * x + intercept
entrada = [[0, 0], [0, 1], [1, 0], [1, 1]]
entrada_x = [p[0] for p in entrada]
entrada_y = [p[1] for p in entrada]

# Crie o gráfico
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=f'y = {slope:.2f}x + {intercept:.2f}')
plt.scatter(entrada_x, entrada_y, color='red', marker='o', label='Pontos de Entrada')  # Plote os pontos de entrada
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.title('Gráfico com Intercept, Slope e Pontos de Entrada')
plt.grid(True)

# Defina os limites dos eixos x e y
plt.xlim([-1, 2])
plt.ylim([-1, 2])

plt.legend()
plt.show()