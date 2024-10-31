import random

def perceptron(treinamento, n = 0.01, W = [], max_epocas = 100, seed = 42):
    if(W == []):
        random.seed(seed)
        for i in range(3):
            W.append(random.uniform(-0.5, 0.5))
    
    controle = True
    n_epocas = 0
    
    # adicionando bias
    N = len(treinamento)
    for i in range(N):
        treinamento[i].insert(0, 1)
        
    classId = len(treinamento[0])  
    classe  = [row[-1] for row in treinamento]   
     
    while(controle and n_epocas < max_epocas):
        c = True
        
        print('\n' + '-'*30 + '\nEpoca ' + str(n_epocas)+ ':' )
        n_epocas += 1
        
        for i in range(N):
            exemplo = treinamento[i][0:classId-1]
            print('\nExemplo: ' + str(i+ 1))

            v = 0
            for j in range(len(exemplo)):
                v += exemplo[j]*W[j]

            f = 0
            if v > 0:
                f = 1
            if f != classe[i]:
                print('ERRO')
                c = False
                k = W
                for m in range(len(W)):
                    k[m] = W[m] + (n*(classe[m] - f)*exemplo[m])
                W = k
            else: print('Certo')
            
            print('W = ', end = '')
            for m in range(len(W)):
                print("{:.4f}".format(W[m]), end ='')
                if m != 2:
                    print(', ', end = '')    
        if c == True:
            controle = False 
    return W           

   
def perceptron_predict(W, teste):
    v = 0
    teste.insert(0, 1)
    for j in range(len(teste)):
        v += teste[j]*W[j]
    f = 0
    if v > 0:
        f = 1
    return f



entrada = [[0, 0, 0], [0, 1, 1], [1 , 0, 1], [1, 1, 1]]
W = perceptron(entrada)
res = perceptron_predict(W, [3,3])
print('\n'+'-'*30+'\npredição')
print(res)