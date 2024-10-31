
def calcV(w, e):
    v = 0
    print('V = ', end='')
    for i in range(len(w)):
        v += w[i]*e[i]
        print("{:.4f}".format(w[i]) + ' * ' + "{:.4f}".format(e[i]), end = '' )
        if i != 2:
            print(' + ', end='')
    print()
    return v

def calcF(v):
    f = 0
    if v > 0:
        f = 1
    return f

def verifica(f, d): 
    if f == d:
        return True
    else: return False

def update(w, d, f, n, e):
    k = w
    for i in range(len(w)):
        k[i] = w[i] + (n*(d - f)*e[i])
    return k

W = [-0.5441, 0.5562, 0.4074]
e = [[-1, 2, 2], [-1, 4, 4]]
d = [1, 0]
n = 0.1

controle = True
c = True
epoca = 0
while(controle):
    c = True
    
    print('\n' + '-'*30 + '\nEpoca ' + str(epoca)+ ':' )
    epoca += 1
    for i in range(2):
        print('\nExemplo: ' + str(i + 1))
        v = calcV(W, e[i])
        print('V = '+ "{:.4f}".format(v))
        f = calcF(v)
        print('F = ', str(f))
        resultado = verifica(f, d[i])
        if resultado == False:
            print('ERRO')
            c = False
            W = update(W, d[i], f, n, e[i])
        else: print('Certo')
        print('W = ', end = '')
        for i in range(len(W)):
            print("{:.4f}".format(W[i]), end ='')
            if i != 2:
                print(', ', end = '')
    if c == True:
        controle = False
        
