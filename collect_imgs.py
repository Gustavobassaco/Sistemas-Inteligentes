import os
import cv2

# Diretório para salvar os dados coletados
DATA_DIR = './data'

# Verifica se o diretório não existe e o cria
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Número de classes (categorias) para coletar dados
number_of_classes = 10

# Tamanho do conjunto de dados para cada classe
dataset_size = 1000

# Inicialização da captura de vídeo
cap = cv2.VideoCapture(0)

# Loop para cada classe
for j in range(number_of_classes):
    # Verifica se o diretório da classe não existe e o cria
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Coletando imagens da classe {}'.format(j))

    done = False
    while True:
        # Captura do quadro atual da câmera
        ret, frame = cap.read()

        # Inverte horizontalmente o quadro (espelha a exibição)
        frame = cv2.flip(frame, 1)

        # Exibe uma mensagem no quadro
        cv2.putText(frame, 'Preparado? Aperte "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)

        # Exibe o quadro
        cv2.imshow('frame', frame)

        # Verifica se a tecla 'Q' foi pressionada para iniciar a coleta de dados
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        # Captura do quadro atual da câmera
        ret, frame = cap.read()

        # Inverte horizontalmente o quadro (espelha a exibição)
        frame = cv2.flip(frame, 1)

        # Exibe o quadro
        cv2.imshow('frame', frame)

        # Aguarda 25 milissegundos
        cv2.waitKey(25)

        # Salva o quadro como uma imagem com o nome correspondente ao contador
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        # Incrementa o contador
        counter += 1

# Libera a captura de vídeo
cap.release()

# Fecha todas as janelas abertas
cv2.destroyAllWindows()
