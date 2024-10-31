import os
import pickle
import mediapipe as mp
import cv2

# Configuração do mediapipe para detecção e rastreamento de mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicialização do objeto para detecção de mãos
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Diretório dos dados
DATA_DIR = './data'

# Listas para armazenar os dados e rótulos
data = []
labels = []

# Percorrendo os diretórios dos dados
for dir_ in os.listdir(DATA_DIR):
    # Percorrendo os arquivos de imagem dentro de cada diretório
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        # Listas para armazenar as coordenadas x e y dos pontos da mão
        x_ = []
        y_ = []

        # Leitura e conversão da imagem em RGB
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Processamento da imagem para detecção de mãos
        results = hands.process(img_rgb)

        # Verifica se foram detectadas mãos na imagem
        if results.multi_hand_landmarks:
            cont = 0
            # Iteração sobre as mãos detectadas
            for hand_landmarks in results.multi_hand_landmarks:
                cont += 1
                # Verifica se é a segunda mão detectada (cont == 2)
                if cont == 2:
                    print('AQUI')
                    break
                # Iteração sobre os pontos da mão
                for i in range(len(hand_landmarks.landmark)):
                    # Normalização das coordenadas em relação ao ponto de referência (base)
                    if i == 0:
                        x_base = hand_landmarks.landmark[i].x
                        y_base = hand_landmarks.landmark[i].y

                    # calcula o valor de cada ponto em relação ao ponto base
                    x = hand_landmarks.landmark[i].x - x_base
                    y = hand_landmarks.landmark[i].y - y_base

                    # salva os valores relativos ao ponto base em uma lista
                    data_aux.append(x)
                    data_aux.append(y)

                # encontra o valor da coordenada mais distante do ponto base
                max_value = max(list(map(abs, data_aux)))

                # função de normalização
                def normalize_(n):
                    return n / max_value
                
                # normaliza a lista de pontos relativos
                data_aux = list(map(normalize_, data_aux))

            # Adiciona os dados e rótulos às listas correspondentes
            data.append(data_aux)
            labels.append(dir_)

# Salvando os dados e rótulos em um arquivo pickle
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()