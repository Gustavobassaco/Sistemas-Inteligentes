import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Obter o tamanho da tela
width, height = pyautogui.size()

# Carregar o modelo treinado
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

# Configurações do Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicializar o objeto de detecção de mãos
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

# Dicionários para mapear os rótulos das classes
labels_dict = {0: 'Cima', 1: 'Mouse', 2: 'Esquerdo', 3: 'Direito', 4: 'Copiar', 5: 'Colar', 6: 'Fechado', 7: 'Pinca', 8: 'Aberto', 9: 'Baixo'}
state = {0: 'Cima', 1: 'Mouse', 2: 'Esquerdo', 3: 'Direito', 4: 'Copiar', 5: 'Colar', 6: 'Fechado', 7: 'Pinca', 8: 'Aberto', 9: 'Baixo'}

position_2x = position_3y = -1

while True:

    data_absolute = []
    data_norm = []

    # Capturar um quadro de vídeo
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Espelhar a exibição

    # tamanho da tela    
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar os resultados da detecção de mãos
    results = hands.process(frame_rgb)

    # se pelo menos uma mão for reconhecida
    if results.multi_hand_landmarks:

        # para cada vetor de pontos da mão
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # imagem para desenhar
                hand_landmarks,  # saída do modelo
                mp_hands.HAND_CONNECTIONS,  # conexões entre pontos
                mp_drawing_styles.get_default_hand_landmarks_style(), #estilo
                mp_drawing_styles.get_default_hand_connections_style())

        # para cada vetor de pontos da mão
        for hand_landmarks in results.multi_hand_landmarks:
            
            # percorre todos os pontos de cada mão
            for i in range(len(hand_landmarks.landmark)):
                # valores absolutos de ponto base '0'
                if i == 0:
                    x_base = hand_landmarks.landmark[i].x
                    y_base = hand_landmarks.landmark[i].y

                # valores dos demais pontos
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                # salva em uma lista auxiliar os valores absolutos dos pontos
                data_absolute.append(x * width)
                data_absolute.append(y * height)

                # calcula o valor de cada ponto em relação ao ponto base
                x = x - x_base
                y = y - y_base

                # salva os valores relativos ao ponto base em uma lista
                data_norm.append(x)
                data_norm.append(y)

            # encontra o valor da coordenada mais distante do ponto base
            max_value = max(list(map(abs, data_norm)))

            # função de normalização
            def normalize_(n):
                return n / max_value

            # normaliza a lista de pontos relativos
            data_norm = list(map(normalize_, data_norm))

        # caso tenha mais de uma mão, apenas desenha 2 na tela
        if len(data_norm) < 44:
            # recebe qual o sinal a mão está fazendo
            prediction = model.predict([np.asarray(data_norm)])
            # nome do sinal
            predicted_character = labels_dict[int(prediction[0])]

            # posição do mouse no display
            x_cursor, y_cursor = pyautogui.position()

            # movimenta o cursor do mouse apenas se a mão não estiver fechada
            if predicted_character != 'Fechado':

                if position_3y != -1:
                    # Calcula a variação da posição da mão com base em um ponto base
                    delta_x = -((position_2x - data_absolute[2]) * 2)
                    delta_y = -((position_3y - data_absolute[3]) * 2)

                    # verifica se é possível mover o cursor sem sair da tela
                    if (x_cursor + delta_x) > (width-2) or (x_cursor + delta_x) < 2:
                        delta_x = 0
                    if (y_cursor + delta_y) > (height-2) or (y_cursor + delta_y) < 2:
                        delta_y = 0
                    # move o cursor
                    pyautogui.moveRel(delta_x, delta_y)

            # pega o valor do ponto base
            position_2x = data_absolute[2]
            position_3y = data_absolute[3]

            # se o gesto é de Mouse
            if predicted_character == 'Mouse':
                # se o estado anterior é o botão esquerdo
                if state == 'Esquerdo':
                    # solta o botão
                    pyautogui.mouseUp(button='left')
                    print('Click Esquerdo')

                # se o estado anterior é o botão direito
                if state == 'Direito':
                    # clica com o botão direito
                    pyautogui.rightClick(x_cursor, y_cursor)
                    print('Click Direito')
                state = 'Mouse'


            # se o gesto é o botão esquerdo
            if predicted_character == 'Esquerdo':
                if state != 'Fechado' and state != 'Esquerdo':
                    # aperta o botão esquedo
                    pyautogui.mouseDown(button='left', x=x_cursor, y=y_cursor)
                state = 'Esquerdo'


            # se o gesto é o botâo direito
            if predicted_character == 'Direito':
                if state != 'Fechado':
                    state = 'Direito'

            
            # Se o gesto é de copiar
            if predicted_character == 'Copiar':
                # se o estado não é de copiar
                if state != 'Copiar':
                    # realiza o comando de copiar
                    pyautogui.hotkey('ctrl', 'c')
                    print('Copiar')
                state = 'Copiar'


            # Se o gesto é de colar
            if predicted_character == 'Colar':
                # se o estado não é de colar
                if state != 'Colar':
                     # realiza o comando de colar
                    pyautogui.hotkey('ctrl', 'v')
                    print('Colar')
                state = 'Colar'


            # Se o gesto é de pinça
            if predicted_character == 'Pinca':
                # se o estado é o Aberto
                if state == 'Aberto':
                    # da zoom out
                    pyautogui.hotkey('ctrl', '-')
                state = 'Pinca'


            # Se o gesto é da mão aberta
            if predicted_character == 'Aberto':
                # se o estado é de pinça
                if state == 'Pinca':
                    # da zoom in
                    pyautogui.hotkey('ctrl', '+')
                state = 'Aberto'


            # Se o gesto é de mão fechada
            if predicted_character == 'Fechado':
                state = 'Fechado'


            # se o gesto é o de apontar para cima
            if predicted_character == 'Cima':
                state = 'Cima'
                # da scroll em 80 pixels para cima
                pyautogui.scroll(80)


            # se o gesto é o de apontar para baixo
            if predicted_character == 'Baixo':
                state = 'Baixo'
                # da scroll em 80 pixels para baixo
                pyautogui.scroll(-80)

            # escreve na tela o gesto atual
            cv2.putText(frame, predicted_character, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()