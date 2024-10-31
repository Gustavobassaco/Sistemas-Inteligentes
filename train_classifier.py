import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

# Função auxiliar para exibir a matriz de confusão
def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    # Plotagem da matriz de confusão como um mapa de calor
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()

    # Imprime o relatório de classificação
    if report:
        print('Resultados da classificação')
        print(classification_report(y_test, y_pred))

# Carrega os dados a partir do arquivo pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Divisão dos dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Criação e treinamento do modelo de classificação
# Para usar o modelo de Random Forest, descomente as duas linhas abaixo e comente as duas linhas seguintes
#model = RandomForestClassifier()
#model.fit(x_train, y_train)
model = SVC(kernel='rbf')
model.fit(x_train, y_train)

# Previsão das classes para os dados de teste
y_predict = model.predict(x_test)

# Cálculo da precisão do modelo
score = accuracy_score(y_predict, y_test)
print('{}% das amostras foram classificadas corretamente!'.format(score * 100))

# Exibição da matriz de confusão
print_confusion_matrix(y_test, y_predict)

# Salvando o modelo treinado em um arquivo pickle
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)