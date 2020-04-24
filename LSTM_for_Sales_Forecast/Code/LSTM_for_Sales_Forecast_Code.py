import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math, time
import datetime
import keras
from sklearn.utils import shuffle
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import model_from_json

# fixar random seed para se puder reproduzir os resultados
seed = 9
np.random.seed(seed)


def get_advertising_data(advertising_name, normalized=0, file_name=None):
    col_names = ['Month', 'Advertising', 'Sales', 'erro1', 'erro2', 'erro3',
                 'erro4']  # Eliminar as colunas de erro, uma vez que não contêm informação útil
    advertising = pd.read_csv(
        "advertising-and-sales-data-36-co.csv",
        sep=",", header=0,
        names=col_names)
    df = pd.DataFrame(advertising)
    date_split = df['Month'].str.split('-').str
    df['Ano'], df['Month'] = date_split

    df.drop(df.columns[[3, 4, 5, 6, 7]], axis=1,
            inplace=True)  # Ficar apenas com as colunas relevantes (Month, Advertising e Sales)
    df = df.drop(df.index[[36, 37]])  # Remover "lixo" nas últimas linhas do dataset

    return df


def load_advertising_dataset():
    advertising_name = 'advertising'
    return get_advertising_data(advertising_name, 0, "advertising-and-sales-data-36-co.csv")


def pre_processar_advertising_dataset(df):
    # Normalizar os dados
    df['Advertising'] = df['Advertising'] / 10
    df['Sales'] = df['Sales'] / 10

    conversion = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12,
    }

    meses = []
    for line in df['Month']:
        meses.append(conversion[line])
    df['Month'] = meses
    df = pd.get_dummies(df, columns=['Month'])  # Categorizar os meses do ano (vetor binário)

    # Ano de registo de vendas - NÃO UTILIZADO NO MODELO ÓTIMO
    # ano = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    # df['Ano'] = ano
    # df = pd.get_dummies(df, columns=['Ano'])
    # print(df)

    # Estações do ano
    estacoes = [4, 4, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                4]
    df['Estacoes'] = estacoes
    df = pd.get_dummies(df, columns=['Estacoes'])  # categorizar estações do ano (vetor binário)

    # Número de dias de cada mês no dataset - NÃO UTILIZADO NO MODELO ÓTIMO
    # dias = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31,
    #        30, 31, 30, 31, 31, 30, 31, 30, 31]
    # df['Dias'] = dias
    # df['Dias'] = df['Dias']/10

    col_list = list(df)
    col_list[1], col_list[18] = col_list[18], col_list[1]
    df = df.ix[:, col_list]

    return df


def load_data(df_dados, janela):
    qt_atributos = len(df_dados.columns)
    mat_dados = df_dados.as_matrix()  # converter dataframe para matriz (lista com lista de cada registo)
    # print(mat_dados)
    tam_sequencia = janela + 1
    res = []
    for i in range(len(mat_dados) - tam_sequencia):  # numero de registos - tamanho da sequencia
        res.append(mat_dados[i: i + tam_sequencia])
    res = np.array(res)  # dá como resultado um np com uma lista de matrizes (janela deslizante ao longo da serie)
    qt_casos_treino = int(
        round(0.667 * res.shape[0]))  # 2/3 passam a ser casos de treino (2 primeiros anos para treino)
    # print(qt_casos_treino, res.shape[0])
    train = res[:qt_casos_treino, :]
    x_train = train[:, :-1]  # menos um registo pois o ultimo registo é o registo a seguir à janela
    y_train = train[:, -1][:, -1]  # para ir buscar o último atributo para a lista dos labels
    x_test = res[qt_casos_treino:, :-1]
    y_test = res[qt_casos_treino:, -1][:, -1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], qt_atributos))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], qt_atributos))
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    # print(x_train)
    return [x_train, y_train, x_test, y_test]


def build_model(janela):
    from keras.layers import Bidirectional
    from keras.constraints import max_norm

    model = Sequential()

    model.add(Bidirectional(
        LSTM(64, activation='relu', return_sequences=True, input_shape=(janela, 19), kernel_constraint=max_norm(3),
             recurrent_constraint=max_norm(3), bias_constraint=max_norm(3))))
    model.add(LSTM(64, activation='relu', return_sequences=False, kernel_constraint=max_norm(3),
                   recurrent_constraint=max_norm(3), bias_constraint=max_norm(3)))
    model.add(Dropout(0.45))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.15))
    model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model


def print_history_loss(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def print_series_prediction(y_test, predic):
    diff = []
    racio = []
    for i in range(len(y_test)):  # para imprimir tabela de previsoes
        racio.append((y_test[i] / predic[i]) - 1)
        diff.append(abs(y_test[i] - predic[i]))
        print('valor: %f ---> Previsão: %f Diff: %f Racio: %f' % (y_test[i], predic[i], diff[i], racio[i]))
    plt.plot(y_test, color='blue', label='y_test')
    plt.plot(predic, color='red', label='prediction')
    # plt.plot(diff,color='green', label='diff')
    # plt.plot(racio,color='yellow', label='racio')
    plt.legend(loc='upper left')
    plt.show()


def LSTM_utilizando_advertising_data():
    df = load_advertising_dataset()
    df = pre_processar_advertising_dataset(df)
    print("df", df.shape)
    print("tamanho do dataset", len(df))
    janela = 3  # tamanho da Janela deslizante 3
    X_train, y_train, X_test, y_test = load_data(df, janela)
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)
    #model = build_model(janela)
    #history = model.fit(X_train, y_train, epochs=2000, validation_split=0.01, verbose=1)
    #print_history_loss(history)
    #save_model_json(model, "modelo.json")
    #save_weights_hdf5(model, "modelo.h5")
    model = load_model_json("modelo.json")
    load_weights_hdf5(model, "modelo.h5")
    compile_model(model)
    #trainScore = model.evaluate(X_train, y_train, verbose=0)
    #print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    #print(model.metrics_names)
    p = model.predict(X_test)
    predic = np.squeeze(
        np.asarray(p))  # para transformar uma matriz de uma coluna e n linhas em um np array de n elementos
    print_series_prediction(y_test, predic)

def save_model_json(model,fich):
    model_json = model.to_json()
    with open(fich, "w") as json_file:
        json_file.write(model_json)

def load_model_json(fich):
     json_file = open(fich, 'r')
     loaded_model_json = json_file.read()
     json_file.close()
     loaded_model = model_from_json(loaded_model_json)
     return loaded_model

def save_weights_hdf5(model, fich):
    model.save_weights(fich)
    print("Saved model to disk")

def load_weights_hdf5(model,fich):
    model.load_weights(fich)
    print("Loaded model from disk")

def compile_model(model):
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    LSTM_utilizando_advertising_data();
