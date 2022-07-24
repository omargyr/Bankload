# Código de Entrenamiento - Modelo de Riesgo de Default en un Banco de Corea
############################################################################

import pandas as pd
import xgboost as xgb
import pickle
import os


# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('./data/processed/', filename)).set_index('id')
    print(filename, ' cargado correctamente')
    return df


def entrenamiento(df): 
    X = df.drop('default',axis=1)
    y = df.default

    from sklearn.linear_model import LogisticRegression
    elastic = LogisticRegression(penalty = 'elasticnet',
                                class_weight="balanced",
                                solver = 'saga',
                                C = 0.25,
                                l1_ratio = 0.99,
                                max_iter=5,
                                random_state = 0) 
    modelo=elastic.fit(X,y)
    return modelo


# Guardamos el modelo entrenado para usarlo en produccion
def save_modelo(filename):
    package = './models/best_model.pkl'
    pickle.dump(filename, open(package, 'wb'))


# Entrenamiento completo
def main():
    df=read_file_csv('bankloan_train.csv')
    modelo=entrenamiento(df)
    save_modelo(modelo)
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()
