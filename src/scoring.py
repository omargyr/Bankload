# Código de Scoring - Modelo de Riesgo de Default en un Banco de Corea
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


# Leemos el modelo entrenado para usarlo
def cargar_modelo(df):
    package = './models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    return model



# Cargar la tabla transformada
def score_model(df, model, scores):
    # Predecimos sobre el set de datos de Scoring    
    res = model.predict(df).reshape(-1,1)
    pred = pd.DataFrame(res, columns=['PREDICT'])
    pred.to_csv(os.path.join('./data/scores/', scores))
    print(scores, 'exportado correctamente en la carpeta scores')


# Scoring desde el inicio
def main():
    df = read_file_csv('bankloan_score.csv')
    modelo=cargar_modelo(df)
    score_model(df, modelo, 'final_score.csv')
    print('Finalizó el Scoring del Modelo')


if __name__ == "__main__":
    main()


# df = read_file_csv('bankloan_score.csv')
# modelo=cargar_modelo(df)
# score_model(df, modelo, 'final_score.csv')
# res = modelo.predict(df).reshape(-1,1)
# pred = pd.DataFrame(res, columns=['PREDICT'])
