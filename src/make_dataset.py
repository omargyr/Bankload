# Script de Preparación de Datos
###################################

import pandas as pd
import numpy as np
import os
from category_encoders import TargetEncoder
from sklearn import preprocessing


# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('./data/raw/', filename)).set_index('id')
    print(filename, ' cargado correctamente')
    return df


# Tratamiento de outliers
def trat_outliers(data,x):
    p1 = data[x].quantile(0.05)
    p99 = data[x].quantile(0.95)    
    data[x] = np.where(data[x] < p1, p1, data[x])
    data[x] = np.where(data[x] > p99, p99, data[x])
    return data


# Realizamos la transformación de datos
def data_preparation(df,target=True):

    # tratamiento de outliers
    df = trat_outliers(df,'income')
    df = trat_outliers(df,'creddebt')
    df = trat_outliers(df,'othdebt')

    # Target Encoding
    df.replace({'ed':{1:'secundaria incompleto',
                                2:'secundaria completa',
                                3:'universitaria incompleto',
                                4:'universitaria completa',
                                5:'posgrado'}},inplace=True)

    # # Target Encoding
    # encoder = TargetEncoder()
    # df['ed_num']= encoder.fit_transform(df['ed'], df['default'])    
    # dict_ed=df['ed_num'].set_index('ed').T.to_dict('list')
    if target==True:

        # normalizar y/o escalar
        var=['age', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt']
        X = df[var]
        std_scale_train = preprocessing.StandardScaler().fit(X)
        X[var] = std_scale_train.transform(X)
        y = df.default

        X['default']=y
        return X
    else:
        # normalizar y/o escalar

        var=['age', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt']
        X = df[var]
        std_scale_train = preprocessing.StandardScaler().fit(X)
        X[var] = std_scale_train.transform(X)
        return X


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, filename):
    df.to_csv(os.path.join('./data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación
def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('bankloan.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1,'bankloan_train.csv')
    # Matriz de Validación
    df2 = read_file_csv('bankloan_new.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2,'bankloan_val.csv')
    # Matriz de Scoring
    df3 = read_file_csv('bankloan_score.csv')
    tdf3 = data_preparation(df3,target=False)
    data_exporting(tdf3,'bankloan_score.csv')
    
if __name__ == "__main__":
    main()