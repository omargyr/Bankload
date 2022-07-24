# Código de Evaluación - Modelo de Riesgo de Default en un Banco de Corea
############################################################################

import pandas as pd
import dataframe_image as dfi
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn import metrics
import os


# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('./data/processed/', filename)).set_index('id')
    print(filename, ' cargado correctamente')
    return df

# Leemos el modelo entrenado para usarlo
def cargar_modelo():
    package = './models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    return model


def report_param(y_t,y_p):
    report_metricas=metrics.classification_report(y_t,y_p, digits = 6, output_dict=True)
    print('Classification report:\n\n{}'.format(
    pd.DataFrame(report_metricas).transpose()
    ))
    dfi.export(pd.DataFrame(report_metricas).transpose(), os.path.join('./models/', 'parametros.png'))


def conf_m(model,X_t,y_t):
    print("\nMatriz de Confusión")
    disp = metrics.plot_confusion_matrix(model, X_t, y_t,
                                cmap=plt.cm.Blues,
                                values_format="")
    disp.ax_.set_title("Confusion Matrix")
    plt.savefig(os.path.join('./models/', 'Matrix_confusion.png'))
    # plt.show()

def fig_ROC(y_t,y_s):
  # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in [0,1]:
        fpr[i], tpr[i], _ = roc_curve(y_t, y_s)
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_t.ravel(), y_s.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print("\nGráfico ROC")
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.0001, 1.0])
    plt.ylim([0.0, 1.0001])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join('./models/', 'ROC.png'))
    # plt.show()

def total_param(model, X_t, y_t):
    # Model: Modelo a evaluar
    # X_t: Datos de X_test
    # y_t: Datos de y_test
    y_p = model.predict(X_t) #Valores predichos
    y_s = model.predict_proba(X_t)[:, 1] #Probabilidad para cada valor de clase 1
    conf_m(model,X_t,y_t) #Gráfico de Matriz de Confusión
    report_param(y_t,y_p) #Reporte de parametros
    fig_ROC(y_t,y_s)  #Gráfico ROC


def eval_model(df,modelo):
    X = df.drop('default',axis=1)
    y = df.default
    total_param(modelo, X, y)


# Validación desde el inicio
def main():
    df=read_file_csv('bankloan_val.csv')
    modelo=cargar_modelo()
    eval_model(df,modelo)
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()