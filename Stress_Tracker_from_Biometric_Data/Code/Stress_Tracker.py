# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from sklearn import impute, preprocessing, metrics
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV


import matplotlib
import pandas as pd
import numpy as np
import seaborn as sb
import sklearn as skl

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=skl.exceptions.DataConversionWarning)
warnings.filterwarnings("ignore", category=skl.exceptions.ConvergenceWarning)



def make_graphics(df):
    fig = plt.figure()
    fig.set_figheight(100)
    fig.set_figwidth(15)
    colNames = df.columns
    fig.subplots_adjust(hspace=0.6, wspace=0.6)
    for i in range(2,len(colNames)+1):

        ax = fig.add_subplot(18,3,3*i-5)
        ax.boxplot(x=df[colNames[i-1]],showmeans=True)
        ax.set_title(colNames[i-1])

        ax = fig.add_subplot(18,3,3*i-4)
        ax.hist(x=df[colNames[i-1]], bins=20)
        ax.axvline(df[colNames[i-1]].mean(), color='k', linestyle='dashed', linewidth=1)
        ax.set_title(colNames[i-1])
        ax.plot()

        ax = fig.add_subplot(18,3,3*i-3)
        ax.scatter(x=df[colNames[i-1]], y=df['PSS_Stress'])
        ax.set_ylabel('PSS_Stress')
        ax.set_title(colNames[i-1])
    return fig


#dados_original = pd.read_csv('/Users/JoelRodrigues/Documents/MIEI/4Ano/1Semestre/SI-AEC/Extração/TP/datasets/Dataset_DecisionPSS.csv', sep=',')
dados_original = pd.read_csv('Dataset_DecisionPSS.csv', sep=',')

dados = dados_original.drop(['StudyID'],axis=1)

label_encoder = preprocessing.LabelEncoder()
input_classes = dados['ExamID'].unique()
label_encoder.fit(input_classes)
dados['ExamID'] = label_encoder.transform(dados['ExamID'])

colNames = dados.columns

# Modificar Nan para média
imp = impute.SimpleImputer(missing_values=np.nan,strategy="mean")
dados = pd.DataFrame(imp.fit_transform(dados))
dados.set_axis(colNames,axis=1)


###################################################################################################################
############################################--- Visualização dos dados ----########################################
###################################################################################################################

dados
graficos_inicial = make_graphics(dados)
# plt.savefig('../figuras/visualization.png')

# Boxplot Classes dos exames vs Stress
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2 = sb.boxplot(x='ExamID',y='PSS_Stress',data=dados,palette="GnBu_d")
# plt.savefig('../figuras/exame_box.png')

# histograma Classes dos exames
sb.catplot(x='ExamID',kind='count',data=dados, height=7, palette='GnBu_d')
# plt.savefig('../figuras/exame_hist.png')

# histograma Classes das totalQuestions
sb.catplot(x='TotalQuestions',kind='count', data = dados, height=7, palette='GnBu_d')
# plt.savefig('../figuras/totalquestions_hist.png')

# scatterplot Racio de boas decisoes e tempo de boas decisoes
sb.scatterplot(x='CDMR', y='GDTE',data = dados)
sb.scatterplot(x='DMR', y='DTE',data = dados)

condicoes_outliers = (dados_original['ATBD'] < 5000)  & (dados_original['DTE'] < 20000)  & (dados_original['GDTE'] < 20000) & (dados_original['MinDuration'] > 40000) & (dados_original['NumDecisions'] > 2500) & (dados_original['QuestionsEnter'] > 2000)
outliers = dados_original[ condicoes_outliers ]
outliers

dados_sem_outliers = dados_original.drop(['StudyID'],axis=1)
dados_sem_outliers = dados_sem_outliers.drop(outliers.index,axis=0)
dados_sem_outliers
dados_sem_outliers['PSS_Stress']

graficos_sem_outliers = make_graphics(dados_sem_outliers)
plt.savefig('visualization2.png')
graficos_sem_outliers

#Percentagem de NaN em cada coluna:
(len(dados_original) - dados_original.count()) / len(dados_original)


# Escolha de features

features = dados_sem_outliers.drop(['ExamID', 'TotalQuestions', 'PSS_Stress'], axis=1)
target = pd.DataFrame(dados_sem_outliers['PSS_Stress'])
target.set_axis(['PSS_Stress'],axis=1)

features
target
# Normalização dos dados  para [0,1]
features_normalized = preprocessing.normalize(features,norm='max',axis=0)
features_normalized = pd.DataFrame(features_normalized, columns=features.columns)
features_normalized


target_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
target_scaler.data_max_ = 52
target_scaler.data_min_ = 0
target_scaler.data_range_ = 52

target_normalized = target_scaler.fit_transform(target)
target_normalized = pd.DataFrame(target_normalized, columns=target.columns)
target_normalized


# Padronização dos dados
features_standard = preprocessing.scale(features)
features_standard = pd.DataFrame(features_standard, columns=features.columns)
features_standard

target_standard = preprocessing.scale(target)
target_standard = pd.DataFrame(target_standard, columns=target.columns)
target_standard

# Discretização


target_discretizer = pd.cut(target['PSS_Stress'], [0,10,20,30,40,52])
target_discretizer2 = pd.qcut(target['PSS_Stress'], 10)
target_discretized2 = label_encoder.fit_transform(target_discretizer2)
target_discretized2


label_encoder = preprocessing.LabelEncoder()
target_discretized = label_encoder.fit_transform(target_discretizer)
target_discretized
sb.countplot(x=target_discretizer,palette="GnBu_d")
sb.countplot(x=target_discretizer2,palette="GnBu_d")


target_normalized_discretizer = pd.cut(target_normalized['PSS_Stress'],5)
target_normalized_discretizer.value_counts(sort=False)
label_encoder = preprocessing.LabelEncoder()
target_normalized_discretized = label_encoder.fit_transform(target_normalized_discretizer)
target_normalized_discretized

target_standard_discretizer = pd.cut(target_standard['PSS_Stress'],5)
target_standard_discretizer.value_counts(sort=False)
label_encoder = preprocessing.LabelEncoder()
target_standard_discretized = label_encoder.fit_transform(target_standard_discretizer)
target_standard_discretized



# Testado com k = 2 até k = 15 (sem variação nos resultados)
kfold = KFold(n_splits = 10)

#Classificaçao
svm = SVC()
gaussianNB = GaussianNB()
lr = LogisticRegression()
knn = KNeighborsClassifier()
rna_clf = MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=(100,4), random_state=1)



score = cross_val_score(svm, X=features_normalized, y=target_discretized2,cv=kfold )
score.mean()


#Regressao
linearR = LinearRegression()
svr = SVR()
rna_reg = MLPRegressor(solver='adam', alpha=0.0001, hidden_layer_sizes=(100,4), random_state=1)

all_features = [('f_norm',features_normalized), ('f_stand',features_standard)]
all_targets_discretized = [('t_disc', target_discretized), ('t_stand_disc',target_standard_discretized), ('t_stand_norm',target_normalized_discretized), ('t_qcut', target_discretized2)]
all_models_classification = [('SVM', svm), ('GNB', gaussianNB), ('LR', lr), ('KNN', knn), ('RNA_CLF',rna_clf)]

all_models_regression = [('linearR',linearR),('SVR', svr),('RNA_REG', rna_reg)]
all_targets = [('t_norm', target_normalized), ('t_stand', target_standard)]

scores_regression = []
scores_classification = []
trained_models_regression = []
trained_models_classification = []
for (f_name, f) in all_features:
    for (t_name, t) in all_targets:

        x_train,x_test,y_train,y_test = train_test_split(f,t,test_size=0.2)

        for (m_name,m) in all_models_regression:
            cross_score = cross_val_score(m,f,t,cv=kfold).mean()
            scores_regression.append([m_name,cross_score,t_name, f_name])

            m_copy = clone(m)
            m_copy.fit(X=x_train, y=y_train)
            m_copy.predict(x_test)
            m_score = m_copy.score(x_test,y_test)
            trained_models_regression.append([m_copy, m_name, m_score, t_name, f_name])

for (f_name, f) in all_features:
    for (t_name, t) in all_targets_discretized:

        x_train,x_test,y_train,y_test = train_test_split(f,t,test_size=0.2)

        for (m_name,m) in all_models_classification:
            cross_score = cross_val_score(m,f,t,cv=kfold).mean()
            scores_classification.append([m_name,cross_score,t_name, f_name])

            m_copy = clone(m)
            m_copy.fit(X=x_train, y=y_train)
            m_copy.predict(x_test)
            m_score = m_copy.score(x_test,y_test)
            trained_models_classification.append([m_copy, m_name, m_score, t_name, f_name])


for i in range (len(trained_models_regression)):
    print(trained_models_regression[i][1], round(trained_models_regression[i][2],3), trained_models_regression[i][3], trained_models_regression[i][4])

for i in range (len(trained_models_classification)):
    if (round(trained_models_classification[i][2],3) > 0.55):
        print(trained_models_classification[i][1], round(trained_models_classification[i][2],3), trained_models_classification[i][3], trained_models_classification[i][4])

for i in range (len(scores_regression)):
        #if (round(scores_regression[i][1],3) > 0):
            print(scores_regression[i][0], round(scores_regression[i][1],3), scores_regression[i][2], scores_regression[i][3])
for i in range (len(scores_classification)):
        #if (round(scores_classification[i][1],3)>0.5):
            print(scores_classification[i][0], round(scores_classification[i][1],3), scores_classification[i][2], scores_classification[i][3])


# Teste RFE (não utilizado)
res=[]
for k in range (1,len(features_normalized)):
    rfe = RFE(lr,k)
    rfe.fit(features_normalized, target_discretized)

    rfe.fit(features_normalized, target_discretized).support_
    features_a_utilizar = [i for i, x in enumerate(rfe.support_) if not(x)]
    features_a_utilizar = features_normalized.columns[features_a_utilizar]

    features_a_utilizar = features_normalized[features_a_utilizar]
    aaa.append(cross_val_score(lr, features_a_utilizar, target_discretized,cv=kfold).mean())

res

# Testado com k = 1 até k = 13 (sem variação nos resultados)
kbest = SelectKBest(score_func=chi2, k=10)
features_normalized.columns[kbest.fit(features_normalized, target_discretized).get_support()]


# Hiperparametrização 
param_grid = {'C':[0.01,0.1,1,10,100],'penalty':['l1','l2']}
lr = LogisticRegression()
gridSearch = GridSearchCV(lr,param_grid)
gridSearch.fit(features_normalized,target_discretized)
gridSearch.best_params_
lr.C=0.001
lr.penalty='l2'
x_train,x_test,y_train,y_test = train_test_split(features_normalized,target_discretized,test_size=0.2)
lr.fit(X=x_train,y=y_train)
lr.score(x_test,y_test)

lr = LogisticRegression()
lr.fit(X=x_train,y=y_train)
lr.score(x_test,y_test)
