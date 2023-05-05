#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pingouin')


# In[ ]:


get_ipython().system('pip install lazypredict')


# In[150]:


# tratamiento de los datos
# ============================================
import pandas as pd
import numpy as np
from scipy.stats import skew
import scipy.stats as st
import sidetable
from scipy.stats import chi2_contingency
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
import pingouin as pg
from scipy.stats import kurtosistest
import statsmodels.api as sm
from lazypredict.Supervised import LazyRegressor

# librerías para la visualización de los datos
# ============================================
import matplotlib.pyplot as plt
import seaborn as sns

# Establecer tamaño fijo de gráficas
# ==================================
plt.rcParams["figure.figsize"] = (10,8)

# Configuración warnings
# ======================
import warnings
warnings.filterwarnings('ignore')


# # Objectives
# Description of data, identification of variables to be taken into account for the calculation of health insurance premiums.
# 
# Predict amount of charges

# In[249]:


df = pd.read_csv("../data/00-insurance.csv", index_col = 0).reset_index()
df.head(2)


# ## Understand the variables we have and what our dataframe looks like.

# In[177]:


# number of rows and columns in the dataframe

df.shape


# In[178]:


# general data frame information

df.info()


# In[154]:


df.isnull().sum()


# In[155]:


# duplicados 

df.duplicated().sum()


# In[156]:


df[df.duplicated()== True] 


# In[250]:


df.drop_duplicates(inplace=True)


# In[158]:


df.to_csv('insurance_ok.csv')


# ### Statistics

# In[180]:


# leading statistics of the numeric columns.

df.describe(include='all')


# Data integrity:
# - Smoking habits, majority non-smokers (274 smokers)
# - BMI mean 30, overweight (more than 75% over BMI 25)

# # Outliers

# In[160]:


numericas = df.select_dtypes(include=np.number)
numericas


# In[161]:


fig, axes = plt.subplots(2,2, figsize=(20,4))
axes = axes.flat
for indice, columna in enumerate(numericas.columns):
    sns.boxplot(x = numericas[columna], data = df, ax=axes[indice], color = "turquoise") 
plt.tight_layout()
plt.show();


# In[162]:


# I'm going to look at the distribution of variable charges
sns.set(style="ticks")
sns.set_style("darkgrid")
sns.distplot(
    df["charges"], 
    color = "blue", 
    kde_kws = {"shade": True, "linewidth": 1});


# ## We analyse the numerical variables of the dataset

# ### Distributions

# In[163]:


fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (30, 10))
axes = axes.flat#iterator
for i, colum in enumerate(numericas.columns): 
    sns.histplot(
        data = numericas[colum],
        kde = True,
        color = "purple", 
        line_kws = {"linewidth": 2}, 
        alpha = 0.5, 
        ax = axes[i])
    axes[i].set_title(colum, fontsize = 20, fontweight = "bold")
    axes[i].tick_params(labelsize = 20)
    axes[i].set_xlabel("")
fig.tight_layout();


# ### Relationship to the response variable

# In[164]:


fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (30, 10))
axes = axes.flat
lista_colores = ["cyan", "magenta", "orange"]

columnas_numeric = df.select_dtypes(include = np.number).columns
columnas_numeric = columnas_numeric.drop("charges")

for i, colum in enumerate(columnas_numeric):
    sns.regplot(
        x = df[colum], 
        y = df["charges"], 
        color = lista_colores[i], 
        marker = ".", 
        scatter_kws = {"alpha": 0.4}, 
        line_kws = {"color": "black", "alpha": 0.7 }, 
        ax = axes[i])
    
    axes[i].set_title(f"Charges vs {colum}", fontsize = 20, fontweight = "bold")
    axes[i].tick_params(labelsize = 20)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    
fig.tight_layout();


# In[165]:


fig, axes = plt.subplots(1, 3, figsize=(20,7))
categoricas=df.select_dtypes(include='object')
for i in range(len(categoricas.columns)): 
    sns.scatterplot(x='age', y='charges', data = df,
                    s = 25,
                    hue = categoricas.columns[i], 
                    ax=axes[i])   
plt.show(); 


# Here we see that in age, there are about three clear trend lines in the distribution of our data.  
# **-------------------------------------All 3 increase with age and tobacco-------------------------------------**

# In[166]:


fig, axes = plt.subplots(1, 3, figsize=(20,7))

for i in range(len(categoricas.columns)): 
    sns.scatterplot(x='bmi', y='charges', data = df,
                    s = 25,    
                    hue = categoricas.columns[i], 
                    ax=axes[i])  
plt.show(); 


# **-------------------------------We see that there is a clear increase  in charges with tobacco--------------------------**

# In[167]:


mask = np.triu(np.ones_like(df.corr(), dtype = np.bool))
sns.heatmap(df.corr(method='spearman'), 
           cmap = "icefire", 
            mask = mask,
           annot = True);


# ## Categorical variables

# ### I create some graphs to go deeper into the data

# In[168]:


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 15))
axes = axes.flat
lista_colores = [ "magenta", "orange","blue", "green"]
columnas = df.select_dtypes(include = 'object')
columnas = columnas.columns
for i, colum in enumerate(columnas): 
    sns.violinplot(
        data = df,
        x = colum,
        y = 'charges',
        color = lista_colores[i],
        line_kws = {"color": "black", "alpha": 0.7 }, 
        ax = axes[i])
    axes[i].set_title(colum, fontsize = 15, fontweight = "bold")
    axes[i].tick_params(labelsize = 20)
    axes[i].set_xlabel("")

fig.tight_layout();


# In[169]:


sns.barplot(x='region', y='charges', data=df, palette='Spectral')
plt.ylim(0,40000);


# In[171]:


sns.barplot(x='children', y='charges', data=df, palette='Set2')
plt.ylim(0,40000);


# In[172]:


sns.barplot(x='smoker', y='charges', data=df, palette='Spectral')
plt.ylim(0,40000);


# In[173]:


sns.barplot(x='sex', y='charges' ,data=df, palette='Spectral')
plt.ylim(0,40000);


# modelo

# In[251]:


df['sex'] = df['sex'].map({'female':0,'male':1})
df['smoker'] = df['smoker'].map({'no':0,'yes':1})
df['region'] = df['region'].map({'northeast':1,'northwest':2,'southeast':3,'southwest':4})
df.head()


# In[182]:


df.info()


# In[252]:


X = df.drop('charges',axis=1)
y = df['charges']


# In[253]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


# In[254]:


from lazypredict.Supervised import LazyRegressor


# In[255]:


clf = LazyRegressor(verbose=0)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)
models


# In[256]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[257]:


# iniciamos la regresión lineal. n_jobs hace referencia al número de nucleos que usaremos de nuestro ordenador. Al indicar -1 significa
#  que los usaremos todos. 

lr = LinearRegression(n_jobs=-1)


# In[259]:


# fiteamos el modelo, lo que significa que le pasamos los datos de entrenamiento para que aprenda el algoritmo. Fijaros que para que aprenda,
#  les paso solo los datos de entrenamiento

lr.fit(x_train, y_train)


# In[260]:


# es el momento de hacer las predicciones, para eso usarmos el método predict() de sklearn 

y_predict_train = lr.predict(x_train) # hacemos las predicciones para las casas que tenemos en el los datos de entrenamiento
y_predict_test = lr.predict(x_test) # hacemos las predicciones para las casas que tenemos en los datos de test


# In[261]:


train_df = pd.DataFrame({'Real': y_train, 'Predicted': y_predict_train, 'Set': ['Train']*len(y_train)})
test_df  = pd.DataFrame({'Real': y_test,  'Predicted': y_predict_test,  'Set': ['Test']*len(y_test)})
resultados = pd.concat([train_df,test_df], axis = 0)
resultados.head()


# In[262]:


resultados['residuos'] = resultados['Real'] - resultados['Predicted']
resultados.head()


# In[222]:


fig, ax = plt.subplots(2,2,figsize=(20,20))


# ploteamos los reales vs los predichos
sns.regplot(data = resultados[resultados['Set'] == "Train"], 
            x = "Real", 
            y = "Predicted", 
            ax = ax[0,0], 
            color = "grey",
            line_kws = {"color": "red", "alpha": 0.7 })


sns.regplot(data = resultados[resultados['Set'] == "Test"], 
            x = "Real",
            y = "Predicted", 
            color = "gray",
            line_kws = {"color": "red", "alpha": 0.7 }, 
            ax = ax[1,0])


# ploteamos los residuos
sns.histplot(resultados[resultados['Set'] == "Train"],
             x="residuos",
             color ="grey",
             kde=True, 
             ax = ax[0,1])


sns.histplot(resultados[resultados['Set'] == "Test"],
             x="residuos",
             color = "grey",
             kde=True, 
             ax = ax[1,1])

ax[0,0].set_title("Train reales vs predichos", fontsize = 15, fontweight = "bold")
ax[0,1].set_title("Train residuos", fontsize = 15, fontweight = "bold")
ax[1,0].set_title("Test reales vs predichos", fontsize = 15, fontweight = "bold")
ax[1,1].set_title("Test residuos", fontsize = 15, fontweight = "bold");


# In[263]:


resultados_metricas = {'MAE': [mean_absolute_error(y_test, y_predict_test), mean_absolute_error(y_train, y_predict_train)],
                'MSE': [mean_squared_error(y_test, y_predict_test), mean_squared_error(y_train, y_predict_train)],
                'RMSE': [np.sqrt(mean_squared_error(y_test, y_predict_test)), np.sqrt(mean_squared_error(y_train, y_predict_train))],
                'R2':  [r2_score(y_test, y_predict_test), r2_score(y_train, y_predict_train)],
                 "set": ["test", "train"], 
                 "modelo": ["Linear Regresion", "LinearRegression"]}

df_resultados = pd.DataFrame(resultados_metricas)

df_resultados


# Smokers

# In[227]:


dffum = df[(df['smoker'] == 1)]
dffum


# In[228]:


X_fum = dffum.drop('charges',axis=1)
y_fum= dffum['charges']


# In[229]:


from sklearn.model_selection import train_test_split
X_train_fum, X_test_fum, y_train_fum, y_test_fum = train_test_split( X_fum, y_fum, test_size=0.2, random_state=42)
X_train_fum.shape, X_test_fum.shape


# In[230]:


clf_fum = LazyRegressor(verbose=0)
models_fum,predictions_fum = clf_fum.fit(X_train_fum, X_test_fum, y_train_fum, y_test_fum)
models_fum


# Non Smokers

# In[231]:


dfnfum = df[(df['smoker'] == 0)]
dfnfum.sample(5)


# In[232]:


X_nfum = dfnfum.drop('charges',axis=1)
y_nfum= dfnfum['charges']


# In[233]:


from sklearn.model_selection import train_test_split
X_train_nfum, X_test_nfum, y_train_nfum, y_test_nfum = train_test_split( X_nfum, y_nfum, test_size=0.2, random_state=42)
X_train_nfum.shape, X_test_nfum.shape


# In[234]:


clf_nfum = LazyRegressor(verbose=0)
models_nfum,predictions_nfum = clf_nfum.fit(X_train_nfum, X_test_nfum, y_train_nfum, y_test_nfum)
models_nfum


# Healthy BMI

# In[235]:


df_bmi_ok = df[(df['bmi'] <25)]
df_bmi_ok.sample(5)


# In[236]:


X_bmi_ok = df_bmi_ok.drop('charges',axis=1)
y_bmi_ok= df_bmi_ok['charges']


# In[237]:


from sklearn.model_selection import train_test_split
X_train_bmi_ok, X_test_bmi_ok, y_train_bmi_ok, y_test_bmi_ok = train_test_split( X_bmi_ok, y_bmi_ok, test_size=0.2, random_state=42)
X_train_bmi_ok.shape, X_test_bmi_ok.shape


# In[238]:


clf_bmi_ok = LazyRegressor(verbose=0)
models_bmi_ok,predictions_bmi_ok = clf_bmi_ok.fit(X_train_bmi_ok, X_test_bmi_ok, y_train_bmi_ok, y_test_bmi_ok)
models_bmi_ok


# Unhealthy BMI

# In[292]:


df = pd.read_csv("../data/00-insurance.csv", index_col = 0).reset_index()
df.head(2)


# In[293]:


df.drop_duplicates(inplace=True)


# In[294]:


df['sex'] = df['sex'].map({'female':0,'male':1})
df['smoker'] = df['smoker'].map({'no':0,'yes':5})
df['region'] = df['region'].map({'northeast':3,'northwest':1,'southeast':4,'southwest':2})
df.head()


# In[295]:


df_bmi_nok = df[(df['bmi'] >25)]
df_bmi_nok.sample(5)


# In[296]:


X_bmi_nok = df_bmi_nok.drop('charges',axis=1)
y_bmi_nok= df_bmi_nok['charges']


# In[297]:


from sklearn.model_selection import train_test_split
X_train_bmi_nok, X_test_bmi_nok, y_train_bmi_nok, y_test_bmi_nok = train_test_split( X_bmi_nok, y_bmi_nok, test_size=0.2, random_state=42)
X_train_bmi_nok.shape, X_test_bmi_nok.shape


# In[298]:


clf_bmi_nok = LazyRegressor(verbose=0)
models_bmi_nok,predictions_bmi_nok = clf_bmi_nok.fit(X_train_bmi_nok, X_test_bmi_nok, y_train_bmi_nok, y_test_bmi_nok)
models_bmi_nok


# In[273]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[299]:


# iniciamos la regresión lineal. n_jobs hace referencia al número de nucleos que usaremos de nuestro ordenador. Al indicar -1 significa
#  que los usaremos todos. 

lr = LinearRegression(n_jobs=-1)


# In[300]:


# fiteamos el modelo, lo que significa que le pasamos los datos de entrenamiento para que aprenda el algoritmo. Fijaros que para que aprenda,
#  les paso solo los datos de entrenamiento

lr.fit(X_train_bmi_nok, y_train_bmi_nok)


# In[301]:


# es el momento de hacer las predicciones, para eso usarmos el método predict() de sklearn 

y_predict_train_bmi_nok = lr.predict(X_train_bmi_nok) # hacemos las predicciones para las casas que tenemos en el los datos de entrenamiento
y_predict_test_bmi_nok = lr.predict(X_test_bmi_nok) # hacemos las predicciones para las casas que tenemos en los datos de test


# In[302]:


resultados_metricas_bmi_nok = {'MAE': [mean_absolute_error(y_test_bmi_nok, y_predict_test_bmi_nok), mean_absolute_error(y_train_bmi_nok, y_predict_train_bmi_nok)],
                'MSE': [mean_squared_error(y_test_bmi_nok, y_predict_test_bmi_nok), mean_squared_error(y_train_bmi_nok, y_predict_train_bmi_nok)],
                'RMSE': [np.sqrt(mean_squared_error(y_test_bmi_nok, y_predict_test_bmi_nok)), np.sqrt(mean_squared_error(y_train_bmi_nok, y_predict_train_bmi_nok))],
                'R2':  [r2_score(y_test_bmi_nok, y_predict_test_bmi_nok), r2_score(y_train_bmi_nok, y_predict_train_bmi_nok)],
                 "set": ["test", "train"], 
                 "modelo": ["Linear Regresion bmi nok", "LinearRegression bmi nok"]}

df_resultados_bmi_nok = pd.DataFrame(resultados_metricas_bmi_nok)

df_resultados_bmi_nok


# In[303]:


resultados = pd.concat([df_resultados_bmi_nok,df_resultados], axis = 0)
resultados.head()


# If BMI > 25, the new model, with the new map(smokers:5, and new distribution of regions), its better than the generic model (smokers:1), we can use it for better predictions in overweight population. 
