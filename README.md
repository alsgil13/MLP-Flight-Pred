
<h1> Estudos Sobre Atrasos de Voos no Brasil</h1>
<h2> Uma tentativa de predição de atrasos em chegadas de voos usando MultiLayer Perceptron</h2>
<p>Muitas das transformações dos dados utilizadas neste notebook foram aprendidas <a href="https://www.kaggle.com/microtang/exploring-brazil-flights-data">neste kernel</a> </p>
<h3>1. Importando as Bibliotecas que utilizaremos no projeto</h3>


```python
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
```

<h3>2. Importando DataSet disponível no <a href="https://www.kaggle.com/ramirobentes/flights-in-brazil">Kaggle</a></h3>
<p>Para os exemplos seguintes funcionarem você deve baixar o detaset no link acima e salvar na mesma pasta do projeto com o nome de: BrFlights2.csv</p>


```python
df = pd.read_csv('BrFlights2.csv', encoding='latin1')
```

<h3>3. Normalizando e traduzindo nomes das colunas e criando tabela para checagem dos tipos de variáveis </h3>


```python
df.columns = ['Flights', 'Airline', 'Flight_Type','Departure_Estimate','Departure_Real','Arrival_Estimate','Arrival_Real','Flight_Situation','Code_Justification','Origin_Airport','Origin_City','Origin_State','Origin_Country','Destination_Airport','Destination_City','Destination_State','Destination_Country','Destination_Long','Destination_Lat','Origin_Long','Origin_Lat']
tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values'}))
tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.
                         rename(index={0:'null values (%)'}))
tab_info
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Flights</th>
      <th>Airline</th>
      <th>Flight_Type</th>
      <th>Departure_Estimate</th>
      <th>Departure_Real</th>
      <th>Arrival_Estimate</th>
      <th>Arrival_Real</th>
      <th>Flight_Situation</th>
      <th>Code_Justification</th>
      <th>Origin_Airport</th>
      <th>...</th>
      <th>Origin_State</th>
      <th>Origin_Country</th>
      <th>Destination_Airport</th>
      <th>Destination_City</th>
      <th>Destination_State</th>
      <th>Destination_Country</th>
      <th>Destination_Long</th>
      <th>Destination_Lat</th>
      <th>Origin_Long</th>
      <th>Origin_Lat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>column type</th>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>...</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>float64</td>
      <td>float64</td>
      <td>float64</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>null values</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>289196</td>
      <td>0</td>
      <td>289196</td>
      <td>0</td>
      <td>1510212</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>null values (%)</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11.3744</td>
      <td>0</td>
      <td>11.3744</td>
      <td>0</td>
      <td>59.3983</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 21 columns</p>
</div>



<p> <i> ... <br> Constatamos que Departure_Real e Arrival_Real tem cerca de 11% de valores nulos. <br>
    Deletaremos esse valores no passo seguinte (Ignorando a coluna Code_Justification e dropando as ,'tuplas' com algum valor nulo) <br>....</i></p>

<h3>4. Criando um novo DataFrame com colunas calculadas para tratar os dados de Data/Hora </h3>


```python
df_time = df[['Flights', 'Airline', 'Flight_Type', 'Departure_Estimate',
       'Departure_Real', 'Arrival_Estimate', 'Arrival_Real',
       'Flight_Situation', 'Origin_Airport',
       'Origin_City', 'Origin_State', 'Origin_Country', 'Destination_Airport',
       'Destination_City', 'Destination_State', 'Destination_Country',
       'Destination_Long', 'Destination_Lat', 'Origin_Long', 'Origin_Lat']]
df_time.dropna(how='any',inplace=True)
df_time['Departure_Estimate'] = pd.to_datetime(df_time['Departure_Estimate'])
df_time['Departure_Real'] = pd.to_datetime(df_time['Departure_Real'])
df_time['Arrival_Estimate'] = pd.to_datetime(df_time['Arrival_Estimate'])
df_time['Arrival_Real'] = pd.to_datetime(df_time['Arrival_Real'])
df_time['Departure_Delays'] =df_time.Departure_Real - df_time.Departure_Estimate
df_time['Arrival_Delays'] = df_time.Arrival_Real - df_time.Arrival_Estimate
df_time['Departure_Delays'] = df_time['Departure_Delays'].apply(lambda x : round(x.total_seconds()/60))
df_time['Arrival_Delays'] = df_time['Arrival_Delays'].apply(lambda x : round(x.total_seconds()/60))
```

<h3>5. Adicionando nossa Classe Objetivo</h3>
<p>Estudaremos os atrasos nas Chegadas dos voos</p>


```python
df_time['ArrivalStatus'] = ""
df_time.loc[df_time.Arrival_Delays > 0 , 'ArrivalStatus'] = "Atrasado"
df_time.loc[df_time.Arrival_Delays < 0 , 'ArrivalStatus'] = "Adiantado"
df_time.loc[df_time.Arrival_Delays == 0 , 'ArrivalStatus'] = "Pontual"
```

<h3>6. Visualizando nosso DataFrame cheio de colunas e com a nossa classe :) </h3>


```python
df_time.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Flights</th>
      <th>Airline</th>
      <th>Flight_Type</th>
      <th>Departure_Estimate</th>
      <th>Departure_Real</th>
      <th>Arrival_Estimate</th>
      <th>Arrival_Real</th>
      <th>Flight_Situation</th>
      <th>Origin_Airport</th>
      <th>Origin_City</th>
      <th>...</th>
      <th>Destination_City</th>
      <th>Destination_State</th>
      <th>Destination_Country</th>
      <th>Destination_Long</th>
      <th>Destination_Lat</th>
      <th>Origin_Long</th>
      <th>Origin_Lat</th>
      <th>Departure_Delays</th>
      <th>Arrival_Delays</th>
      <th>ArrivalStatus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAL - 203</td>
      <td>AMERICAN AIRLINES INC</td>
      <td>Internacional</td>
      <td>2016-01-30 08:58:00+00:00</td>
      <td>2016-01-30 08:58:00+00:00</td>
      <td>2016-01-30 10:35:00+00:00</td>
      <td>2016-01-30 10:35:00+00:00</td>
      <td>Realizado</td>
      <td>Afonso Pena</td>
      <td>Sao Jose Dos Pinhais</td>
      <td>...</td>
      <td>Porto Alegre</td>
      <td>RS</td>
      <td>Brasil</td>
      <td>-51.175381</td>
      <td>-29.993473</td>
      <td>-49.172481</td>
      <td>-25.532713</td>
      <td>0</td>
      <td>0</td>
      <td>Pontual</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAL - 203</td>
      <td>AMERICAN AIRLINES INC</td>
      <td>Internacional</td>
      <td>2016-01-13 12:13:00+00:00</td>
      <td>2016-01-13 12:13:00+00:00</td>
      <td>2016-01-13 21:30:00+00:00</td>
      <td>2016-01-13 21:30:00+00:00</td>
      <td>Realizado</td>
      <td>Salgado Filho</td>
      <td>Porto Alegre</td>
      <td>...</td>
      <td>Miami</td>
      <td>N/I</td>
      <td>Estados Unidos</td>
      <td>-80.287046</td>
      <td>25.795865</td>
      <td>-51.175381</td>
      <td>-29.993473</td>
      <td>0</td>
      <td>0</td>
      <td>Pontual</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AAL - 203</td>
      <td>AMERICAN AIRLINES INC</td>
      <td>Internacional</td>
      <td>2016-01-29 12:13:00+00:00</td>
      <td>2016-01-29 12:13:00+00:00</td>
      <td>2016-01-29 21:30:00+00:00</td>
      <td>2016-01-29 21:30:00+00:00</td>
      <td>Realizado</td>
      <td>Salgado Filho</td>
      <td>Porto Alegre</td>
      <td>...</td>
      <td>Miami</td>
      <td>N/I</td>
      <td>Estados Unidos</td>
      <td>-80.287046</td>
      <td>25.795865</td>
      <td>-51.175381</td>
      <td>-29.993473</td>
      <td>0</td>
      <td>0</td>
      <td>Pontual</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AAL - 203</td>
      <td>AMERICAN AIRLINES INC</td>
      <td>Internacional</td>
      <td>2016-01-19 12:13:00+00:00</td>
      <td>2016-01-18 12:03:00+00:00</td>
      <td>2016-01-19 21:30:00+00:00</td>
      <td>2016-01-18 20:41:00+00:00</td>
      <td>Realizado</td>
      <td>Salgado Filho</td>
      <td>Porto Alegre</td>
      <td>...</td>
      <td>Miami</td>
      <td>N/I</td>
      <td>Estados Unidos</td>
      <td>-80.287046</td>
      <td>25.795865</td>
      <td>-51.175381</td>
      <td>-29.993473</td>
      <td>-1450</td>
      <td>-1489</td>
      <td>Adiantado</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAL - 203</td>
      <td>AMERICAN AIRLINES INC</td>
      <td>Internacional</td>
      <td>2016-01-30 12:13:00+00:00</td>
      <td>2016-01-30 12:13:00+00:00</td>
      <td>2016-01-30 21:30:00+00:00</td>
      <td>2016-01-30 21:30:00+00:00</td>
      <td>Realizado</td>
      <td>Salgado Filho</td>
      <td>Porto Alegre</td>
      <td>...</td>
      <td>Miami</td>
      <td>N/I</td>
      <td>Estados Unidos</td>
      <td>-80.287046</td>
      <td>25.795865</td>
      <td>-51.175381</td>
      <td>-29.993473</td>
      <td>0</td>
      <td>0</td>
      <td>Pontual</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AAL - 203</td>
      <td>AMERICAN AIRLINES INC</td>
      <td>Internacional</td>
      <td>2016-01-03 23:05:00+00:00</td>
      <td>2016-01-03 23:05:00+00:00</td>
      <td>2016-01-04 07:50:00+00:00</td>
      <td>2016-01-04 07:50:00+00:00</td>
      <td>Realizado</td>
      <td>Miami</td>
      <td>Miami</td>
      <td>...</td>
      <td>Sao Jose Dos Pinhais</td>
      <td>PR</td>
      <td>Brasil</td>
      <td>-49.172481</td>
      <td>-25.532713</td>
      <td>-80.287046</td>
      <td>25.795865</td>
      <td>0</td>
      <td>0</td>
      <td>Pontual</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AAL - 203</td>
      <td>AMERICAN AIRLINES INC</td>
      <td>Internacional</td>
      <td>2016-01-05 23:05:00+00:00</td>
      <td>2016-01-05 23:35:00+00:00</td>
      <td>2016-01-06 07:50:00+00:00</td>
      <td>2016-01-06 08:35:00+00:00</td>
      <td>Realizado</td>
      <td>Miami</td>
      <td>Miami</td>
      <td>...</td>
      <td>Sao Jose Dos Pinhais</td>
      <td>PR</td>
      <td>Brasil</td>
      <td>-49.172481</td>
      <td>-25.532713</td>
      <td>-80.287046</td>
      <td>25.795865</td>
      <td>30</td>
      <td>45</td>
      <td>Atrasado</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AAL - 203</td>
      <td>AMERICAN AIRLINES INC</td>
      <td>Internacional</td>
      <td>2016-01-18 12:13:00+00:00</td>
      <td>2016-01-18 13:09:00+00:00</td>
      <td>2016-01-18 21:30:00+00:00</td>
      <td>2016-01-18 22:24:00+00:00</td>
      <td>Realizado</td>
      <td>Salgado Filho</td>
      <td>Porto Alegre</td>
      <td>...</td>
      <td>Miami</td>
      <td>N/I</td>
      <td>Estados Unidos</td>
      <td>-80.287046</td>
      <td>25.795865</td>
      <td>-51.175381</td>
      <td>-29.993473</td>
      <td>56</td>
      <td>54</td>
      <td>Atrasado</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AAL - 203</td>
      <td>AMERICAN AIRLINES INC</td>
      <td>Internacional</td>
      <td>2016-01-22 23:05:00+00:00</td>
      <td>2016-01-22 23:05:00+00:00</td>
      <td>2016-01-23 07:50:00+00:00</td>
      <td>2016-01-23 07:50:00+00:00</td>
      <td>Realizado</td>
      <td>Miami</td>
      <td>Miami</td>
      <td>...</td>
      <td>Sao Jose Dos Pinhais</td>
      <td>PR</td>
      <td>Brasil</td>
      <td>-49.172481</td>
      <td>-25.532713</td>
      <td>-80.287046</td>
      <td>25.795865</td>
      <td>0</td>
      <td>0</td>
      <td>Pontual</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AAL - 203</td>
      <td>AMERICAN AIRLINES INC</td>
      <td>Internacional</td>
      <td>2016-01-15 23:05:00+00:00</td>
      <td>2016-01-15 23:55:00+00:00</td>
      <td>2016-01-16 07:50:00+00:00</td>
      <td>2016-01-16 08:28:00+00:00</td>
      <td>Realizado</td>
      <td>Miami</td>
      <td>Miami</td>
      <td>...</td>
      <td>Sao Jose Dos Pinhais</td>
      <td>PR</td>
      <td>Brasil</td>
      <td>-49.172481</td>
      <td>-25.532713</td>
      <td>-80.287046</td>
      <td>25.795865</td>
      <td>50</td>
      <td>38</td>
      <td>Atrasado</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 23 columns</p>
</div>



<p>
    Pronto, já temos nosso Dataset importado num dataframe com todas as informações que precisamos... Na verdade veremos futuramente se precisaremos mesmo de todas essas informações... <br>
    Seria interessante excluir colunas como Departure_Real e Departure_Delay pois essas informações tornam obvio saber se o voo atrasou ou não... Discutiremos isso mais pra frente <br>
    Nosso Próximo Passo agora vai ser extratificar o dataframe em outros dois dataframes um pra treinamento da rede neural e um para validação da nossa rede, cada um destes dataframes vai ser ainda separado em duas Colunas uma com todas as informações e outra somente com a nossa classe (ArrivalStatus)<br>
    Extratificar significa que separaremos os dados na proporção que eles tem de valores da nossa classe objetivo, por exemplo, se temos 50% de Voos Pontuais, 40% de atrasos e 10% de adiantamentos os nosso dois dataframes gerados terão essas proporções.
</p>


```python
df_time.shape

```




    (2253323, 23)



<h3>7. Estratificando nosso dataframe</h3>



```python
train, test = train_test_split(df_time, test_size=0.3, stratify=df_time['ArrivalStatus'])
```


```python

```


```python
atrasados = train.groupby('ArrivalStatus').get_group('Atrasado').count()[0]
adiantados = train.groupby('ArrivalStatus').get_group('Adiantado').count()[0]
pontuais = train.groupby('ArrivalStatus').get_group('Pontual').count()[0]
total = atrasados + adiantados + pontuais
colunas = ['Situation','%']
train_info = [[atrasados/total*100],[adiantados/total*100],[pontuais/total*100]]


pd.DataFrame(train_info, index=['Atrasados','Adiantados','Pontuais'], columns = ['Percent'])

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Atrasados</th>
      <td>15.508969</td>
    </tr>
    <tr>
      <th>Adiantados</th>
      <td>18.276501</td>
    </tr>
    <tr>
      <th>Pontuais</th>
      <td>66.214530</td>
    </tr>
  </tbody>
</table>
</div>




```python
atrasados = test.groupby('ArrivalStatus').get_group('Atrasado').count()[0]
adiantados = test.groupby('ArrivalStatus').get_group('Adiantado').count()[0]
pontuais = test.groupby('ArrivalStatus').get_group('Pontual').count()[0]
total = atrasados + adiantados + pontuais
colunas = ['Situation','%']
test_info = [[atrasados/total*100],[adiantados/total*100],[pontuais/total*100]]
pd.DataFrame(test_info, index=['Atrasados','Adiantados','Pontuais'], columns = ['Percent'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Atrasados</th>
      <td>15.508945</td>
    </tr>
    <tr>
      <th>Adiantados</th>
      <td>18.276560</td>
    </tr>
    <tr>
      <th>Pontuais</th>
      <td>66.214495</td>
    </tr>
  </tbody>
</table>
</div>



<p>Dessa forma temos em ambos os DataFrames aproximadamente 17% de Voos Adiantados, 16% de Voos Atrasados e 68% de Voos Pontuais</p>

<h3>8. Separando as perguntas das respostas</h3>


```python
train.shape
```




    (1577326, 23)




```python
X_train = train.iloc[:,:22]
Y_train = train.iloc[:,22:]
X_test = test.iloc[:,:22]
Y_test = test.iloc[:,22:]
```


```python
X_train.shape
```




    (1577326, 22)




```python
Y_train.shape
```




    (1577326, 1)




```python
X_test.shape
```




    (675997, 22)




```python
Y_test.shape
```




    (675997, 1)



<p>O Algoritmo que utilizaremos só aceita variáveis numéricas na origem, dessa forma, o jeito mais fácil, só pra testar o modelo, foi excluir todas,é claro que isso colocou a eficiência da nossa rede lá embaixo... 
O que precisamos fazer agora é:</p>
    <ul>
        <h5><li>Definir quais  variáveis serão</li></h5>
        <p>É difícil prever a diferença que cada variável faria, mas podemos começar definindo as mais óbvias como data e cia aérea por exemplo...</p>
    <h5><li>Transformar as variáveis em conteúdo processável</li></h5>
        <p>Definidas quais variáveis usaremos devemos ver a melhor técnica para transformá-la em conteúdo que pode ser processado pela RN (ou seja, números)</p>
            <p>Conheco duas, uma é basicamente pegar os caracteres e transformar em números, porém para esta técnica os dados devem possuir algum valor semântico quando transformados em números (funciona para data e hora por exemplo)</p>
            <p>A outra técnica é para dados que não tem valor semântico quando transformados em algarismos (como nomes, por exemplo). Essa técnica é chamada de Dummies, que é bassicamente montar uma tabela onde as colunas teriam os nomes das Cias Aéreas, por exemplo, as linhas seriam os voos e a célula (LinhaXColuna) seria preenchida com 1 se o voo pertence àquela cia aérea e com 0 caso não pertença. Isso impacta diretamente no problema seguinte: </p>         
    <li> <h5> Definir a arquitetura da rede</h5> </li>
        <p>Depois de sabermos exatamente o tamanho da nossa entrada (quantas colunas terão nas nossas tabelas X) precisamos definir quantas camadas e quantos neurônios em cada camadas teremos.<br>
    Existem alguns estudos sobre o assunto, mas é um ponto que não é uma unânimidade na área. Podemos tomar como base <a href="http://dstath.users.uth.gr/papers/IJRS2009_Stathakis.pdf">este artigo</a></p>
    </ul>



```python
#Mes
#X_Train_mes = pd.DataFrame(X_train[['Departure_Estimate']])
#X_Train_mes['Departure_Estimate'] = X_Train_mes['Departure_Estimate'].dt.month
#look_up = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
#X_Train_mes['Departure_Estimate'] = X_Train_mes['Departure_Estimate'].apply(lambda x: look_up[x])
#X_Train_mes = pd.get_dummies(X_Train_mes)
#X_Train_mes.head()
```


```python
#Pegando os valores relevantes
New_X_train = X_train[['Destination_Long', 'Destination_Lat', 'Origin_Long', 'Origin_Lat','Departure_Delays']]
#Transformando Valores relevantes em dados calculáveis
############## Treino
#Linha aérea
Dummies_X_train = X_train[['Airline']]
Dummies_X_train = pd.get_dummies(Dummies_X_train)
#Dia da Semana
X_data = X_train[['Departure_Estimate']]
X_data['Dia_Semana'] = X_data['Departure_Estimate'].dt.weekday_name

#Mes
X_Train_mes = pd.DataFrame(X_train[['Departure_Estimate']])
X_Train_mes['Departure_Estimate'] = X_Train_mes['Departure_Estimate'].dt.month
look_up = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
X_Train_mes['Departure_Estimate'] = X_Train_mes['Departure_Estimate'].apply(lambda x: look_up[x])
X_Train_mes = pd.get_dummies(X_Train_mes)

#Hora
X_hora = pd.DataFrame(X_data['Departure_Estimate'].dt.hour)
X_data = X_data['Dia_Semana']
X_hora.columns = ['Hora']
X_hora = X_hora['Hora'].apply(str)
X_hora = pd.get_dummies(X_hora)
Dummies_X_Data = pd.get_dummies(X_data)
X_train = pd.concat([New_X_train, Dummies_X_train, Dummies_X_Data, X_hora,X_Train_mes], axis=1)
############## Teste
New_X_test = X_test[['Destination_Long', 'Destination_Lat', 'Origin_Long', 'Origin_Lat','Departure_Delays']]
Dummies_X_test = X_test[['Airline']]
Dummies_X_test = pd.get_dummies(Dummies_X_test)
#Dia da Semana
X_datat = X_test[['Departure_Estimate']]
X_datat['Dia_Semana'] = X_datat['Departure_Estimate'].dt.weekday_name

#Mes
X_test_mes = pd.DataFrame(X_test[['Departure_Estimate']])
X_test_mes['Departure_Estimate'] = X_test_mes['Departure_Estimate'].dt.month
#look_up = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
X_test_mes['Departure_Estimate'] = X_test_mes['Departure_Estimate'].apply(lambda x: look_up[x])
X_test_mes = pd.get_dummies(X_test_mes)












#Hora
X_horat = pd.DataFrame(X_datat['Departure_Estimate'].dt.hour)

X_datat = X_datat['Dia_Semana']
X_horat.columns = ['Hora']
X_horat = X_horat['Hora'].apply(str)
X_horat = pd.get_dummies(X_horat)
Dummies_X_Datat = pd.get_dummies(X_datat)
X_test = pd.concat([New_X_test, Dummies_X_test, Dummies_X_Datat, X_horat,X_test_mes], axis=1)


X_test_mes.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Departure_Estimate_Apr</th>
      <th>Departure_Estimate_Aug</th>
      <th>Departure_Estimate_Dec</th>
      <th>Departure_Estimate_Feb</th>
      <th>Departure_Estimate_Jan</th>
      <th>Departure_Estimate_Jul</th>
      <th>Departure_Estimate_Jun</th>
      <th>Departure_Estimate_Mar</th>
      <th>Departure_Estimate_May</th>
      <th>Departure_Estimate_Nov</th>
      <th>Departure_Estimate_Oct</th>
      <th>Departure_Estimate_Sep</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1687115</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1783207</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2154287</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2074967</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>535615</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#X_test_mes['mes'] = pd.DataFrame(X_datat['Departure_Estimate'].dt.month)
#X_test_mes.head()
#X_test_mes['Departure_Estimate'] = X_test_mes.apply(lambda x: look_up[x])
```


```python
X_train.shape



```




    (1577326, 103)




```python
X_test.shape

```




    (675997, 103)




```python

X_train.dropna(how='any',inplace=True)
X_train.shape
```




    (1577326, 103)




```python

X_test.dropna(how='any',inplace=True)
X_test.shape
```




    (675997, 103)



<h3>9. Vendo como ficou nosso DataFrame aparado</h3>


```python

```


```python
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Destination_Long</th>
      <th>Destination_Lat</th>
      <th>Origin_Long</th>
      <th>Origin_Lat</th>
      <th>Departure_Delays</th>
      <th>Airline_AEROLINEAS ARGENTINAS</th>
      <th>Airline_AEROMEXICO</th>
      <th>Airline_AIR CANADA</th>
      <th>Airline_AIR CHINA</th>
      <th>Airline_AIR EUROPA S/A</th>
      <th>...</th>
      <th>Departure_Estimate_Dec</th>
      <th>Departure_Estimate_Feb</th>
      <th>Departure_Estimate_Jan</th>
      <th>Departure_Estimate_Jul</th>
      <th>Departure_Estimate_Jun</th>
      <th>Departure_Estimate_Mar</th>
      <th>Departure_Estimate_May</th>
      <th>Departure_Estimate_Nov</th>
      <th>Departure_Estimate_Oct</th>
      <th>Departure_Estimate_Sep</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1584578</th>
      <td>-53.700874</td>
      <td>-29.707958</td>
      <td>-51.175381</td>
      <td>-29.993473</td>
      <td>46</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>485150</th>
      <td>-56.117269</td>
      <td>-15.653079</td>
      <td>-47.917235</td>
      <td>-15.869737</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1050608</th>
      <td>-48.545966</td>
      <td>-27.670118</td>
      <td>-46.656584</td>
      <td>-23.627325</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>90584</th>
      <td>-87.907321</td>
      <td>41.974162</td>
      <td>-46.478126</td>
      <td>-23.434553</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>204307</th>
      <td>-47.917235</td>
      <td>-15.869737</td>
      <td>-51.175381</td>
      <td>-29.993473</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 103 columns</p>
</div>




```python

X_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Destination_Long</th>
      <th>Destination_Lat</th>
      <th>Origin_Long</th>
      <th>Origin_Lat</th>
      <th>Departure_Delays</th>
      <th>Airline_AEROLINEAS ARGENTINAS</th>
      <th>Airline_AEROMEXICO</th>
      <th>Airline_AIR CANADA</th>
      <th>Airline_AIR CHINA</th>
      <th>Airline_AIR EUROPA S/A</th>
      <th>...</th>
      <th>Departure_Estimate_Dec</th>
      <th>Departure_Estimate_Feb</th>
      <th>Departure_Estimate_Jan</th>
      <th>Departure_Estimate_Jul</th>
      <th>Departure_Estimate_Jun</th>
      <th>Departure_Estimate_Mar</th>
      <th>Departure_Estimate_May</th>
      <th>Departure_Estimate_Nov</th>
      <th>Departure_Estimate_Oct</th>
      <th>Departure_Estimate_Sep</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1687115</th>
      <td>-46.656584</td>
      <td>-23.627325</td>
      <td>-43.965396</td>
      <td>-19.634099</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1783207</th>
      <td>-43.249423</td>
      <td>-22.813410</td>
      <td>-46.656584</td>
      <td>-23.627325</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2154287</th>
      <td>-47.917235</td>
      <td>-15.869737</td>
      <td>-34.950614</td>
      <td>-7.147060</td>
      <td>31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2074967</th>
      <td>-38.331241</td>
      <td>-12.911098</td>
      <td>-38.533097</td>
      <td>-3.777156</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>535615</th>
      <td>-57.514181</td>
      <td>-25.241513</td>
      <td>-46.478126</td>
      <td>-23.434553</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 103 columns</p>
</div>




```python

```

<h3>Aqui que a IA começa...(pode demorar para processar)</h3>
<h4><a href="http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html">Documentação</a></h4>


```python
#Definir arquitetura:

mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(9, 7, 5), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
#Magic....
mlp.fit(X_train,Y_train)
#Nostradamus mode on...
predictions = mlp.predict(X_test)

```


```python
#Plotando nosso resultado...
print(classification_report(Y_test,predictions))
```

                  precision    recall  f1-score   support
    
       Adiantado       0.93      0.91      0.92    123549
        Atrasado       0.95      0.89      0.92    104840
         Pontual       0.98      1.00      0.99    447608
    
        accuracy                           0.97    675997
       macro avg       0.95      0.93      0.94    675997
    weighted avg       0.97      0.97      0.97    675997
    
    
