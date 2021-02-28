```python
import pandas as pd
```


```python
dataset = pd.read_csv("Churn_Modelling.csv")
```

Exited feature is the telling that a person left the bank. So we have to find out that why the person has left the bank.  


```python
dataset
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
      <th>RowNumber</th>
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>9996</td>
      <td>15606229</td>
      <td>Obijiaku</td>
      <td>771</td>
      <td>France</td>
      <td>Male</td>
      <td>39</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>96270.64</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>9997</td>
      <td>15569892</td>
      <td>Johnstone</td>
      <td>516</td>
      <td>France</td>
      <td>Male</td>
      <td>35</td>
      <td>10</td>
      <td>57369.61</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101699.77</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>9998</td>
      <td>15584532</td>
      <td>Liu</td>
      <td>709</td>
      <td>France</td>
      <td>Female</td>
      <td>36</td>
      <td>7</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>42085.58</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>9999</td>
      <td>15682355</td>
      <td>Sabbatini</td>
      <td>772</td>
      <td>Germany</td>
      <td>Male</td>
      <td>42</td>
      <td>3</td>
      <td>75075.31</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>92888.52</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>10000</td>
      <td>15628319</td>
      <td>Walker</td>
      <td>792</td>
      <td>France</td>
      <td>Female</td>
      <td>28</td>
      <td>4</td>
      <td>130142.79</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38190.78</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 14 columns</p>
</div>




```python
dataset.info() 
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 14 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   RowNumber        10000 non-null  int64  
     1   CustomerId       10000 non-null  int64  
     2   Surname          10000 non-null  object 
     3   CreditScore      10000 non-null  int64  
     4   Geography        10000 non-null  object 
     5   Gender           10000 non-null  object 
     6   Age              10000 non-null  int64  
     7   Tenure           10000 non-null  int64  
     8   Balance          10000 non-null  float64
     9   NumOfProducts    10000 non-null  int64  
     10  HasCrCard        10000 non-null  int64  
     11  IsActiveMember   10000 non-null  int64  
     12  EstimatedSalary  10000 non-null  float64
     13  Exited           10000 non-null  int64  
    dtypes: float64(2), int64(9), object(3)
    memory usage: 1.1+ MB
    

By using the weight, we can understood which are the important features that decides why the person left the bank. 


```python
y = dataset['Exited']

```


```python
X = dataset[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
'IsActiveMember', 'EstimatedSalary']]
```


```python
X.shape
```




    (10000, 8)




```python
dataset.columns
```




    Index(['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography',
           'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
           'IsActiveMember', 'EstimatedSalary', 'Exited'],
          dtype='object')



If you see, there are many features in the dataset, but some feature might be very imp for the prediction....here "Geography" and "Gender" are categorical data, we have to convert it to continuous data. SO that we can apply the classification. 

One hot encoding: 
Here as you can see in the
Geography reason you have you have three columns after one hotencoding i.e Germany, Spain, France. So to avoid dummy trap i have used a fn from pandas i.e
get_dummies in which i have use drop_first=True that will helps us to avoid dummy trap.





```python
Geo = pd.get_dummies(dataset['Geography'],drop_first=True)
```


```python
Geo
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
      <th>Germany</th>
      <th>Spain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 2 columns</p>
</div>



One Hot Encoding on Gender


```python
Gen = pd.get_dummies(dataset['Gender'],drop_first=True)
```


```python
Gen
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
      <th>Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>1</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>1</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>1</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 1 columns</p>
</div>




```python
pd.concat([X,Geo,Gen])
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
      <th>CreditScore</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Germany</th>
      <th>Spain</th>
      <th>Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>619.0</td>
      <td>42.0</td>
      <td>2.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>101348.88</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>608.0</td>
      <td>41.0</td>
      <td>1.0</td>
      <td>83807.86</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>112542.58</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>502.0</td>
      <td>42.0</td>
      <td>8.0</td>
      <td>159660.80</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>113931.57</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>699.0</td>
      <td>39.0</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>93826.63</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>850.0</td>
      <td>43.0</td>
      <td>2.0</td>
      <td>125510.82</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>79084.10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>30000 rows × 11 columns</p>
</div>



above you can see the there are lot of null values this is pandas perform any function in row wise, so perform it in the column wise we have to give axis=1 


```python
X=pd.concat([X,Geo,Gen],axis=1)
```


```python
X
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
      <th>CreditScore</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Germany</th>
      <th>Spain</th>
      <th>Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>619</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>608</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>502</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>699</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>850</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>771</td>
      <td>39</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>96270.64</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>516</td>
      <td>35</td>
      <td>10</td>
      <td>57369.61</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101699.77</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>709</td>
      <td>36</td>
      <td>7</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>42085.58</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>772</td>
      <td>42</td>
      <td>3</td>
      <td>75075.31</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>92888.52</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>792</td>
      <td>28</td>
      <td>4</td>
      <td>130142.79</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38190.78</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 11 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split
```


```python
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)
```


```python
X_train.shape
```




    (8000, 11)




```python
X_test.shape
```




    (2000, 11)




```python
from keras.models import  Sequential
```


```python
model = Sequential()
```


```python
from keras.layers import Dense

```




```python
model.add(Dense(units=8,activation='relu',input_dim=11))
```


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 8)                 96        
    =================================================================
    Total params: 96
    Trainable params: 96
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.add(Dense(units=6,activation='relu'))
model.summary()

```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 8)                 96        
    _________________________________________________________________
    dense_1 (Dense)              (None, 6)                 54        
    =================================================================
    Total params: 150
    Trainable params: 150
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.add(Dense(units=1,activation='sigmoid'))
```


```python
model.get_config()
```




    {'name': 'sequential',
     'layers': [{'class_name': 'InputLayer',
       'config': {'batch_input_shape': (None, 11),
        'dtype': 'float32',
        'sparse': False,
        'ragged': False,
        'name': 'dense_input'}},
      {'class_name': 'Dense',
       'config': {'name': 'dense',
        'trainable': True,
        'batch_input_shape': (None, 11),
        'dtype': 'float32',
        'units': 8,
        'activation': 'relu',
        'use_bias': True,
        'kernel_initializer': {'class_name': 'GlorotUniform',
         'config': {'seed': None}},
        'bias_initializer': {'class_name': 'Zeros', 'config': {}},
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'bias_constraint': None}},
      {'class_name': 'Dense',
       'config': {'name': 'dense_1',
        'trainable': True,
        'dtype': 'float32',
        'units': 6,
        'activation': 'relu',
        'use_bias': True,
        'kernel_initializer': {'class_name': 'GlorotUniform',
         'config': {'seed': None}},
        'bias_initializer': {'class_name': 'Zeros', 'config': {}},
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'bias_constraint': None}},
      {'class_name': 'Dense',
       'config': {'name': 'dense_2',
        'trainable': True,
        'dtype': 'float32',
        'units': 1,
        'activation': 'sigmoid',
        'use_bias': True,
        'kernel_initializer': {'class_name': 'GlorotUniform',
         'config': {'seed': None}},
        'bias_initializer': {'class_name': 'Zeros', 'config': {}},
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'bias_constraint': None}}]}




```python
from keras.optimizers import Adam
```


```python
model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate = 0.00005))
```


```python
model.fit(X_train,y_train,epochs=200)
```

    Epoch 1/200
    250/250 [==============================] - 0s 1ms/step - loss: 7462.8252
    Epoch 2/200
    250/250 [==============================] - 0s 1ms/step - loss: 5717.0796
    Epoch 3/200
    250/250 [==============================] - 0s 1ms/step - loss: 4299.4092
    Epoch 4/200
    250/250 [==============================] - 0s 1ms/step - loss: 3178.0310
    Epoch 5/200
    250/250 [==============================] - 0s 1ms/step - loss: 2332.1702
    Epoch 6/200
    250/250 [==============================] - 0s 1ms/step - loss: 1621.0200
    Epoch 7/200
    250/250 [==============================] - 0s 1ms/step - loss: 1018.0232
    Epoch 8/200
    250/250 [==============================] - 0s 2ms/step - loss: 598.3518
    Epoch 9/200
    250/250 [==============================] - 0s 1ms/step - loss: 236.9072
    Epoch 10/200
    250/250 [==============================] - 0s 1ms/step - loss: 5.3995
    Epoch 11/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.7117A: 0s - loss: 0
    Epoch 12/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6589
    Epoch 13/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.6506
    Epoch 14/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6471
    Epoch 15/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6437A: 0s - loss: 0.643
    Epoch 16/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6421
    Epoch 17/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6399
    Epoch 18/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6376
    Epoch 19/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6352
    Epoch 20/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6335
    Epoch 21/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6294
    Epoch 22/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6262
    Epoch 23/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6248
    Epoch 24/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6227
    Epoch 25/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6208
    Epoch 26/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.6175
    Epoch 27/200
    250/250 [==============================] - ETA: 0s - loss: 0.6145- ETA: 0s - loss: 0. - 1s 2ms/step - loss: 0.6156
    Epoch 28/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.6139
    Epoch 29/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.6127
    Epoch 30/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6100
    Epoch 31/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6053
    Epoch 32/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6045
    Epoch 33/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.6029
    Epoch 34/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.6002
    Epoch 35/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5981
    Epoch 36/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5962
    Epoch 37/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5929
    Epoch 38/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5924
    Epoch 39/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5898
    Epoch 40/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5881A: 0s - loss: 0
    Epoch 41/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5869
    Epoch 42/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5833
    Epoch 43/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5818
    Epoch 44/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5808
    Epoch 45/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5774
    Epoch 46/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5759
    Epoch 47/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5751
    Epoch 48/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5736
    Epoch 49/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5717
    Epoch 50/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.5702
    Epoch 51/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5687
    Epoch 52/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.5672
    Epoch 53/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.5658
    Epoch 54/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5639
    Epoch 55/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5618
    Epoch 56/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5598
    Epoch 57/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.5594
    Epoch 58/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5584
    Epoch 59/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.5587
    Epoch 60/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.5551
    Epoch 61/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5532
    Epoch 62/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5518
    Epoch 63/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5503
    Epoch 64/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5502
    Epoch 65/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5482
    Epoch 66/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5466
    Epoch 67/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5468
    Epoch 68/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5454
    Epoch 69/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5438
    Epoch 70/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5428
    Epoch 71/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5416
    Epoch 72/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5401
    Epoch 73/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5384
    Epoch 74/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5379
    Epoch 75/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5367
    Epoch 76/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5361
    Epoch 77/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5355
    Epoch 78/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5346A: 0s - loss: 0.534
    Epoch 79/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5330
    Epoch 80/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5321
    Epoch 81/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5316
    Epoch 82/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5306
    Epoch 83/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5294
    Epoch 84/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5290
    Epoch 85/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5281
    Epoch 86/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5272
    Epoch 87/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5266
    Epoch 88/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5257
    Epoch 89/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5275
    Epoch 90/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5242
    Epoch 91/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5236
    Epoch 92/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5228
    Epoch 93/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5223
    Epoch 94/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5216
    Epoch 95/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5211
    Epoch 96/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5206
    Epoch 97/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5201
    Epoch 98/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5195
    Epoch 99/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5189
    Epoch 100/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5174
    Epoch 101/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5055
    Epoch 102/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5043
    Epoch 103/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5033
    Epoch 104/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5032
    Epoch 105/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5025
    Epoch 106/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5020
    Epoch 107/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5020
    Epoch 108/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5015A: 0s - loss: 0
    Epoch 109/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5017
    Epoch 110/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5008
    Epoch 111/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5010
    Epoch 112/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5011
    Epoch 113/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5001
    Epoch 114/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5009
    Epoch 115/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5009
    Epoch 116/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5007
    Epoch 117/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5011
    Epoch 118/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5003
    Epoch 119/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5003
    Epoch 120/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5007
    Epoch 121/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5003
    Epoch 122/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5008
    Epoch 123/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5007
    Epoch 124/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5000
    Epoch 125/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5006
    Epoch 126/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5003
    Epoch 127/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4997
    Epoch 128/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5003
    Epoch 129/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5004
    Epoch 130/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5001
    Epoch 131/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5004
    Epoch 132/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4999
    Epoch 133/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4999
    Epoch 134/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5001
    Epoch 135/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5000
    Epoch 136/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5008
    Epoch 137/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5005
    Epoch 138/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5000
    Epoch 139/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5004
    Epoch 140/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5004
    Epoch 141/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5000
    Epoch 142/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5001
    Epoch 143/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4998
    Epoch 144/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5001
    Epoch 145/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4999
    Epoch 146/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5007
    Epoch 147/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5003
    Epoch 148/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.5000
    Epoch 149/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.5001
    Epoch 150/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4999
    Epoch 151/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5003
    Epoch 152/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5005
    Epoch 153/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.5005
    Epoch 154/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.5000
    Epoch 155/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5001
    Epoch 156/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.4999
    Epoch 157/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4999
    Epoch 158/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5003
    Epoch 159/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.4995
    Epoch 160/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5001
    Epoch 161/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4999
    Epoch 162/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4999
    Epoch 163/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5006
    Epoch 164/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4997
    Epoch 165/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5002
    Epoch 166/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5001
    Epoch 167/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4996
    Epoch 168/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4998
    Epoch 169/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4996
    Epoch 170/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4998A: 0s - loss: 0.49
    Epoch 171/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5002
    Epoch 172/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4998
    Epoch 173/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5003
    Epoch 174/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5001
    Epoch 175/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5000
    Epoch 176/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5000
    Epoch 177/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5002
    Epoch 178/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4998
    Epoch 179/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4996
    Epoch 180/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4999
    Epoch 181/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5001
    Epoch 182/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5000
    Epoch 183/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4999
    Epoch 184/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4998
    Epoch 185/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5017
    Epoch 186/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4998
    Epoch 187/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4994
    Epoch 188/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4996
    Epoch 189/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5000
    Epoch 190/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4995
    Epoch 191/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4997
    Epoch 192/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5000
    Epoch 193/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4997
    Epoch 194/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4994
    Epoch 195/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4995
    Epoch 196/200
    250/250 [==============================] - 0s 2ms/step - loss: 0.4994
    Epoch 197/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4997
    Epoch 198/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5003
    Epoch 199/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.4996
    Epoch 200/200
    250/250 [==============================] - 0s 1ms/step - loss: 0.5000
    




    <tensorflow.python.keras.callbacks.History at 0x21624a12a48>




```python
l =  pd.DataFrame(model.history.history)
```


```python
l.plot()
```


```python

```
