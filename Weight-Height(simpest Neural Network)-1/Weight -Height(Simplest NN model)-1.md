#### Keras Models
Keras has come up with two types of in-built models; Sequential Model and an advanced Model class with functional API. The Sequential model tends to be one of the simplest models as it constitutes a linear set of layers, whereas the functional API model leads to the creation of an arbitrary network structure.


```python
import pandas as pd 
```


```python
dataset = pd.read_csv("weight-height.csv")
```


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
      <th>Gender</th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>73.847017</td>
      <td>241.893563</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>68.781904</td>
      <td>162.310473</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>74.110105</td>
      <td>212.740856</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>71.730978</td>
      <td>220.042470</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>69.881796</td>
      <td>206.349801</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>Female</td>
      <td>66.172652</td>
      <td>136.777454</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>Female</td>
      <td>67.067155</td>
      <td>170.867906</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>Female</td>
      <td>63.867992</td>
      <td>128.475319</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>Female</td>
      <td>69.034243</td>
      <td>163.852461</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>Female</td>
      <td>61.944246</td>
      <td>113.649103</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 3 columns</p>
</div>




```python
dataset.columns
```




    Index(['Gender', 'Height', 'Weight'], dtype='object')




```python
y = dataset['Weight']
```


```python
X=dataset['Height']
```

#### Keras Sequential Model
The layers within the sequential models are sequentially arranged, so it is known as Sequential API. In most of the Artificial Neural Network, the layers are sequentially arranged, such that the data flow in between layers is in a specified sequence until it hit the output layer.


```python
from keras.models import Sequential 
model = Sequential()
from keras.layers import Dense 
```

Dense is the function which tells about the number of hidden layers should be there in the neural network. Here I have created a simplest neural network model with only one single hidden layer. 


```python
model.add(Dense(activation="linear",input_shape=(1,),units=1,kernel_initializer="zeros",bias_initializer="zeros"))
```

Above add() function is used to add the hidden layer.
- activation :  it is the activation function which we are using, here I have used the *linear* 
- input_shape : it is the no. of input feature. I am using input feature i.e X here as only one feature. 
- units : it means the no. of output. So we have here only one output value as y.
- kernel_initializer : It is the initializer for the Weights. Here I have used the *Zeros* initializer. 
- bias_initializer : It is the initializer for the bias.



```python
X.shape
```




    (10000,)



summary() - this function gives summary about our model. It automatically finds the paramater (no. of weights and bias to be calculated)


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 1)                 2         
    =================================================================
    Total params: 2
    Trainable params: 2
    Non-trainable params: 0
    _________________________________________________________________
    




```python
model.get_config()
```




    {'name': 'sequential_1',
     'layers': [{'class_name': 'InputLayer',
       'config': {'batch_input_shape': (None, 1),
        'dtype': 'float32',
        'sparse': False,
        'ragged': False,
        'name': 'dense_1_input'}},
      {'class_name': 'Dense',
       'config': {'name': 'dense_1',
        'trainable': True,
        'batch_input_shape': (None, 1),
        'dtype': 'float32',
        'units': 1,
        'activation': 'linear',
        'use_bias': True,
        'kernel_initializer': {'class_name': 'Zeros', 'config': {}},
        'bias_initializer': {'class_name': 'Zeros', 'config': {}},
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'bias_constraint': None}}]}



See here weights value is 0 , first array is for the weight and second is for the bias. After we fit our model then the value of weight and bias will gonna be change. 


```python
model.get_weights()
```




    [array([[0.]], dtype=float32), array([0.], dtype=float32)]




```python

```


```python
model.compile(loss='mean_squared_error',optimizer=Adam())
```


```python
from keras.optimizers import Adam
```


```python
model.fit(X,y)
```

    313/313 [==============================] - 0s 1ms/step - loss: 23866.6562
    




    <tensorflow.python.keras.callbacks.History at 0x12450d3e488>




```python
model.get_weights()
```




    [array([[0.30447775]], dtype=float32), array([0.30443692], dtype=float32)]




```python
model.predict([78])
```




    array([[24.053701]], dtype=float32)



We can save this model, and can give to our team mates. So that they can predict throught the model which I have created.


```python
model.save("my_model.h5")
```

So after saving this model. To use this model for the prediction we have to write the following code--> 

*from kernas.models import load_model*

*model = load_model("my_model.h5")*

*model.predict([78])*


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
