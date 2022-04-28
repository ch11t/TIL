# 머신러닝 day 8

<div><br class="Apple-interchange-newline">Perceptron</div>

```
from sklearn.linear_model import Perceptron
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,0,0,1]
p=Perceptron(tol=1e-3,random_state=10)
p.fit(X,y)
p.predict(X)
```

Out[1]:

```
array([0, 0, 0, 1])
```

In [2]:

```
#y=wx-b
#뉴런의 계단 함수
def n_f(in_data):
    global w
    global b
    at_f=b
    for i in range(2):
        at_f+=w[i]*in_data[i]
    if at_f >=0.0:
        return 1
    else:
        return 0
```

In [3]:

```
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,0,0,1]
w = [0.0,0.0]#가중치
b = 0.0#입계값
n_f(X[0]),n_f(X[1]),n_f(X[2]),n_f(X[3])
```

Out[3]:

```
(1, 1, 1, 1)
```

In [4]:

```
def t_f(X,y,l_r,epch):
    global w
    global b
    for en in range(epch):
        sum_e=0.0
        for r,t in zip(X,y):
            at=n_f(r)#예측값
            err=t - at #오차
            b = b+l_r*err
            sum_e += err**2
            for i in range(2):
                w[i] = w[i]+l_r*err*r[i]
            print(w,b)
        print(f'에포그={en} 학습률={l_r} 에러={sum_e}')
    return w
#data준비
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,1,1,1]
w = [0.0,0.0]#가중치
b = 0.0#입계값
l_r=0.1#학습률
e=5#반복수
w = t_f(X,y,l_r,e)


[0.0, 0.0] -0.1
[0.0, 0.1] 0.0
[0.0, 0.1] 0.0
[0.0, 0.1] 0.0
에포그=0 학습률=0.1 에러=2.0
[0.0, 0.1] -0.1
[0.0, 0.1] -0.1
[0.1, 0.1] 0.0
[0.1, 0.1] 0.0
에포그=1 학습률=0.1 에러=2.0
[0.1, 0.1] -0.1
[0.1, 0.1] -0.1
[0.1, 0.1] -0.1
[0.1, 0.1] -0.1
에포그=2 학습률=0.1 에러=1.0
[0.1, 0.1] -0.1
[0.1, 0.1] -0.1
[0.1, 0.1] -0.1
[0.1, 0.1] -0.1
에포그=3 학습률=0.1 에러=0.0
[0.1, 0.1] -0.1
[0.1, 0.1] -0.1
[0.1, 0.1] -0.1
[0.1, 0.1] -0.1
에포그=4 학습률=0.1 에러=0.0
```

In [5]:

```
#y=w0x0+w1x1-b
#y=0.2x0+0.1x1-0.20000000000004
n_f(X[0]),n_f(X[1]),n_f(X[2]),n_f(X[3])
```

Out[5]:

```
(0, 1, 1, 1)
```

In [6]:

```
X = [[0,0],[0,1],[1,0],[1,1]]
```

In [7]:

```
def AND(X):
    and_w = [0.2, 0.1]
    and_b = -0.20000000000000004
    at_f=and_b
    for i in range(2):
        at_f+=and_w[i]*X[i]
    if at_f >=0.0:
        return 1
    else:
        return 0
```

In [8]:

```
def OR(X):
    or_w = [0.1, 0.1]
    or_b = -0.1
    at_f=or_b
    for i in range(2):
        at_f+=or_w[i]*X[i]
    if at_f >=0.0:
        return 1
    else:
        return 0
```

In [9]:

```
AND(X[0]),AND(X[1]),AND(X[2]),AND(X[3])
```

Out[9]:

```
(0, 0, 0, 1)
```

In [10]:

```
OR(X[0]),OR(X[1]),OR(X[2]),OR(X[3])
```

Out[10]:

```
(0, 1, 1, 1)
```

In [12]:

```
def XOR(X):
    o_1 = not AND(X)
    o_2 = OR(X)
    return AND([o_1,o_2])

XOR(X[0]),XOR(X[1]),XOR(X[2]),XOR(X[3])
```

Out[12]:

```
(0, 1, 1, 0)
```

---

---



```
import numpy as np
def actf(x):
    return 1/(1+np.exp(-x))
def d_actf(x):
    return x*(1-x)
```

In [2]:

```
#y=wx+b
w=np.array([[1,2,3],
            [3,4,5]])
x=np.array([[4,5],
            [6,7],
           [8,9]])
w.dot(x)
```

Out[2]:

```
array([[40, 46],
       [76, 88]])
```

In [3]:

```
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0],[1],[1],[0]])
in_n = 3
h_n = 6
out_n = 1
np.random.seed(5)
#-1~1
w0=2*np.random.random((in_n,h_n)) -1
w1=2*np.random.random((h_n,out_n)) -1
```

In [4]:

```
for i in range(10000):
    l0=X
    #순전파
    #in*W0
    net1 = np.dot(l0,w0)
    l1=actf(net1)
    l1[:,-1] =1
    net2 = np.dot(l1,w1)
    l2=actf(net2)#결과
    #역전파
    l2_e= l2-y#출력오차 
    l2_d=l2_e*d_actf(l2)#미분(출력단의 델타값)
    
    l1_e = np.dot(l2_d,w1.T)#은닉오차 
    l1_d=l1_e*d_actf(l1)#미분(은닉단의 델타값)
    #가중치 변화
    w1 +=-0.2*np.dot(l1.T,l2_d)
    w0 +=-0.2*np.dot(l0.T,l1_d)
    if i==10:
        print(l2)
        print()
    if i==100:
        print(l2)
        print()
    if i==1000:
        print(l2)
        print()
print(l2)

[[0.44867697]
 [0.48397079]
 [0.41941189]
 [0.45527361]]

[[0.49810382]
 [0.52553137]
 [0.47556608]
 [0.50256195]]

[[0.46051417]
 [0.52154431]
 [0.50562375]
 [0.52060927]]

[[0.02391914]
 [0.9757925 ]
 [0.97343127]
 [0.03041428]]
```

In [5]:

```
y
```

Out[5]:

```
array([[0],
       [1],
       [1],
       [0]])
```

---

---

```
import numpy as np
def actf(x):
    return 1/(1+np.exp(-x))
def d_actf(x):
    return x*(1-x)
```

In [2]:

```
import numpy as np
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0],[1],[1],[0]])
in_n = 3
h_n = 6
out_n = 1
np.random.seed(5)
#-1~1
w0=2*np.random.random((in_n,h_n)) -1
w1=2*np.random.random((h_n,out_n)) -1
X.shape ,w0.shape ,X.dot(w0).shape ,w1.shape,X.dot(w0).dot(w1).shape
```

Out[2]:

```
((4, 3), (3, 6), (4, 6), (6, 1), (4, 1))
```

In [10]:

```
end=X.dot(w0).dot(w1)
(end-y).shape,w1.T.shape,(end-y).dot(w1.T).shape
```

Out[10]:

```
((4, 1), (1, 6), (4, 6))
```

In [6]:

```

```

Out[6]:

```
(6, 1)
```

---

---

<div><br class="Apple-interchange-newline">from</div>

```
from tensorflow import keras
import numpy as np
```

In [2]:

```
(t_x,t_y),(tt_x,tt_y) = keras.datasets.fashion_mnist.load_data()
```

In [3]:

```
t_x.shape
```

Out[3]:

```
(60000, 28, 28)
```

In [4]:

```
s_t_x= t_x/255.0
s_t_x= s_t_x.reshape(-1,28*28)
s_t_x.shape
```

Out[4]:

```
(60000, 784)
```

In [5]:

```
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
sc = SGDClassifier(loss='log',max_iter=5,random_state=42)
scr=cross_validate(sc,s_t_x,t_y,n_jobs=-1)
np.mean(scr['test_score'])
```

Out[5]:

```
0.8192833333333333
```

In [6]:

```
import tensorflow as tf
```

In [7]:

```
from sklearn.model_selection import train_test_split
t_x,v_x,t_y,v_y=train_test_split(s_t_x,t_y,test_size=0.2,random_state=42)
```

Out[7]:

```
(48000,)
```

In [8]:

```
dense = keras.layers.Dense(10,activation='softmax',input_shape=(784,))
```

In [9]:

```
model = keras.Sequential(dense)
model.compile(loss='sparse_categorical_crossentropy',metrics='accuracy')
```

In [15]:

```
model.fit(t_x,t_y,epochs=10)
Epoch 1/10
1500/1500 [==============================] - 1s 680us/step - loss: 0.4308 - accuracy: 0.8572
Epoch 2/10
1500/1500 [==============================] - 1s 742us/step - loss: 0.4280 - accuracy: 0.8599
Epoch 3/10
1500/1500 [==============================] - 1s 734us/step - loss: 0.4240 - accuracy: 0.8605
Epoch 4/10
1500/1500 [==============================] - 1s 726us/step - loss: 0.4220 - accuracy: 0.8609
Epoch 5/10
1500/1500 [==============================] - 1s 740us/step - loss: 0.4199 - accuracy: 0.8626
Epoch 6/10
1500/1500 [==============================] - 1s 722us/step - loss: 0.4180 - accuracy: 0.8628
Epoch 7/10
1500/1500 [==============================] - 1s 745us/step - loss: 0.4170 - accuracy: 0.8636
Epoch 8/10
1500/1500 [==============================] - 1s 762us/step - loss: 0.4146 - accuracy: 0.8645
Epoch 9/10
1500/1500 [==============================] - 1s 739us/step - loss: 0.4133 - accuracy: 0.8658
Epoch 10/10
1500/1500 [==============================] - 1s 749us/step - loss: 0.4133 - accuracy: 0.8645
```

Out[15]:

```
<keras.callbacks.History at 0x231cfed6e20>
```

In [16]:

```
model.evaluate(v_x,v_y)
375/375 [==============================] - 0s 549us/step - loss: 0.4439 - accuracy: 0.8583
```

Out[16]:

```
[0.44394198060035706, 0.8582500219345093]
```

---

---

<div><br class="Apple-interchange-newline">import</div>

```
import tensorflow as tf
from tensorflow import keras
import numpy as np
(t_x,t_y),(tt_x,tt_y) = keras.datasets.fashion_mnist.load_data()
s_t_x= t_x/255.0
s_t_x= s_t_x.reshape(-1,28*28)
from sklearn.model_selection import train_test_split
t_x,v_x,t_y,v_y=train_test_split(s_t_x,t_y,test_size=0.2,random_state=42)
t_x.shape
```

Out[1]:

```
(48000, 784)
```

In [2]:

```
dense1 = keras.layers.Dense(100,activation='sigmoid',input_shape=(784,))
dense2 = keras.layers.Dense(10,activation='softmax')
model=keras.Sequential([dense1,dense2])
```

In [3]:

```
model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 100)               78500     
                                                                 
 dense_1 (Dense)             (None, 10)                1010      
                                                                 
=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________
```

In [4]:

```
model=keras.Sequential([ 
    keras.layers.Dense(100,activation='sigmoid',input_shape=(784,),name='hidden'),
    keras.layers.Dense(10,activation='softmax',name='output')
],name ='ck')
model.summary()
Model: "ck"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 hidden (Dense)              (None, 100)               78500     
                                                                 
 output (Dense)              (None, 10)                1010      
                                                                 
=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________
```

In [5]:

```
model=keras.Sequential(name='ck')
model.add(keras.layers.Dense(100,activation='sigmoid',input_shape=(784,),name='hidden'))
model.add(keras.layers.Dense(10,activation='softmax',name='output'))
model.summary()
Model: "ck"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 hidden (Dense)              (None, 100)               78500     
                                                                 
 output (Dense)              (None, 10)                1010      
                                                                 
=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________
```

In [6]:

```
model.compile(loss='sparse_categorical_crossentropy',metrics='accuracy')
model.fit(t_x,t_y,epochs=5)
Epoch 1/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.5615 - accuracy: 0.8077
Epoch 2/5
1500/1500 [==============================] - 2s 1000us/step - loss: 0.4063 - accuracy: 0.8536
Epoch 3/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.3725 - accuracy: 0.8655
Epoch 4/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.3498 - accuracy: 0.8725
Epoch 5/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.3324 - accuracy: 0.8805
```

Out[6]:

```
<keras.callbacks.History at 0x1a7546d7fd0>
```

In [7]:

```
model=keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100,activation='relu',name='hidden'))
model.add(keras.layers.Dense(10,activation='softmax',name='output'))
model.summary()
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 784)               0         
                                                                 
 hidden (Dense)              (None, 100)               78500     
                                                                 
 output (Dense)              (None, 10)                1010      
                                                                 
=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________
```

In [8]:

```
(t_x,t_y),(tt_x,tt_y) = keras.datasets.fashion_mnist.load_data()
s_t_x= t_x/255.0
t_x,v_x,t_y,v_y=train_test_split(s_t_x,t_y,test_size=0.2,random_state=42)
t_x.shape
```

Out[8]:

```
(48000, 28, 28)
```

In [9]:

```
model.compile(loss='sparse_categorical_crossentropy',metrics='accuracy')
model.fit(t_x,t_y,epochs=5)
Epoch 1/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.5346 - accuracy: 0.8142
Epoch 2/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.3898 - accuracy: 0.8593
Epoch 3/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.3541 - accuracy: 0.8730
Epoch 4/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.3326 - accuracy: 0.8805
Epoch 5/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.3200 - accuracy: 0.8858
```

Out[9]:

```
<keras.callbacks.History at 0x1a77d2d35e0>
```