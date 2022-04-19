# 머신러닝 day 1



## <1>

<div><br class="Apple-interchange-newline">import numpy as np from numpy.linalg import inv a=np.array([2,1]) print(a) # 벡터 선 print(np.linalg.norm(a)) 행렬1=np.array([[1,2],[3,4]]) # 행렬 면 print(행렬1)</div>

```
import numpy as np
from numpy.linalg import inv
a=np.array([2,1])
print(a) # 벡터 선
print(np.linalg.norm(a))
행렬1=np.array([[1,2],[3,4]]) # 행렬 면
print(행렬1)
[2 1]
2.23606797749979
[[1 2]
 [3 4]]
```

In [2]:

```
m1=np.array([[1,2],[3,4]])
m2=np.array([[1,2],[3,4]])
print(m1+m2)
[[2 4]
 [6 8]]
```

In [13]:

```
i_m1=inv(m1)
print(m1)
print(i_m1)
[[1 2]
 [3 4]]
[[-2.   1. ]
 [ 1.5 -0.5]]
```

In [15]:

```
print(np.dot(m1,m2)) # 내적
[[ 7 10]
 [15 22]]
```

In [16]:

```
x=np.array([1,2,3])
y=np.array([4,5,6])
print(np.dot(x,y))
32
```

In [23]:

```
m_x=np.array([[1,2,3]]) # 열과 행이 맞아야한다 하나라도 없으면 오류
m_y=np.array([[4,5,6]])
print(m_x.shape)
print(m_y.shape)
m_y=m_y.T # 행열의 전치
print(m_y)
print(m_y.shape)
print("(n,m).(m,n)")
(1, 3)
(1, 3)
[[4]
 [5]
 [6]]
(3, 1)
(n,m).(m,n)
```

In [24]:

```
print(np.dot(m_x,m_y))
print(m_x.dot(m_y))
[[32]]
[[32]]
```

In [25]:

```
a1=np.array([[1,2,3],[-1,-2,-3]])
a2=np.array([[4,-4],[5,-5],[6,-6]])
y=a1.dot(a2)
print(y)
[[ 32 -32]
 [-32  32]]
```

In [26]:

```
k=np.arange(1,10).reshape((3,3))
print(k)
print(inv(k))
[[1 2 3]
 [4 5 6]
 [7 8 9]]
[[-4.50359963e+15  9.00719925e+15 -4.50359963e+15]
 [ 9.00719925e+15 -1.80143985e+16  9.00719925e+15]
 [-4.50359963e+15  9.00719925e+15 -4.50359963e+15]]
```

In [28]:

```
# Y=WX
# W=YX^-1
X=np.array([[2,-1],[1,1]])
Y=np.array([[0],[3]])
inv_X=inv(X)
print(Y.shape)
print(inv_X.shape)
(2, 1)
(2, 2)
```

In [29]:

```
#Q
x=np.array([[2,3],[1,-2]])
y=np.array([[1],[4]])
inv_x=inv(x)
print(y.shape)
print(inv_x.shape)
w=inv_x.dot(y)
print(w)
(2, 1)
(2, 2)
[[ 2.]
 [-1.]]
```

In [ ]:

```

```

In [5]:

```
import matplotlib.pyplot as plt
```

In [6]:

```
def f(w0,w1):
    return w0**2+2*w0*w1+3
def df_dw0(w0,w1):
    return 2*w0+2*w1
def df_dw1(w0,w1):
    return 2*w0+0*w1
```

In [9]:

```
w_r=2
dw=0.25
w0 = np.arange(-w_r,w_r+dw,dw) # 음수부터 양수까지 dw 간격으로
w1 = np.arange(-w_r,w_r+dw,dw) # 음수부터 양수까지 dw 간격으로
wn= w0.shape[0]
ww0,ww1=np.meshgrid(w0,w1)
ff=np.zeros((len(w0),len(w1)))
dff_dw0= np.zeros((len(w0),len(w1)))
dff_dw1= np.zeros((len(w0),len(w1)))
for i0 in range(wn):
    for i1 in range(wn):
        ff[i1,i0] =  f(w0[i0],w1[i1])
        dff_dw0[i1, i0] = df_dw0(w0[i0],w1[i1]) 
        dff_dw1[i1, i0] = df_dw1(w0[i0],w1[i1])

```



## <2>

```
# 1. 데이터 수집
길이=[25.4,26.5,27.5,28.4,29.0,29.2,30.1,30.5,31.4,31.2]
무게=[243,290,340,363,430,450,500,394,450,500]
print(len(길이),len(무게))
10 10
```

In [3]:

```
import matplotlib.pyplot as plt
plt.scatter(길이,무게)
```

Out[3]:

```
<matplotlib.collections.PathCollection at 0x24f96cd4760>
```

![img](data:image/png;base64,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)

In [4]:

```
길이1=[5.4,7.5,7.5,8.4,9.0,4.2,3.1,7.5,1.4,1.2]
무게1=[43,20,30,63,30,40,50,34,40,50]
plt.scatter(길이,무게)
plt.scatter(길이1,무게1)
```

Out[4]:

```
<matplotlib.collections.PathCollection at 0x24f96df3190>
```

![img](data:image/png;base64,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)

In [19]:

```
# 2. 데이터 정리
a=길이+길이1
b=무게+무게1
data=[[a,b]for a,b in zip(길이,무게)]
print(data)
x=data
[[25.4, 243], [26.5, 290], [27.5, 340], [28.4, 363], [29.0, 430], [29.2, 450], [30.1, 500], [30.5, 394], [31.4, 450], [31.2, 500]]
```

In [20]:

```
# 3. 모델 생성
y=[0]*5+[1]*5
y
```

Out[20]:

```
[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
```

In [21]:

```
from sklearn.neighbors import KNeighborsClassifier # 가까이 있는 것들 끼리 묶음
kn = KNeighborsClassifier()
# 4. 학습
kn.fit(x,y)

```

Out[21]:

```
KNeighborsClassifier()
```

In [23]:

```
# 5. 테스트
kn.predict([[7,40],[30,400]])
```

Out[23]:

```
array([0, 1])
```



## <3>

```
from sklearn.datasets import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
d=load_digits()
sns.heatmap(d.images[0],annot=True,cbar=True)
plt.title("확인")
plt.show()
```

![img](data:image/png;base64,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)

In [4]:

```
d.images[0]
```

Out[4]:

```
array([[ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.],
       [ 0.,  0., 13., 15., 10., 15.,  5.,  0.],
       [ 0.,  3., 15.,  2.,  0., 11.,  8.,  0.],
       [ 0.,  4., 12.,  0.,  0.,  8.,  8.,  0.],
       [ 0.,  5.,  8.,  0.,  0.,  9.,  8.,  0.],
       [ 0.,  4., 11.,  0.,  1., 12.,  7.,  0.],
       [ 0.,  2., 14.,  5., 10., 12.,  0.,  0.],
       [ 0.,  0.,  6., 13., 10.,  0.,  0.,  0.]])
```

In [5]:

```
d.images[0].flatten()
```

Out[5]:

```
array([ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.,  0.,  0., 13., 15., 10.,
       15.,  5.,  0.,  0.,  3., 15.,  2.,  0., 11.,  8.,  0.,  0.,  4.,
       12.,  0.,  0.,  8.,  8.,  0.,  0.,  5.,  8.,  0.,  0.,  9.,  8.,
        0.,  0.,  4., 11.,  0.,  1., 12.,  7.,  0.,  0.,  2., 14.,  5.,
       10., 12.,  0.,  0.,  0.,  0.,  6., 13., 10.,  0.,  0.,  0.])
```

In [8]:

```
import pandas as pd
t=load_boston()
df=pd.DataFrame(t.data,columns=t.feature_names)
df
```

Out[8]:

|      |    CRIM |   ZN | INDUS | CHAS |   NOX |    RM |  AGE |    DIS |  RAD |   TAX | PTRATIO |      B | LSTAT |
| ---: | ------: | ---: | ----: | ---: | ----: | ----: | ---: | -----: | ---: | ----: | ------: | -----: | ----: |
|    0 | 0.00632 | 18.0 |  2.31 |  0.0 | 0.538 | 6.575 | 65.2 | 4.0900 |  1.0 | 296.0 |    15.3 | 396.90 |  4.98 |
|    1 | 0.02731 |  0.0 |  7.07 |  0.0 | 0.469 | 6.421 | 78.9 | 4.9671 |  2.0 | 242.0 |    17.8 | 396.90 |  9.14 |
|    2 | 0.02729 |  0.0 |  7.07 |  0.0 | 0.469 | 7.185 | 61.1 | 4.9671 |  2.0 | 242.0 |    17.8 | 392.83 |  4.03 |
|    3 | 0.03237 |  0.0 |  2.18 |  0.0 | 0.458 | 6.998 | 45.8 | 6.0622 |  3.0 | 222.0 |    18.7 | 394.63 |  2.94 |
|    4 | 0.06905 |  0.0 |  2.18 |  0.0 | 0.458 | 7.147 | 54.2 | 6.0622 |  3.0 | 222.0 |    18.7 | 396.90 |  5.33 |
|  ... |     ... |  ... |   ... |  ... |   ... |   ... |  ... |    ... |  ... |   ... |     ... |    ... |   ... |
|  501 | 0.06263 |  0.0 | 11.93 |  0.0 | 0.573 | 6.593 | 69.1 | 2.4786 |  1.0 | 273.0 |    21.0 | 391.99 |  9.67 |
|  502 | 0.04527 |  0.0 | 11.93 |  0.0 | 0.573 | 6.120 | 76.7 | 2.2875 |  1.0 | 273.0 |    21.0 | 396.90 |  9.08 |
|  503 | 0.06076 |  0.0 | 11.93 |  0.0 | 0.573 | 6.976 | 91.0 | 2.1675 |  1.0 | 273.0 |    21.0 | 396.90 |  5.64 |
|  504 | 0.10959 |  0.0 | 11.93 |  0.0 | 0.573 | 6.794 | 89.3 | 2.3889 |  1.0 | 273.0 |    21.0 | 393.45 |  6.48 |
|  505 | 0.04741 |  0.0 | 11.93 |  0.0 | 0.573 | 6.030 | 80.8 | 2.5050 |  1.0 | 273.0 |    21.0 | 396.90 |  7.88 |

506 rows × 13 columns

In [10]:

```
df['가격']=t.target
df
```

Out[10]:

|      |    CRIM |   ZN | INDUS | CHAS |   NOX |    RM |  AGE |    DIS |  RAD |   TAX | PTRATIO |      B | LSTAT | 가격 |
| ---: | ------: | ---: | ----: | ---: | ----: | ----: | ---: | -----: | ---: | ----: | ------: | -----: | ----: | ---: |
|    0 | 0.00632 | 18.0 |  2.31 |  0.0 | 0.538 | 6.575 | 65.2 | 4.0900 |  1.0 | 296.0 |    15.3 | 396.90 |  4.98 | 24.0 |
|    1 | 0.02731 |  0.0 |  7.07 |  0.0 | 0.469 | 6.421 | 78.9 | 4.9671 |  2.0 | 242.0 |    17.8 | 396.90 |  9.14 | 21.6 |
|    2 | 0.02729 |  0.0 |  7.07 |  0.0 | 0.469 | 7.185 | 61.1 | 4.9671 |  2.0 | 242.0 |    17.8 | 392.83 |  4.03 | 34.7 |
|    3 | 0.03237 |  0.0 |  2.18 |  0.0 | 0.458 | 6.998 | 45.8 | 6.0622 |  3.0 | 222.0 |    18.7 | 394.63 |  2.94 | 33.4 |
|    4 | 0.06905 |  0.0 |  2.18 |  0.0 | 0.458 | 7.147 | 54.2 | 6.0622 |  3.0 | 222.0 |    18.7 | 396.90 |  5.33 | 36.2 |
|  ... |     ... |  ... |   ... |  ... |   ... |   ... |  ... |    ... |  ... |   ... |     ... |    ... |   ... |  ... |
|  501 | 0.06263 |  0.0 | 11.93 |  0.0 | 0.573 | 6.593 | 69.1 | 2.4786 |  1.0 | 273.0 |    21.0 | 391.99 |  9.67 | 22.4 |
|  502 | 0.04527 |  0.0 | 11.93 |  0.0 | 0.573 | 6.120 | 76.7 | 2.2875 |  1.0 | 273.0 |    21.0 | 396.90 |  9.08 | 20.6 |
|  503 | 0.06076 |  0.0 | 11.93 |  0.0 | 0.573 | 6.976 | 91.0 | 2.1675 |  1.0 | 273.0 |    21.0 | 396.90 |  5.64 | 23.9 |
|  504 | 0.10959 |  0.0 | 11.93 |  0.0 | 0.573 | 6.794 | 89.3 | 2.3889 |  1.0 | 273.0 |    21.0 | 393.45 |  6.48 | 22.0 |
|  505 | 0.04741 |  0.0 | 11.93 |  0.0 | 0.573 | 6.030 | 80.8 | 2.5050 |  1.0 | 273.0 |    21.0 | 396.90 |  7.88 | 11.9 |

506 rows × 14 columns

In [12]:

```
ck=sns.pairplot(df[['가격','RM','AGE','CRIM']])
ck
```

Out[12]:

```
<seaborn.axisgrid.PairGrid at 0x24d38d8a430>
C:\Users\yoon\anaconda3\lib\site-packages\matplotlib\backends\backend_agg.py:203: RuntimeWarning: Glyph 44032 missing from current font.
  font.set_text(s, 0, flags=flags)
C:\Users\yoon\anaconda3\lib\site-packages\matplotlib\backends\backend_agg.py:203: RuntimeWarning: Glyph 44201 missing from current font.
  font.set_text(s, 0, flags=flags)
```

In [18]:

```
from  sklearn.datasets import *
t=load_boston()
x=t.data
x
```

Out[18]:

```
array([[6.3200e-03, 1.8000e+01, 2.3100e+00, ..., 1.5300e+01, 3.9690e+02,
        4.9800e+00],
       [2.7310e-02, 0.0000e+00, 7.0700e+00, ..., 1.7800e+01, 3.9690e+02,
        9.1400e+00],
       [2.7290e-02, 0.0000e+00, 7.0700e+00, ..., 1.7800e+01, 3.9283e+02,
        4.0300e+00],
       ...,
       [6.0760e-02, 0.0000e+00, 1.1930e+01, ..., 2.1000e+01, 3.9690e+02,
        5.6400e+00],
       [1.0959e-01, 0.0000e+00, 1.1930e+01, ..., 2.1000e+01, 3.9345e+02,
        6.4800e+00],
       [4.7410e-02, 0.0000e+00, 1.1930e+01, ..., 2.1000e+01, 3.9690e+02,
        7.8800e+00]])
```

In [19]:

```
y=t.target # 정답
y
```

Out[19]:

```
array([24. , 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 18.9, 15. ,
       18.9, 21.7, 20.4, 18.2, 19.9, 23.1, 17.5, 20.2, 18.2, 13.6, 19.6,
       15.2, 14.5, 15.6, 13.9, 16.6, 14.8, 18.4, 21. , 12.7, 14.5, 13.2,
       
		. . .
		
       20.6, 21.2, 19.1, 20.6, 15.2,  7. ,  8.1, 13.6, 20.1, 21.8, 24.5,
       23.1, 19.7, 18.3, 21.2, 17.5, 16.8, 22.4, 20.6, 23.9, 22. , 11.9])
```

In [21]:

```
from sklearn.linear_model import LinearRegression
m=LinearRegression() # 모델 생성
m.fit(x,y) # 학습
out_d=m.predict(t.data)
plt.scatter(y,out_d)
plt.show()
```

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXAAAAD4CAYAAAD1jb0+AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAAnRUlEQVR4nO3df4wc5Zkn8O8z7QL3kD23TQYOGgZzWWTvOo49YQ4s+f5Yk02Mjl8jCDhcOPFHJP+zJwWOm71hhc4m4sScrE1y0u0/3G60rOBYm0AGEyIBwkR7642dtTP2en2AWBIwGSzMxh424AHaM8/90V3t6up663dVV3V/PxLyTM2Pqqlhnnr7eZ/nfUVVQURE5TPU6wsgIqJ4GMCJiEqKAZyIqKQYwImISooBnIiopJblebLPf/7zunr16jxPSURUeocPH/5nVR1xH881gK9evRqHDh3K85RERKUnIu94HWcKhYiopBjAiYhKigGciKikGMCJiEqKAZyIqKRyrUIhIho0M7Nz2PXiG3hvfgGX16qY3LoGE2P1VL43AzgRUUZmZufw4LPHsNBYBADMzS/gwWePAUAqQZwpFCKijOx68Y128LYtNBax68U3Uvn+DOBERBl5b34h0vGoGMCJiDJyea0a6XhUDOBERBmZ3LoGVavScaxqVTC5dU0q358BnIgoIxNjdXx5dEXHsS+PrkitCoUBnIgoIw/NHMP+t053HNv/1mk8NHMsle/PAE5ElJGnDr4b6XhUDOBERBlZVI10PCoGcCKijFREIh2PigGciCiimdk5bJ7eh6unXsDm6X2YmZ3z/Ly7r78y0vGoGMCJiCKw2+Pn5hegON8e7xXEH5lYj81fWNVxbPMXVuGRifWpXAsDOBFRBFHa42dm5/CLEx92HPvFiQ+NI/aoGMCJiCKI0h7PtVCIiArE1AY/JNI1suZaKEREBeLVHg80SwPduXCuhUJEVCATY3U8evt6z1JAd3pk9cXegdp0PCoGcCIaKGFLAP1MjNWxZGjGcaZHfvbL056fYzoeVegALiIVEZkVkR+33l8lIi+LyJutf1emckVERBmJUgIYJEx6ZMnQcGk6HlWUEfi3AbzmeH8KwCuqeg2AV1rvExEVVppVIVkvFRtGqAAuIlcAuAnAnzsO3wbg8dbbjwOYSPXKiIhSlmZViJ0Lr9eqEAD1WhWP3r6+Y6lYyxBhTcejCrup8fcB/DGA33Ecu1RVTwKAqp4UkUu8vlBEtgPYDgCjo6Pxr5SICMl2eb+8VsWcR7COWxUyMVb3Pffnlls4c7bheTwNgc8BEbkZwClVPRznBKr6mKqOq+r4yMhInG9BRAQgeQ4777THvEfw9jseVZgR+GYAt4rIvwewHMC/EpEnALwvIpe1Rt+XATiVyhURERn45bBNI2H3iP2Oa+t49fUPYo3go0p7xO8WOAJX1QdV9QpVXQ3gGwD2qeo9APYCuLf1afcCeC6VKyIiMoiaw/YasT9zeA6TW9fgV9M3Yf/UDZkFbyD7EX/YHLiXaQB7RORbAE4AuDOVKyIiMggzonWOuIdEujZPCBqxp8k+R9ycfZBIAVxVfwrgp623fwPgK6lcBRENpKgTkpNb1+DBZ491pFGcI1p7xG1/3LTzTVprkYQRNNGZRJIROBFRbO5ga09IAjAGvKARrVeO3EtaOeheYwCnXCQp/aJ85fW7ijMhCXSOaO1rvX/3EWN6xS3vZpssMYBT5uKMtKg38vxdJW2q8bpWAeCVNKmIYEm1J4OHLB+IDOCUubgjLcpfnr8r04h5SARXT73QEey8gqDXtSrQFcSrVqWrQzKJKAE56wciVyOkzGW9qD2lJ8/fld+62s4mnYdmjnk275jSJQr4trcnEbWRKOsdeTgCp8xl3cxAwcKOGvP8XbknJE0lf08dfNfzeMXj84Fm0N4/dUPq12tfa5RXKKaHTJhcfRgcgVPmirBq2yCLMmrM+3c1MVbH/qkb8Kvpm4zra5tKARdVc///KuorFK9NH/yOR8UATpkLs2rbIPPbYCCNzQeivIzv5e/KNMo3BTv72vK81qhbpPk9fNLAFArlIstmhjLzm+QCkMoEmGl0ODe/gM3T+7rSKX6/qywrKkxNOndcW8czh+c8m3fy/v8qqJHIrW5ISdVTSkkxgBP1kN/o+ONPzyVeuGly6xrf+ugoD4WsKyr8mnTGr1qVaW162AdT1Nb4LWtH8MSBE57H0yCa0lA+jPHxcT106FBu5yMququnXvCsWw7y9vRNXcdmZucw+fRRNBz7dVlDgm3XXdk1gnULM/G3eXqfcTSZ1aRhHtwPJiC90sO07pmIHFbVcfdx5sCJUhA3Vx017+v3sZ17j3cEbwBoLCl+fPRkO1dsEqZMsGjloGnMDwDZlvplfc+YQiFKKElqwZRT9RstL6q2c9fA+ZfzppH8/EKjnSs2jQjDlAlGXQkwy67HNNM5WQbZnq8HTkT+kozgvKo+7ri2jqAis7n5BUz+8Cgmnz7aLg8MI0mZYNDXprnje5A0R81RK0uimNy6BtZQ52/TGpJCrAdORPAfwYUZkborKTZP7wsVkBuL4cL2kDSDq/M8XtcUdK1xVgLMqg0/zVFz1MqSyNxP43RKwAEwgBMlZnqZvKJqxXqZn3ZOeUnRcV6v0ruwKQm/sr08c+Rppiay3HRh14tvdD1oG4ua2kONKRSihEypBRH4vsw3TcJl0bYelF4wjZ4f2HM09CRhlqkIt7Q7Rp0doWlus5b1Q40BnCghU/eiaedxO7Viyhd7BSerIt25VI9jfvyChulj7oWl/IJ4nm34ZenuzfqhxhQKkUGUigqv1MKuF9/wfJlfG7bwwJ6jxr0a7fpg97lNx+7fcwRh2jn8gkaYzRCC8tlRUhFpVKuUobs36/w6G3loYCRZxxmI3tzh9T0qQ4LFJf+/Oa8mnaDzuBt43OyW9Fdf/8Dz5/e6Vi8C4FcRr8/rerNqnCmiNB5WpkYejsBpIEStG06josI9Iq0NWzhjSKvY7CadqKN/53kur1WxZe1IR7Desnakoxtzbn4B9+0+gp17j2PnresAABcuG2p/fEiak59uabz0H7QNPripMVFCfpN0QGcQn5mdM6YT/PLIpqBrf++ND78UeJ2LqrE3+/ULEpun93mOrucXGpj84VFA0TGCr4igMtRZqpjWS/+idXRmjVuqESXkN0nnDI528DRxj0DtP073fozuoDszO4f5Bf/RN9CcjDM9bB5+/njsP3y/4OhVT95YUtSqFi66cFnqgWeQNvjIegEwBnAaCH6TdAuNRdy3+wh2vfgGzn7WvQKgzT0Cdf9xusOgs3TPHun7sb///buPeH78zNlGuyEnqrA7tjt9uNDAkR1fi3yuIJk3zhRI1ukilhHSQDDtv+g0N7/gm6Nebg3h/t1H2jXRXn+cXt/zwWePBS7g7yyD8xuJxl1gKczP75bViLgsJYBp4GJWRCmwg4NX+V5YdnC3JwDDqIgEBnmg+QdtB+fJrWuM3z/uH7798z/8/PGuh5RVka4ceNYj4jKUAKaBi1kRpWRirI4/vWtD5JFoXFbFe9NdL85mGQCoVS3Pz0vyhz8xVsfsf/savr9tY8fod9fXN2DXnRsGYkSct6ybm1gHTgPHOfGYlQsqgs9CLjblVhHB3dd3b8IQpVY6r2VdKViWdeAM4FRKafxRhG1e6YWgxhs/URtlGOyLj408VCp+QSWt0ixnA4y7DLDXFhqLeOLACdRrVXxv20bfn8t9r7wqaUyVD1mXuVG2GMCpcB6aOYYnD5ww1lSnWZrlnEzLI7USVVBA9QrAJl4ToIPWFdlvAicxRWS5iPxcRI6KyHERebh1fJWIvCwib7b+XZn95VK/m5md6wjeNrtW27QlGNAMXkl2f7GXFE1xvf1Atarlu/8l4L8UbJhSRpvXBOigdUX2mzBVKJ8CuEFVNwDYCOBGEdkEYArAK6p6DYBXWu8TJbLrxTd80xh2qsMkjS284lZ6CPw3I3av/Fq1Kth567pQlTFJA62p8iHPNbzjSGvj4n4VGMC16aPWu1brPwVwG4DHW8cfBzCRxQXSYLD/UMOkLxTmXanS2E08TtNLrWrh8lrVt2zwu3dt9CzVsxtb/MQJtBWRwLLAPNfwjirPPTbLKlQOXEQqAA4D+F0Af6aqB0XkUlU9CQCqelJELjF87XYA2wFgdHQ0naumvhKnGsRvlJ705b97crMi/vXc1pDg48/O+a51Uq9VfZtX7Ny+1wNMAGNA9Wv6WVINXPo1y+3EkmJ+PlioAK6qiwA2ikgNwI9E5IthT6CqjwF4DGiWEca5SOpPWU0apvHy3x1sTQ+ZWtWCCHxb8KPs+u4+hwD45qZR38Dv1V0JhL8PRe2KZH4+WKQqFFWdF5GfArgRwPsicllr9H0ZgFNZXCD1D2e5W23YwkefnPPdhCAOqyJdwTKt3V8A75Hq1VMvGL+uHuF8cUfDO25Z15eLQw3SqoVxBQZwERkB0GgF7yqAPwTwPwDsBXAvgOnWv89leaFUbu4RbNDGBnFddMEy35Fz2DrnoLW9nUyBpl6rYv/UDe38fpyt2cJ8bZHTIEkM0qqFcQV2YorIl9CcpKygOem5R1W/IyIXA9gDYBTACQB3quppv+/FTszBFXaCMg3OLclM57WDq9vM7Bx27j3umc8eEuA/XD+KRybWd32NqfMRQOztwwZt6zEv7BJtit2Jqar/AGDM4/hvAHwlncujfpdX3tJdxmc6r1dQD5pMXVLgiQMnAKAjiPuNgL12wgk7EcdJvOLm54uCnZiUizgbCsThrhbxO697c4SwTTFPHXy3axRuCjRJJuLSnsTjaLb/cDlZyoVXvbFVSb/nse6a4PLLl+7ce7zj/bCBMcp64kkaZdJssrF3rnfWVE8+fZQ11SXHAE65mBir44qVyzuOrb54OPXzbFk70nVek/mFRkeHX9jAGNT67pSkUSbNJpude493Vfw0lrTrIUblwgBOqQhqef7m//4Z3jz1cccx9/tpeOrgu5FGlc4Ovy1rR0J1YN59/ZWhv3+S7cPS3HrM1GQUZqNlKi7mwCmxMKV6+9/yLVBKjXuXeQBYOWwFli0uNBbx6usf4NHb13fkiVdfXMWBX57Bomp7owV3/jtIkok4TuKRHwZwSqxo1RLuc++4ZR0mf3gUjYAdct6bX+jbgGl6iK0c9t66jcqBKRRKrIgtz85zT4zVsevr5/d8NOWw+7nDb8ct67omja2KYMct63p0RZQGjsApsTAtz5u/sCq3NIr73ED3xg2D1uHXr92ag44BnBIL0/J85/hoagFcBPCr5AsKxoMazPo1PVR0WdbfM4BTYhNjdRx65zSeOvhue7LvjmubwSKLFQeXCdAwBHDT4lFef0RerfRRsTmG/GS95yh3pS+ZXgWMKJsMA+d3VX/m8Fyuu76/7bH+dVZrivgtMbvz1nUM5BR5LR4T7krfB7J+mpuCdNB5TVUo9og8TxsffqkreEapknHfgy1rR/Dq6x94PrhMrffzCw3u7E4Asp/gZxVKifgFoqT8tq8KOq/pf8a8gzdwPng6m3nC/hF53YMnDpwwbunllxZK6/dC5Zb1nqMcgZdIlk9zvyAddN68FqoKyz269quScY64hwK2TnN+7zC4cwxtWTvSXsHSfTwNHIGXSJZPc78gHXTeOAtVuRedSpvz5zGtKbJl7UjHiDvsK4b35hfw8PPBa4j0c105hfPq6x9EOh4VA3iJZLmDuCnYrKhaOPvZua7jzvN6rdlx0QX+L+4+/rT7e6bJ/nmcKSC7gcdeU+TV1z+INcF6ea0a2Jrf73XlFA5z4NSW5uJGbl4PB6CZU3YHq1rV6jiv1+TnhwGLJGW5iJIdPJ05baA5wrY/NjFWj/VHFCYwp/l7oXJjDpw6ZNWM4aysmJtfgKC5Up+Xiy5s/m/jVSJlT/QNX1DBx5/lVz7oZAfPoN1wTLnxigiWVH2rUEzbrtWqVir15dQfJreu6VqHx2vj7bgYwKnNfjgE7V9pbwZg2lE+z7pvt3qt2n4YBb18NXWQhhk977x1Xdc9sIYEO29NZ20RNgj1EfefSYrFWUyhUJcwqQVT8O4ld3oj6OVr0rW6d925oeNrd925IbV6fFNJJ5XLrhff8NxII60SU47AqUvRygLD8GqhD7NGSxHX6i7a8rwUHycxKXemCc0i86qUsUfYzjWvL1xW/P/li7g8L8XDScwSK0oe03kdK6oWRID5s42Ot53XZ1+jaaKuiM6cbWDyh0cBdLevf9JYar9dhjb3MMvzUjmEeRWYRPGHIyVVlDym+zrsskD32+7rmxir48iOr+H72zbmer1JNBa1q8HGlI4o8ma+Wdb7U76yLP0FOALPTK/zmHGWcTVdnzUkhZi0tIYE51R91wJ316yb0g7zCw3MzM4VchQ+qOuV96ss12FnAM9IL/OYpmVOw7CvL4t1vOMSoB3EAET62fwmZIs8KcjNFygMBvCM9CKPmUbQvbxWxUMzx/DkgRNplqvGJgC+t21jVzC7f/cRz+urVTs36Z3cugb37T7i+b05KUhlxxx4RvLOY7rbxuOwF3gqSvAGmj0PzppZ+yHldX1ejTQTY3XjzuucFKSyYwDPSNaTF26mzQXcalULK4ctiOtt5wJPRQneNmdax/2Qstc89Guk2XHLOk4KUl9iCiVDeeYxg9IBYVvE7zekG/Jg2qzYHil7PaQUwdtTcVKQ+hUDeJ8I6p5caCy2S+y8ApdfaiJLdo7blLu30zp+67OESRtxUpD6EQN4D83MzuHh54+3S99qVQs3b7jMuAej+2vtoFcJsZMM0Cyxe+Dpo+337RFpbdjCR5+c60mp4IqqZawqEQBfHl0RuDGyvc430aAJDOAiciWAvwLwrwEsAXhMVf+niKwCsBvAagBvA7hLVc9kd6n9ZWZ2rmuZyfmFRsf2S6ZNi91lglH2nlxcUty/+wiWVaR97qDNCbIkYl69UAEc+OWZwJ8vyd6bzi7V2rAFVeDDhQbTLFQKYSYxzwF4QFV/D8AmAH8kIr8PYArAK6p6DYBXWu9TSLtefKMjeJt47cEYdsLSRIFQ587aymEL8wEPjzDBOe72bO4u1TNnG5hf8O5MJSqiwACuqidV9Rett38L4DUAdQC3AXi89WmPA5jI6Br7UpQaZPfnFqG5JimrIthxy7rAUr6g9IgAsapJZmbn8MCeo74PQu4sT0UXKQcuIqsBjAE4COBSVT0JNIO8iFxi+JrtALYDwOjoaKKLLbuoO6DbVlQtbJ7e136ZX3YXXVCBVRnCfbuPICh9fff1V/rmwBXRF6WyR95h7j+bfajIQteBi8jnADwD4D5V/ZewX6eqj6nquKqOj4yMxLnGvuB+uR42eFtDgo8/O9fxMj/o84etYpT3u2Nz1argnk2jWNLze2L63YaVwxYemViPR29fbxyJx0mfRElBsdmHiizUX7qIWGgG7ydV9dnW4fdF5LLWxy8DcCqbS+wPpqAx5IhLtaqFezaNtoNSRZqLSIXNV9drVWy77kpoV+jsDbtG290oFCZ4Vq0KdtzS7KqcGKvjT+/akFozTthRNZt9qOjCVKEIgL8A8Jqqftfxob0A7gUw3fr3uUyusODCrvltChqqwNvTN3V9z6iLUdVrVay+uNpRxdJrK4e7N/gN0yjktbtOms04ppp5EWDFcotVKFQaYXLgmwH8RwDHRORI69ifoBm494jItwCcAHBnJldYYO5A61X2F9Qgs6LandOOU2UyfMEQ9r91OtLXZO1DjyVbgxqO/Loq02rGSbKZMVGRhKlC+VtVFVX9kqpubP33E1X9jap+RVWvaf1brOiRA781v4FwC0z99tNzXaVqcapM3jz1se/HRWBc1CkrS4quDRb8tmvLK2WR9zo1RFnp607MrLc0C1rzO8xIenGpuYuMc8SeBdXOrcXy4p50daZCnA+qigjuuDa/dne21lM/KEa5Qgby2NIsaMPSsJNldpCzrzkrC43FwLK9PEyM1btG4ouqeObwHBtniCLo2wAelN5IQ9Ca31FK0OxXC0k6LMNI0HUei3uDBZvp9/PAnqO4euoFbJ7ex2BOFKBvA3geW5oF5VL98r1uSTdj6LWKNGvQnbw2WLCZfg+Lqu1XTJNPH2UQJ/LRtznwLLY0M+XUTblUd+lbbdjCh2cb8MpELzQWQ68qaOKspEiyL2YcQyLYdt2VoVZSBIKrUQCgsaTYufc4c9VEBn0bwE2lYnGrHMKUDHpxB/iZ2TnjHo1Jgre7dto0WZiVxpLiqYPvYkk11ISx1+/Hi92xSUTd+jaFErZUbGZ2Dpun9wXmXdPKqU+M1WOvnmdi1067f7aJsTr2T92Q+vlMnOmPoAlj9++HiKLr2xE4EFwqFmVUHWU3mKDyxbCjzzCsIQl8VeG3M3tW7Idb2FcnY995yXOdl7xr14nKpG9H4GFEGVWbFlNyHw9TvmiPPqMuOmUNCS66wD0pqvjPe45g9dQL+MKDP8FDM91liBNjdVR7sMBVlAnjHbesg1VxTYK2lpwlIm99PQIPEqVSxZSfdh/3eyi4R6MLERprBMC2667E+FWrOkbvzm+xqNpeC+WRifUdrwTiZNcFwO9echHe+uBj2LutVa0hLLcqoXbxcU4YB70q4cbDRNENdACPUqlSN3yuO78c9qEQdQNhBfDCP5zEkwdPBNZyP3HgBJ44cALS+rq4FM0WffcE6dVTLwR+rXPCOGyqit2RRNEMdAolqBEnzucGdWfa4tSjnznbiNSIk1bPjjsNFFSK6Z4wzqOpimgQDXQAj7KoUdjPTRroBebuxazUqlZgpYoz4Hr9jALgnk2jeHv6pq6KmDyaqogG0UCnUILysl4fNy11apsYq+PQO6fx1MF3sahqXKTJVImy3BrCzRsu891GLE3Obsmgyhg74EbNV2fRVEVEAxzAg/Kyfh8HOoPXlrUj7Q7E4Qsq+Piz80HQXqRp/KpVXfneQ++c7sppLzSW8MzhOdxxbb39PaPsnxnV55Yva/88Qd2gzrXLo+Sr026qSlPWK1YSZUk0x9WNxsfH9dChQ7mdz8/m6X3GScn9UzcYP75y2MInjaXIo2P3RgVBre72xGHSTsowE5lVq9JxHVZFsLioXS3/VkWw7d+Gb5d3KmKg9PodcGMHKiIROayq4+7jAzsCD8rLmj4epnwuzPmCVh60R/xJ0ijO6hHTA6ki0nWOxqI29+p0Rf7GouLJAyfah8MuJ2B/vGhBMUrJJ1ERDewkZlC1SNr52ThVKH7B2xqSrsYXJ3d7/Za1I56fZ0qXLBmG7e7DZa4m4eQqld1ABHCv9U6CqkVMH49TISKt7+fktRdmWBUR7LpzA3Z9fYNHZ6Z3fvnV1z8wfq+kyhrwwpZ8EhVV3wdwU2v7oXdO48Jl53/8lcNWR+5zYqyOO66ttwOcXU2y89Z1odf4tn1z0ygAtB8iGx9+CR8mWGVvSbV9ne6RsgCeVS9+62+H/XlMob6sAS9KHwBREfV9ADflOZ88cKJjqVL3fpEzs3N45vBcO8VgV5MA6AjsYdjt7/ZDZH6hkajJxg6YXj+bwnu0bQqydj2731ovdt37NzeN9lXA4+bGVHZ9P4lpGnmacrlB3YM79x7Hp+eWOnLHfpUe9Vo11a3SnAEzSg7Xr5RvYqyO+w2rFS6p4lfTN7XfH79qVeGqSZIo4uQqUVh9H8DD7PxicwY+U3D02mBA4V1eaAdIU3CMyr0mSZQGmaDmm7DfiwGPqDj6PoB7jTxNI2ZnsIoS+AFg/mwD39u2sR0gV1QtiAD37z6SuBHnnk2jeGRifdfxya1rMPn0UTQciXC/9cH9gm+Rm22IyFvpAnjUhhCvkeeWtSNdreruYGUKaMutIc9a8Mtr1XaAdDeIJAneVkUwftWqjmP2PfB8wAhw6J3TkdMcXM6VqHxK1YmZZudcmAeB1+cA3muGrBy2sOOWdb5NMybWUOe63m7OLs4wmxW7X2EImpUwXqN4Iiq+vujETLNzLkwu1+9zdu493pEPP3O20e5KjFIXbQ01d3Pf/fN3O1IhTs7vF2ZC1P1dFMCTB050rcdCROVWqjLConTOTYzVcdGF3c8++2ESpS66saR49fUPsOvODcZSPuf3i/uzKlC6jsmwG04TDapSjcBNE4u1YQubp/flmrv1e5h8b9vGSOuYvDe/0L7eoInEoMlVv5LGondMOlNWtWELH31yrv2qJMq6K0SDolQjcK/OOasi+OiTc76bCGfBrw3bbhAJ2+xjf68wjSWmzRSA8802ZeyYdHfMnjnb6EoplXndFaIslGoE7lUp8fGn57pqs/NYUS6o7M40onZzj7CDcvNhq0WcqwZ6nadowjY7Ff1VBFGeAgO4iPwAwM0ATqnqF1vHVgHYDWA1gLcB3KWqZ7K7zPPcAc60wW7Wf+hhAqmphDHOetruc/t9zSMT60vXMRn291XkVxFEeQszAv9LAP8LwF85jk0BeEVVp0VkqvX+f03/8oLF3a4rbD253+clrWTJUpjzFmmThTCNU0V/FUGUt8AcuKr+DYDTrsO3AXi89fbjACbSvazw/FaUM1UxmFYodOfNw35eGRXtZzPNb9SqFheaIjKImwO/VFVPAoCqnhSRS0yfKCLbAWwHgNHR0ZinMzOlMgAY97QMW0/ezzu2FO1nYycoUXSZT2Kq6mMAHgOanZhZnMMrXbB5ep8xQIWtJy9C3XlWaY4i/GxuXCiLKJq4ZYTvi8hlAND691R6l5QOvwBlyo8r0JFqSbJjS9gmFL/PyzLNwd1oiMovbgDfC+De1tv3AngunctJj1+A8sq32pxBMu6OLWnl2P3WJE+Ku9EQlV9gABeRpwD8DMAaEfm1iHwLwDSAr4rImwC+2no/E3HbqbesHelqaHFuYGA3zHhx5oLj7Njil1+O8nl+a5InHYVzNxqi8gvMgavq3YYPfSXla+niXnkvbDu1vR2aO+H+5dEVXSWAV0+94Nl6bgfPOHlZU+B1l8kF5aH9SuvSmGxkzpmo3ArdSh92JBvm6wDg79463TVyzSIXbPpaATrOH3Ruv3QGOxKJqNABPG6lhN8+mO7gn0UueHLrGs/1SNznDzr3xFgdK4ctz3NwspGICh3A446O/T7uFdwvXHb+NqwcthLngifG6qFWBAyTh95xyzpONhKRp0IvZhV3n0Z7I+GgfS+9drf5xG9rnAjqKW0SzAYXIjIpdACPG7wmxuo49M7pwBX54nYjhmmuSXOTYE42EpGXQgfwJMKsyBcnxx62MoYjZyLKWqEDeNwyQlvQyDXqSoYzs3N4YM/Rrl3mTaN2jpyJKEuFnsSMW0YYlqkj8+xn54xdk+7gbUurrI/7QBJRWIUegWe94JI9OvbbYd6ZCvHbMaZmKPeLIukrDiIaLIUegafRZBM0og3aYd4W9NAwDMwjyfoVBxH1l0IH8KRNNmEXlQoz0g96aHzo2pczjiIu8UpExVXoAJ50waWwI9owI32/FQz9vkcUXOKViKIodA4cSFbJETSiteu55+YXIIBvzbh9DQ8/fxxnznaOttPqjEyzdpyI+l/hA3gSfmWC7glDBdpBvG6o2bYfJlntksPacSKKQjSN2beQxsfH9dChQ7mdz6tVvmpV8Ojt69sjb7d6rYr9Uzfkdo1EREFE5LCqjruPFzoHnpRfDp0ThkRUdn2dQgHMOfSoXZi9klW6hojKr+8DuJeZ2Tmc/exc1/GiTRiysYeI/PR1CsWLHRTdlSTD1hAuXDaE+3cfKUwLOxt7iMjPwI3ATS3xC40lnG2tBV6UkS7z9ETkZ+BG4H7brTkVYaTLxh4i8jNwATxK8Ov1SDeL/TqJqH8MXAD3CopeGxADvR/pJl1KgIj628DlwL26HbesHcEzh+cK2cLOTSGIyGTgAjjgHRSDtl8jIiqagQzgXgZxpMsmIaJyYwAfUGwSIiq/gZvEpCY2CRGVH0fgBZNXWoNNQkTlxxF4gYTdAi4NbBIiKj8G8ALJM63BJiGi8ksUwEXkRhF5Q0T+SUSm0rqoQZVnWoNNQkTlFzsHLiIVAH8G4KsAfg3g70Vkr6r+v7QubtDkvUb5IJZOEvWTJCPw6wD8k6r+UlU/A/DXAG5L57IGE9MaRBRFkiqUOoB3He//GsD1yS5nsHFTYyKKIkkA91oDqmuHZBHZDmA7AIyOjiY43WBgWoOIwkqSQvk1gCsd718B4D33J6nqY6o6rqrjIyMjCU5HREROSQL43wO4RkSuFpELAHwDwN50LouIiILETqGo6jkR+U8AXgRQAfADVT2e2pUREZGvRK30qvoTAD9J6VqIiCgCdmISEZUUAzgRUUkxgBMRlRQDOBFRSTGAExGVFAM4EVFJMYATEZUUAzgRUUkxgBMRlVTfbmqc1+bARES90pcB3N4c2N5f0t4cGACDOBH1jb5MoeS5OTARUa/0ZQDPc3NgIqJe6csAbtoEOKvNgYmIeqEvAzg3ByaiQdCXk5jcHJiIBkFfBnCAmwMTUf/ryxQKEdEgYAAnIiopBnAiopJiACciKikGcCKikhJVze9kIh8AeCe3E2bj8wD+udcXUSC8H+fxXnTi/eiU5H5cpaoj7oO5BvB+ICKHVHW819dRFLwf5/FedOL96JTF/WAKhYiopBjAiYhKigE8usd6fQEFw/txHu9FJ96PTqnfD+bAiYhKiiNwIqKSYgAnIiopBnAfIvIDETklIv/oOLZKRF4WkTdb/67s5TXmRUSuFJFXReQ1ETkuIt9uHR/U+7FcRH4uIkdb9+Ph1vGBvB8AICIVEZkVkR+33h/ke/G2iBwTkSMicqh1LPX7wQDu7y8B3Og6NgXgFVW9BsArrfcHwTkAD6jq7wHYBOCPROT3Mbj341MAN6jqBgAbAdwoIpswuPcDAL4N4DXH+4N8LwBgi6pudNR+p34/GMB9qOrfADjtOnwbgMdbbz8OYCLPa+oVVT2pqr9ovf1bNP9Q6xjc+6Gq+lHrXav1n2JA74eIXAHgJgB/7jg8kPfCR+r3gwE8uktV9STQDGoALunx9eRORFYDGANwEAN8P1opgyMATgF4WVUH+X58H8AfA1hyHBvUewE0H+YvichhEdneOpb6/ejbHXkoGyLyOQDPALhPVf9FRHp9ST2jqosANopIDcCPROSLPb6knhCRmwGcUtXDIvIHPb6cotisqu+JyCUAXhaR17M4CUfg0b0vIpcBQOvfUz2+ntyIiIVm8H5SVZ9tHR7Y+2FT1XkAP0VzvmQQ78dmALeKyNsA/hrADSLyBAbzXgAAVPW91r+nAPwIwHXI4H4wgEe3F8C9rbfvBfBcD68lN9Icav8FgNdU9buODw3q/RhpjbwhIlUAfwjgdQzg/VDVB1X1ClVdDeAbAPap6j0YwHsBACJykYj8jv02gK8B+EdkcD/YielDRJ4C8AdoLgP5PoAdAGYA7AEwCuAEgDtV1T3R2XdE5N8B+L8AjuF8nvNP0MyDD+L9+BKaE1EVNAdCe1T1OyJyMQbwfthaKZT/oqo3D+q9EJF/g+aoG2imqf+Pqv73LO4HAzgRUUkxhUJEVFIM4EREJcUATkRUUgzgREQlxQBORFRSDOBERCXFAE5EVFL/H2+D/HBghPrHAAAAAElFTkSuQmCC)

In [22]:

```
t1=load_iris()
t1
```

Out[22]:

```
{'data': array([[5.1, 3.5, 1.4, 0.2],
        [4.9, 3. , 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [5.5, 3.5, 1.3, 0.2],
        [4.9, 3.6, 1.4, 0.1],
        [4.4, 3. , 1.3, 0.2],
        [5.1, 3.4, 1.5, 0.2],
        
		. . .
		
        [6.5, 3. , 5.2, 2. ],
        [6.2, 3.4, 5.4, 2.3],
        [5.9, 3. , 5.1, 1.8]]),
 'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),

```

In [23]:

```
from sklearn.svm import SVC
f=[2,3]
X=t1.data[:,f]
Y=t1.target
m=SVC(kernel='linear',random_state=0)
m.fit(X,Y) # 학습
```

Out[23]:

```
SVC(kernel='linear', random_state=0)
```