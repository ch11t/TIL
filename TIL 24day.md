# 머신러닝

```
mport numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
df=pd.read_csv('perch_full.csv')
X=df.to_numpy()
Y = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]
     )
t_x,tt_x,t_y,tt_y=train_test_split(X,Y,random_state=42)
t_x.shape
```

Out[1]:

```
(42, 3)
```

In [2]:

```
from sklearn.preprocessing import PolynomialFeatures
p_m=PolynomialFeatures(include_bias=False)# 전처리기
p_m.fit([[2,3]])
p_m.transform([[2,3]])
```

Out[2]:

```
array([[2., 3., 4., 6., 9.]])
```

In [3]:

```
p_m=PolynomialFeatures(include_bias=False)
p_m.fit(t_x)
p_t_x=p_m.transform(t_x)
p_tt_x=p_m.transform(tt_x)
```

In [4]:

```
p_m.get_feature_names()
```

Out[4]:

```
['x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2', 'x2^2']
```

In [5]:

```
from sklearn.linear_model import LinearRegression
m_lr=LinearRegression()
m_lr.fit(p_t_x,t_y)
lr=LinearRegression().fit(t_x,t_y)
print(lr.score(tt_x,tt_y),lr.score(t_x,t_y))
0.8796419177546366 0.9559326821885706
```

In [6]:

```
m_lr.score(p_tt_x,tt_y)
```

Out[6]:

```
0.9714559911594155
```

In [7]:

```
m_lr.score(p_t_x,t_y)
```

Out[7]:

```
0.9903183436982125
```

In [8]:

```
p_m1=PolynomialFeatures(degree=5,include_bias=False).fit(t_x)
d_t_x=p_m1.transform(t_x)
d_tt_x=p_m1.transform(tt_x)
```

In [9]:

```
d_t_x.shape
```

Out[9]:

```
(42, 55)
```

In [10]:

```
m_lr1=LinearRegression().fit(d_t_x,t_y)
print(m_lr1.score(d_tt_x,tt_y),m_lr1.score(d_t_x,t_y))
-144.40744532797535 0.9999999999938143
```

In [11]:

```
from sklearn.preprocessing import StandardScaler
ss=StandardScaler().fit(d_t_x)
sc_t_x= ss.transform(d_t_x)
sc_tt_x= ss.transform(d_tt_x)
sc_t_x.shape
```

Out[11]:

```
(42, 55)
```

In [12]:

```
from sklearn.linear_model import Ridge, Lasso
rg=Ridge().fit(sc_t_x,t_y)
rg.score(sc_t_x,t_y)
```

Out[12]:

```
0.9896101671037343
```

In [13]:

```
rg.score(sc_tt_x,tt_y)
```

Out[13]:

```
0.979069397761539
```

In [14]:

```
import matplotlib.pyplot as plt
t_l=[]
tt_l=[]
ap_l=[0.001,0.01,0.1,1,10,100]
for i in ap_l:
    f_rg=Ridge(alpha=i).fit(sc_t_x,t_y)
    t_l.append(f_rg.score(sc_t_x,t_y))
    tt_l.append(f_rg.score(sc_tt_x,tt_y))
```

In [15]:

```
plt.plot(np.log10(ap_l),t_l)#R^2
plt.plot(np.log10(ap_l),tt_l)
```

Out[15]:

```
[<matplotlib.lines.Line2D at 0x27e83c17670>]
```

![img](data:image/png;base64,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)

In [16]:

```
rlr=Ridge(alpha=0.1).fit(sc_t_x,t_y)
print(rlr.score(sc_t_x,t_y))
print(rlr.score(sc_tt_x,tt_y))
0.9903815817570368
0.9827976465386954
```

In [17]:

```
t_l=[]
tt_l=[]
ap_l=[0.001,0.01,0.1,1,10,100]
for i in ap_l:
    lo=Lasso(alpha=i).fit(sc_t_x,t_y)
    t_l.append(lo.score(sc_t_x,t_y))
    tt_l.append(lo.score(sc_tt_x,tt_y))
plt.plot(np.log10(ap_l),t_l)#R^2
plt.plot(np.log10(ap_l),tt_l)

```

Out[17]:

```
[<matplotlib.lines.Line2D at 0x27e83d4ef70>]
```

![img](data:image/png;base64,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)

In [18]:

```
lasso = Lasso(alpha=10)
lasso.fit(sc_t_x,t_y)
print(lasso.score(sc_t_x,t_y),lasso.score(sc_tt_x,tt_y))
0.9888067471131867 0.9824470598706695
```

---

---



외부 모듈 로드

In [1]:

```
import numpy as np #data계산을 위한 모듈
import pandas as pd #data 로드를 위한 모듈
from sklearn.model_selection import train_test_split #학습 data와 태스트 data 분할 모듈
from sklearn.preprocessing import PolynomialFeatures #피처 증가를 위한 모듈
from sklearn.linear_model import LinearRegression ,Lasso, Ridge #선형 회기 모델 모듈
from sklearn.preprocessing import StandardScaler #data 전처리 (중복 정보 정리)-> 벨런싱 작업
import matplotlib.pyplot as plt #시각화 모듈
```

다중 선형 회귀

1. data 가지고오기

In [2]:

```
df=pd.read_csv('perch_full.csv')
X=df.to_numpy()
Y = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]
     )
```

2.학습 data와 태스트 data 분할

In [3]:

```
t_x,tt_x,t_y,tt_y=train_test_split(X,Y,train_size=0.7,random_state=42)#7:3비율조정
```

3.data 전처리

data 피쳐 증가 (학습의 진행을 상승)->과소적합 방지

In [4]:

```
#p_m=PolynomialFeatures(include_bias=False).fit(t_x)#학습 data 기반으로 피처 증가
p_m=PolynomialFeatures(degree=5,include_bias=False).fit(t_x)
p_t_x=p_m.transform(t_x)
p_tt_x=p_m.transform(tt_x)
```

data 전처리 밸런싱 작업 (특징 감소)-> 과대적합 방지

In [5]:

```
ss=StandardScaler().fit(p_t_x)#한쪽으로 치우친 피처 정보를 균등할수있게 정리
sc_t_x= ss.transform(p_t_x)
sc_tt_x= ss.transform(p_tt_x)
```

4.모델 생성 및 학습

In [6]:

```
lr = LinearRegression()#학습 모델을 결정하고 알맞은 모델을 생성 (회귀모델)
lr.fit(sc_t_x,t_y)# 학습 진행(지도: [입력,결과])
```

Out[6]:

```
LinearRegression()
```

하이퍼 파라미터 결정

In [7]:

```
rg = Ridge(alpha=0.1)#alpha 값을 결정하여 이상적인 하이퍼 파라미터 값 제시
rg.fit(sc_t_x,t_y)# 학습 진행(지도: [입력,결과])
lso = Lasso(alpha=10)
lso.fit(sc_t_x,t_y)# 학습 진행(지도: [입력,결과])
```

Out[7]:

```
Lasso(alpha=10)
```

5.테스트 및 검증

In [8]:

```
y_p=lr.predict(sc_t_x)#학습된 모델을 이용하여 예측값생성(오차를 계산할수있는 기본적인 내용 정의 가능)
t_s=lr.score(sc_t_x,t_y)#학습 data를 이용하여 점수 확인
tt_s=lr.score(sc_tt_x,tt_y)#테스트 data를 이용하여 점수 확인
print(t_s,tt_s)# t_s-tt_s 1.둘다 낮은 값 이라면 과소적합 2. 학습data 점수는 높은데 테스트 data 점수는 낮은 값이라면 과대적합
1.0 -26.192517921905765
```

In [9]:

```
t_s=rg.score(sc_t_x,t_y)
tt_s=rg.score(sc_tt_x,tt_y)
print(t_s,tt_s)
0.9897983159614502 0.9842243738800824
```

In [10]:

```
t_s=lso.score(sc_t_x,t_y)
tt_s=lso.score(sc_tt_x,tt_y)
print(t_s,tt_s)
0.9882059522438204 0.9834044009315154
```

---

----

# K-NN(k-nearest neighbors) 회귀

In [1]:

```
import pandas as pd
import numpy as np
data=pd.read_csv('data1_all.csv')
```

1.data수집

In [2]:

```
data_X=data[['Weight','Length','Diagonal','Height','Width']].to_numpy()
data_Y=data['Name'].to_numpy()
```

2.data 전처리

In [3]:

```
from sklearn.preprocessing import StandardScaler
ss=StandardScaler().fit(data_X)
data_X=ss.transform(data_X)
3.입력 data 정리
```

In [4]:

```
from sklearn.model_selection import train_test_split
t_x,tt_x,t_y,tt_y=train_test_split(data_X,data_Y,random_state=42)
```

4.모델 생성 및 학습

In [5]:

```
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=3)
kn.fit(t_x,t_y)
```

Out[5]:

```
KNeighborsClassifier(n_neighbors=3)
```

5.검증

In [6]:

```
kn.score(t_x,t_y),kn.score(tt_x,tt_y)
```

Out[6]:

```
(0.8823529411764706, 0.85)
```

In [7]:

```
kn.predict(tt_x[:4])
```

Out[7]:

```
array(['E', 'G', 'F', 'E'], dtype=object)
```

In [8]:

```
tt_y[:4]
```

Out[8]:

```
array(['E', 'G', 'F', 'C'], dtype=object)
```

In [9]:

```
print(np.round(kn.predict_proba(tt_x[:4]),decimals=3))
[[0.    0.    0.    0.    1.    0.    0.   ]
 [0.    0.    0.    0.    0.    0.    1.   ]
 [0.    0.    0.    0.    0.    1.    0.   ]
 [0.    0.333 0.    0.    0.667 0.    0.   ]]
```

---

---



# 로지스틱 회귀(다중분류)

In [1]:

```
import pandas as pd
import numpy as np
#1.data수집
data=pd.read_csv('data1_all.csv')
data_X=data[['Weight','Length','Diagonal','Height','Width']].to_numpy()
data_Y=data['Name'].to_numpy()

#2.data전처리
from sklearn.preprocessing import StandardScaler
ss=StandardScaler().fit(data_X)
data_X=ss.transform(data_X)

#3.data분리
from sklearn.model_selection import train_test_split
t_x,tt_x,t_y,tt_y=train_test_split(data_X,data_Y,random_state=42)
#4.모델 생성 및 학습 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=20,max_iter=1000)#다중분류용
lr.fit(t_x,t_y)
lr.score(t_x,t_y),lr.score(tt_x,tt_y)
lr.coef_ , lr.intercept_
```

Out[1]:

```
(array([[-1.43104781, -1.03881083,  2.68339449,  7.77024432, -1.18106054],
        [-1.41304735, -6.04954057,  5.28192373, -0.97312708,  1.92823922],
        [ 0.66899584, -2.29121452, -0.86671496,  1.64919507,  3.7778608 ],
        [ 0.20008722, -2.00035538, -3.74686768,  6.64643665, -1.95319595],
        [ 3.57308534,  6.36292416, -8.57477602, -5.8900385 ,  3.8016005 ],
        [-0.16774626,  3.61014133,  3.93270813, -3.58340455, -1.8315986 ],
        [-1.43032697,  1.4068558 ,  1.29033231, -5.61930592, -4.54184543]]),
 array([ 0.17817092,  2.5409984 ,  1.34452841, -0.06020156,  3.06520085,
        -0.24112062, -6.82757639]))
```

In [2]:

```
d=lr.decision_function(t_x[:1])#학습된 공식을 꺼냄
d
```

Out[2]:

```
array([[ 12.68635502,   2.05193796,   6.44611303,   5.40571709,
         -3.14126816,  -3.00365919, -20.44519576]])
```

In [3]:

```
from scipy.special import softmax
np.round(softmax(d))
```

Out[3]:

```
array([[1., 0., 0., 0., 0., 0., 0.]])
```

# 로지스틱 회귀(이중분류)

2종 data로 변경

In [4]:

```
i=(t_y == 'A')|(t_y == 'G')
b_t_x=t_x[i]
b_t_y=t_y[i]
lr = LogisticRegression()
lr.fit(b_t_x,b_t_y)
```

Out[4]:

```
LogisticRegression()
```

In [5]:

```
lr.coef_ , lr.intercept_
```

Out[5]:

```
(array([[-0.41981254, -0.60056128, -0.68786241, -1.00159093, -0.74532495]]),
 array([-2.1898649]))
```

In [6]:

```
d=lr.decision_function(b_t_x[:1])#학습된 공식을 꺼냄
d
```

Out[6]:

```
array([-5.96982853])
```

시그모이드 함수(구현)

In [7]:

```
def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig
import matplotlib.pyplot as plt
z=np.arange(-5,5,0.1)
plt.plot(z,sigmoid(z))
```

Out[7]:

```
[<matplotlib.lines.Line2D at 0x2c4496d3b80>]
```

![img](data:image/png;base64,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)

시그모이드 함수 (내장)

In [8]:

```
from scipy.special import expit
expit(d)
```

Out[8]:

```
array([0.00254817])
```

---

---

## Q1

<div><br class="Apple-interchange-newline">from sklearn.linear_model import LinearRegression <br></div>

```
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

X = [[174], [152], [138], [128], [186]]
y = [71, 55, 46, 38, 88]
reg.fit(X, y)

print(reg.predict([[165]]))
plt.scatter(X, y,c='r')

y_pred = reg.predict(X)

plt.plot(X, y_pred)
plt.show()

[67.30998637]
```

![img](data:image/png;base64,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)

In [2]:

```
from sklearn import datasets
from sklearn.model_selection import train_test_split
diabetes = datasets.load_diabetes()
X=diabetes.data
y=diabetes.target
lr=LinearRegression()
t_x,tt_x,t_y,tt_y=train_test_split(X,y,test_size=0.2,random_state=42)
lr.fit(t_x,t_y)#다중 선형 회귀
y_pred = lr.predict(t_x)
plt.scatter(t_y, y_pred)
plt.show()
```

---

---

```
import numpy as np #data계산을 위한 모듈
import pandas as pd #data 로드를 위한 모듈
from sklearn.model_selection import train_test_split #학습 data와 태스트 data 분할 모듈
from sklearn.preprocessing import PolynomialFeatures #피처 증가를 위한 모듈
from sklearn.linear_model import LinearRegression ,Lasso, Ridge #선형 회기 모델 모듈
from sklearn.preprocessing import StandardScaler #data 전처리 (중복 정보 정리)-> 벨런싱 작업
import matplotlib.pyplot as plt #시각화 모듈
```

In [2]:

```
from sklearn.datasets import load_boston
b_data=load_boston()
b_data.keys()
```

Out[2]:

```
dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])
```

In [3]:

```
X=b_data.data
Y=b_data.target
Y.shape
```

Out[3]:

```
(506,)
```

In [4]:

```
#y_data=Y.reshape(Y.size,1)
y_data=Y.reshape(-1,1)
y_data.shape
```

Out[4]:

```
(506, 1)
```

In [ ]:

```
from sklearn.preprocessing import MinMaxScaler
mm_m=MinMaxScaler(feature_range=(0,5)).fit(X)
sc_x_data=mm_m.transform(X)
```

# 전처리된 data를 이용하여 코드를 완성하시오

1. 4가지 선형 회귀 모델을 학습 (LinearRegression, Lasso, Ridge, SGDRegressor)
2. 학습 성능 평가

```
import numpy as np #data계산을 위한 모듈
import pandas as pd #data 로드를 위한 모듈
from sklearn.model_selection import train_test_split #학습 data와 태스트 data 분할 모듈
from sklearn.preprocessing import PolynomialFeatures #피처 증가를 위한 모듈
from sklearn.linear_model import LinearRegression ,Lasso, Ridge,SGDRegressor #선형 회기 모델 모듈
from sklearn.preprocessing import StandardScaler #data 전처리 (중복 정보 정리)-> 벨런싱 작업
import matplotlib.pyplot as plt #시각화 모듈
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
b_data=load_boston()
X=b_data.data
Y=b_data.target
y_data=Y
mm_m=MinMaxScaler(feature_range=(0,5)).fit(X)
sc_x_data=mm_m.transform(X)
```

In [2]:

```
t_x, tt_x, t_y, tt_y = train_test_split(sc_x_data, y_data, test_size=0.3,random_state=10)
```

In [3]:

```
regr = LinearRegression(fit_intercept=True,normalize=False)
lasso_regr = Lasso(fit_intercept=True,normalize=False)
ridge_regr = Ridge(fit_intercept=True,normalize=False,alpha=0.01)
SGD__regr = SGDRegressor(fit_intercept=True)
regr.fit(t_x, t_y),lasso_regr.fit(t_x, t_y),ridge_regr.fit(t_x, t_y),SGD__regr.fit(t_x, t_y)
```

Out[3]:

```
(LinearRegression(), Lasso(), Ridge(alpha=0.01), SGDRegressor())
```

In [4]:

```
y_true = tt_y.copy()
y_hat = regr.predict(tt_x)
r2_score(y_true, y_hat), mean_absolute_error(y_true, y_hat), mean_squared_error(y_true, y_hat)
```

Out[4]:

```
(0.6996255772983109, 3.707127137271813, 29.3265965261233)
```

In [5]:

```
y_true = tt_y.copy()
y_hat = lasso_regr.predict(tt_x)
r2_score(y_true, y_hat), mean_absolute_error(y_true, y_hat), mean_squared_error(y_true, y_hat)
```

Out[5]:

```
(0.6192939837289688, 4.213744760928959, 37.16964858002032)
```

In [6]:

```
y_true = tt_y.copy()
y_hat = ridge_regr.predict(tt_x)
r2_score(y_true, y_hat), mean_absolute_error(y_true, y_hat), mean_squared_error(y_true, y_hat)
```

Out[6]:

```
(0.699630203179085, 3.7070744375887266, 29.326144885341712)
```

In [7]:

```
y_true = tt_y.copy()
y_hat = SGD__regr.predict(tt_x)
r2_score(y_true, y_hat), mean_absolute_error(y_true, y_hat), mean_squared_error(y_true, y_hat)
```

Out[7]:

```
(0.6123597645885241, 3.9782748681473588, 37.84666044117171)
```

In [8]:

```
plt.scatter(y_true, y_hat)
```

Out[8]:

```
<matplotlib.collections.PathCollection at 0x22cb333f8b0>
```

