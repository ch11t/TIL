# 데이터 전처리



<div><br class="Apple-interchange-newline">from sklearn.metrics import confusion_matrix y_t=[1,0,1,1,0,1] y_p=[0,0,1,1,0,1]</div>

```
from sklearn.metrics import confusion_matrix
y_t=[1,0,1,1,0,1]
y_p=[0,0,1,1,0,1]
confusion_matrix(y_t,y_p)
```

Out[1]:

```
array([[2, 0],
       [1, 3]], dtype=int64)
```

In [2]:

```
confusion_matrix(y_t,y_p).ravel()
```

Out[2]:

```
array([2, 0, 1, 3], dtype=int64)
```

In [3]:

```
tn,fp,fn,tp = confusion_matrix(y_t,y_p).ravel()
tn,fp,fn,tp
```

Out[3]:

```
(2, 0, 1, 3)
```

정확도

In [4]:

```
import numpy as np
y_t=np.array(y_t)
y_p=np.array(y_p)
sum(y_t==y_p)/len(y_t)
```

Out[4]:

```
0.8333333333333334
```

In [5]:

```
from sklearn.metrics import accuracy_score
accuracy_score(y_t,y_p)
```

Out[5]:

```
0.8333333333333334
```

In [6]:

```
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
```

In [7]:

```
precision_score(y_t,y_p)
```

Out[7]:

```
1.0
```

In [8]:

```
recall_score(y_t,y_p)
```

Out[8]:

```
0.75
```

In [9]:

```
f1_score(y_t,y_p)
```

Out[9]:

```
0.8571428571428571
```

---

---



```
import pandas as pd
df = pd.read_csv('day5_data1.csv')
df
```

Out[1]:

|       |     who | Newbie |  Age | Gender | Household Income | Sexual Preference |    Country | Education Attainment | Major Occupation | Marital Status | Years on Internet |
| ----: | ------: | -----: | ---: | -----: | ---------------: | ----------------: | ---------: | -------------------: | ---------------: | -------------: | ----------------: |
|     0 | id74364 |      0 | 54.0 |   Male |           $50-74 |          Gay male |    Ontario |         Some College |         Computer |          Other |            4-6 yr |
|     1 | id84505 |      0 | 39.0 | Female |        Over $100 |      Heterosexual |     Sweden |         Professional |            Other |          Other |            1-3 yr |
|     2 | id84509 |      1 | 49.0 | Female |           $40-49 |      Heterosexual | Washington |         Some College |       Management |          Other |        Under 6 mo |
|     3 | id87028 |      1 | 22.0 | Female |           $40-49 |      Heterosexual |    Florida |         Some College |         Computer |        Married |           6-12 mo |
|     4 | id76087 |      0 | 20.0 |   Male |           $30-39 |          Bisexual | New Jersey |         Some College |        Education |         Single |            1-3 yr |
|   ... |     ... |    ... |  ... |    ... |              ... |               ... |        ... |                  ... |              ... |            ... |               ... |
| 19578 | id83400 |      0 | 22.0 |   Male |        Over $100 |      Heterosexual |      Texas |         Some College |        Education |         Single |            4-6 yr |
| 19579 | id72216 |      0 | 19.0 |   Male |              NaN |      Heterosexual | New Jersey |         Some College |        Education |         Single |            4-6 yr |
| 19580 |  id8654 |      0 | 49.0 | Female |           $50-74 |      Heterosexual |   Missouri |             Doctoral |        Education |        Married |            1-3 yr |
| 19581 | id84503 |      1 | 42.0 | Female |           $50-74 |      Heterosexual |   Kentucky |         Some College |            Other |        Married |        Under 6 mo |
| 19582 | id87674 |      0 | 24.0 | Female |           $50-74 |       Transgender | California |              College |         Computer |         Single |            1-3 yr |

19583 rows × 11 columns

2.data 전처리 1.필요 유무에 따른 정리

In [2]:

```
df.pop('who')
df.pop('Country')
df.pop('Years on Internet')
df.dtypes
```

Out[2]:

```
Newbie                    int64
Age                     float64
Gender                   object
Household Income         object
Sexual Preference        object
Education Attainment     object
Major Occupation         object
Marital Status           object
dtype: object
```

In [3]:

```
ck_c=['Gender','Household Income','Sexual Preference',
      'Education Attainment','Major Occupation','Marital Status']
for i in ck_c:
    df[i] = df[i].astype('category')
df.dtypes
```

Out[3]:

```
Newbie                     int64
Age                      float64
Gender                  category
Household Income        category
Sexual Preference       category
Education Attainment    category
Major Occupation        category
Marital Status          category
dtype: object
```

In [4]:

```
df_one_hot=pd.get_dummies(df)
```

2.data 전처리 2.결측치

In [5]:

```
df_one_hot.isnull().sum()
```

Out[5]:

```
Newbie                                 0
Age                                  561
Gender_Female                          0
Gender_Male                            0
Household Income_$10-19                0
Household Income_$20-29                0
Household Income_$30-39                0
Household Income_$40-49                0
Household Income_$50-74                0
Household Income_$75-99                0
Household Income_Over $100             0
Household Income_Under $10             0
Sexual Preference_Bisexual             0
Sexual Preference_Gay male             0
Sexual Preference_Heterosexual         0
Sexual Preference_Lesbian              0
Sexual Preference_Transgender          0
Sexual Preference_na                   0
Education Attainment_College           0
Education Attainment_Doctoral          0
Education Attainment_Grammar           0
Education Attainment_High School       0
Education Attainment_Masters           0
Education Attainment_Other             0
Education Attainment_Professional      0
Education Attainment_Some College      0
Education Attainment_Special           0
Major Occupation_Computer              0
Major Occupation_Education             0
Major Occupation_Management            0
Major Occupation_Other                 0
Major Occupation_Professional          0
Marital Status_Divorced                0
Marital Status_Married                 0
Marital Status_Other                   0
Marital Status_Separated               0
Marital Status_Single                  0
Marital Status_Widowed                 0
dtype: int64
```

In [6]:

```
df_one_hot.loc[pd.isnull(df_one_hot['Age']),'Age']=df_one_hot['Age'].mean()
```

In [15]:

```
df_one_hot
```

Out[15]:

|       | Newbie |  Age | Gender_Female | Gender_Male | Household Income_$10-19 | Household Income_$20-29 | Household Income_$30-39 | Household Income_$40-49 | Household Income_$50-74 | Household Income_$75-99 |  ... | Major Occupation_Education | Major Occupation_Management | Major Occupation_Other | Major Occupation_Professional | Marital Status_Divorced | Marital Status_Married | Marital Status_Other | Marital Status_Separated | Marital Status_Single | Marital Status_Widowed |
| ----: | -----: | ---: | ------------: | ----------: | ----------------------: | ----------------------: | ----------------------: | ----------------------: | ----------------------: | ----------------------: | ---: | -------------------------: | --------------------------: | ---------------------: | ----------------------------: | ----------------------: | ---------------------: | -------------------: | -----------------------: | --------------------: | ---------------------: |
|     0 |      0 | 54.0 |             0 |           1 |                       0 |                       0 |                       0 |                       0 |                       1 |                       0 |  ... |                          0 |                           0 |                      0 |                             0 |                       0 |                      0 |                    1 |                        0 |                     0 |                      0 |
|     1 |      0 | 39.0 |             1 |           0 |                       0 |                       0 |                       0 |                       0 |                       0 |                       0 |  ... |                          0 |                           0 |                      1 |                             0 |                       0 |                      0 |                    1 |                        0 |                     0 |                      0 |
|     2 |      1 | 49.0 |             1 |           0 |                       0 |                       0 |                       0 |                       1 |                       0 |                       0 |  ... |                          0 |                           1 |                      0 |                             0 |                       0 |                      0 |                    1 |                        0 |                     0 |                      0 |
|     3 |      1 | 22.0 |             1 |           0 |                       0 |                       0 |                       0 |                       1 |                       0 |                       0 |  ... |                          0 |                           0 |                      0 |                             0 |                       0 |                      1 |                    0 |                        0 |                     0 |                      0 |
|     4 |      0 | 20.0 |             0 |           1 |                       0 |                       0 |                       1 |                       0 |                       0 |                       0 |  ... |                          1 |                           0 |                      0 |                             0 |                       0 |                      0 |                    0 |                        0 |                     1 |                      0 |
|   ... |    ... |  ... |           ... |         ... |                     ... |                     ... |                     ... |                     ... |                     ... |                     ... |  ... |                        ... |                         ... |                    ... |                           ... |                     ... |                    ... |                  ... |                      ... |                   ... |                    ... |
| 19578 |      0 | 22.0 |             0 |           1 |                       0 |                       0 |                       0 |                       0 |                       0 |                       0 |  ... |                          1 |                           0 |                      0 |                             0 |                       0 |                      0 |                    0 |                        0 |                     1 |                      0 |
| 19579 |      0 | 19.0 |             0 |           1 |                       0 |                       0 |                       0 |                       0 |                       0 |                       0 |  ... |                          1 |                           0 |                      0 |                             0 |                       0 |                      0 |                    0 |                        0 |                     1 |                      0 |
| 19580 |      0 | 49.0 |             1 |           0 |                       0 |                       0 |                       0 |                       0 |                       1 |                       0 |  ... |                          1 |                           0 |                      0 |                             0 |                       0 |                      1 |                    0 |                        0 |                     0 |                      0 |
| 19581 |      1 | 42.0 |             1 |           0 |                       0 |                       0 |                       0 |                       0 |                       1 |                       0 |  ... |                          0 |                           0 |                      1 |                             0 |                       0 |                      1 |                    0 |                        0 |                     0 |                      0 |
| 19582 |      0 | 24.0 |             1 |           0 |                       0 |                       0 |                       0 |                       0 |                       1 |                       0 |  ... |                          0 |                           0 |                      0 |                             0 |                       0 |                      0 |                    0 |                        0 |                     1 |                      0 |

19583 rows × 38 columns

3.입력data정리

In [7]:

```
X=df_one_hot.iloc[:,1:].values
Y=df_one_hot.iloc[:,0].values.reshape(-1,1)
X.shape,Y.shape
```

Out[7]:

```
((19583, 37), (19583, 1))
```

In [8]:

```
from sklearn.preprocessing import MinMaxScaler
m_m_s=MinMaxScaler()
X_data=m_m_s.fit_transform(X)
```

In [9]:

```
from sklearn.model_selection import train_test_split
t_x,tt_x,t_y,tt_y=train_test_split(X_data,Y,test_size=0.3,random_state=42)
```

4.모델 생성 및 학습

In [10]:

```
from sklearn.linear_model import LogisticRegression
lo_g=LogisticRegression(fit_intercept=True)
lo_g.fit(t_x,t_y.flatten())
```

Out[10]:

```
LogisticRegression()
```

5.테스트 및 검증

In [11]:

```
lo_g.predict(tt_x[:5])
```

Out[11]:

```
array([0, 0, 0, 0, 0], dtype=int64)
```

In [12]:

```
lo_g.predict_proba(tt_x[:5])
```

Out[12]:

```
array([[0.56799173, 0.43200827],
       [0.91026881, 0.08973119],
       [0.7911079 , 0.2088921 ],
       [0.85449667, 0.14550333],
       [0.62307651, 0.37692349]])
```

In [13]:

```
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
y_t = tt_y.copy()
y_p = lo_g.predict(tt_x)
confusion_matrix(y_t,y_p)
```

Out[13]:

```
array([[4093,  240],
       [1221,  321]], dtype=int64)
```

In [14]:

```
accuracy_score(y_t,y_p)
```

Out[14]:

```
0.7513191489361702
```

---

---

ovr=>[a,b,c] a!=[b,c] b!=[a,c] c!=[a,b] ovo=>[a,b,c] a!=b a!=c b!=a b!=c c!=a c!=b

In [2]:

```
import numpy as np
def s_max(z):
    a_v=np.exp(z)
    return a_v/sum(a_v)
z=[2,1,5,0.5]
y = s_max(z)
```

Out[2]:

```
1.0
```

minist 데이터셋 이용

In [3]:

```
from sklearn.datasets import load_digits
data = load_digits()
data.keys()
```

Out[3]:

```
dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])
```

In [4]:

```
data['images'].shape
```

Out[4]:

```
(1797, 8, 8)
```

In [5]:

```
data['target'][0]
```

Out[5]:

```
0
```

In [6]:

```
data['images'][0]
```

Out[6]:

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

In [7]:

```
data['data'][0].shape
```

Out[7]:

```
(64,)
```

입력 data분류

In [8]:

```
X=data['data']
Y=data['target']
from sklearn.model_selection import train_test_split
t_x,tt_x,t_y,tt_y = train_test_split(X,Y,random_state=42)
```

In [11]:

```
from sklearn.linear_model import LogisticRegression
lo_g_ovr=LogisticRegression(multi_class='ovr')
lo_g_s_max = LogisticRegression(multi_class='multinomial',solver='sag')
lo_g_ovr.fit(t_x,t_y)
lo_g_s_max.fit(t_x,t_y)

```

Out[11]:

```
LogisticRegression(multi_class='multinomial', solver='sag')
```

In [12]:

```
from sklearn.metrics import confusion_matrix
y_t = tt_y.copy()
y_p = lo_g_ovr.predict(tt_x)
confusion_matrix(y_t,y_p)
```

Out[12]:

```
array([[43,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0, 35,  1,  0,  0,  0,  0,  0,  1,  0],
       [ 0,  0, 38,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0, 44,  0,  1,  0,  0,  1,  0],
       [ 0,  1,  0,  0, 54,  0,  0,  0,  0,  0],
       [ 0,  0,  1,  0,  0, 56,  0,  0,  1,  1],
       [ 0,  0,  0,  0,  0,  1, 44,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0, 40,  0,  1],
       [ 0,  1,  0,  0,  0,  1,  0,  0, 36,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  4, 44]], dtype=int64)
```

In [14]:

```
from sklearn.metrics import classification_report
print(classification_report(y_t,y_p))
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        43
           1       0.95      0.95      0.95        37
           2       0.95      1.00      0.97        38
           3       1.00      0.96      0.98        46
           4       1.00      0.98      0.99        55
           5       0.95      0.95      0.95        59
           6       1.00      0.98      0.99        45
           7       1.00      0.98      0.99        41
           8       0.84      0.95      0.89        38
           9       0.96      0.92      0.94        48

    accuracy                           0.96       450
   macro avg       0.96      0.97      0.96       450
weighted avg       0.97      0.96      0.96       450
```