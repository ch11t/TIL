```
from sklearn.datasets import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
```

In [2]:

```
from sklearn.svm import SVC
t1=load_iris()
t1
f=[2,3]
X=t1.data[:,f]
Y=t1.target
m=SVC(kernel='linear',random_state=0)
m.fit(X,Y)

X_min=X[:,0].min() -1
X_max=X[:,0].max() +1
Y_min=X[:,1].min() -1
Y_max=X[:,1].max() +1
XX,YY=np.meshgrid(np.linspace(X_min,X_max,1000),np.linspace(Y_min,Y_max,1000))
ZZ=m.predict(np.c_[XX.ravel(),YY.ravel()]).reshape(XX.shape)
plt.contourf(XX,YY,ZZ)#영역 면으로 구분
plt.contour(XX,YY,ZZ)#영역  선으로 구분
plt.scatter(X[Y== 0,0],X[Y== 0,1] ,s=20,label=t1.target_names[0])
plt.scatter(X[Y== 1,0],X[Y== 1,1] ,s=20,label=t1.target_names[1])
plt.scatter(X[Y== 2,0],X[Y== 2,1] ,s=20,label=t1.target_names[2])
```

Out[2]:

```
<matplotlib.collections.PathCollection at 0x21914059fd0>
```

![img](data:image/png;base64,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)

In [4]:

```
from sklearn.cluster import AffinityPropagation

X , _ = make_blobs(n_features=2, centers=3 ,random_state=1)

m=AffinityPropagation().fit(X)


```

Out[4]:

```
<matplotlib.collections.PathCollection at 0x21913d76490>
```

![img](data:image/png;base64,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)

In [6]:

```
plt.scatter(X[:,0],X[:,1])
for k in range(3):
    c=X[m.cluster_centers_indices_[k]]
    for i in X[m.labels_ == k]:
        plt.plot([c[0],i[0]],[c[1],i[1]])

```

---

---



# 머신러닝



```
import numpy as np
import pandas as pd
data = {
    '이름':["길동",'둘리',np.nan,'또치','희동',np.nan],
    '나이':[40,np.nan,15,np.nan,5,np.nan],
    '성별':['남자',np.nan,'여자','여자','남자',np.nan],
    '시험점수':[np.nan,20,80,10,2,np.nan]
}
df= pd.DataFrame(data,columns=['이름','나이','성별','시험점수'])
df
```

Out[1]:

|      | 이름 | 나이 | 성별 | 시험점수 |
| ---: | ---: | ---: | ---: | -------: |
|    0 | 길동 | 40.0 | 남자 |      NaN |
|    1 | 둘리 |  NaN |  NaN |     20.0 |
|    2 |  NaN | 15.0 | 여자 |     80.0 |
|    3 | 또치 |  NaN | 여자 |     10.0 |
|    4 | 희동 |  5.0 | 남자 |      2.0 |
|    5 |  NaN |  NaN |  NaN |      NaN |

In [2]:

```
df.isnull().sum()
```

Out[2]:

```
이름      2
나이      3
성별      2
시험점수    2
dtype: int64
```

In [3]:

```
df.isnull().sum()/len(df)
```

Out[3]:

```
이름      0.333333
나이      0.500000
성별      0.333333
시험점수    0.333333
dtype: float64
```

In [4]:

```
df2=df.dropna()
df2
```

Out[4]:

|      | 이름 | 나이 | 성별 | 시험점수 |
| ---: | ---: | ---: | ---: | -------: |
|    4 | 희동 |  5.0 | 남자 |      2.0 |

In [5]:

```
df3=df.dropna(how='all')
df3
```

Out[5]:

|      | 이름 | 나이 | 성별 | 시험점수 |
| ---: | ---: | ---: | ---: | -------: |
|    0 | 길동 | 40.0 | 남자 |      NaN |
|    1 | 둘리 |  NaN |  NaN |     20.0 |
|    2 |  NaN | 15.0 | 여자 |     80.0 |
|    3 | 또치 |  NaN | 여자 |     10.0 |
|    4 | 희동 |  5.0 | 남자 |      2.0 |

In [6]:

```
df['수정요망']=np.nan
df.dropna(axis=1,how='all')
```

Out[6]:

|      | 이름 | 나이 | 성별 | 시험점수 |
| ---: | ---: | ---: | ---: | -------: |
|    0 | 길동 | 40.0 | 남자 |      NaN |
|    1 | 둘리 |  NaN |  NaN |     20.0 |
|    2 |  NaN | 15.0 | 여자 |     80.0 |
|    3 | 또치 |  NaN | 여자 |     10.0 |
|    4 | 희동 |  5.0 | 남자 |      2.0 |
|    5 |  NaN |  NaN |  NaN |      NaN |

In [7]:

```
df.dropna(axis=0,thresh=1)
df.dropna(axis=1,thresh=1)
```

Out[7]:

|      | 이름 | 나이 | 성별 | 시험점수 |
| ---: | ---: | ---: | ---: | -------: |
|    0 | 길동 | 40.0 | 남자 |      NaN |
|    1 | 둘리 |  NaN |  NaN |     20.0 |
|    2 |  NaN | 15.0 | 여자 |     80.0 |
|    3 | 또치 |  NaN | 여자 |     10.0 |
|    4 | 희동 |  5.0 | 남자 |      2.0 |
|    5 |  NaN |  NaN |  NaN |      NaN |

In [8]:

```
df4=df.dropna(thresh=2).dropna(axis=1,thresh=1)
df4
```

Out[8]:

|      | 이름 | 나이 | 성별 | 시험점수 |
| ---: | ---: | ---: | ---: | -------: |
|    0 | 길동 | 40.0 | 남자 |      NaN |
|    1 | 둘리 |  NaN |  NaN |     20.0 |
|    2 |  NaN | 15.0 | 여자 |     80.0 |
|    3 | 또치 |  NaN | 여자 |     10.0 |
|    4 | 희동 |  5.0 | 남자 |      2.0 |

In [9]:

```
df4.fillna(0)
```

Out[9]:

|      | 이름 | 나이 | 성별 | 시험점수 |
| ---: | ---: | ---: | ---: | -------: |
|    0 | 길동 | 40.0 | 남자 |      0.0 |
|    1 | 둘리 |  0.0 |    0 |     20.0 |
|    2 |    0 | 15.0 | 여자 |     80.0 |
|    3 | 또치 |  0.0 | 여자 |     10.0 |
|    4 | 희동 |  5.0 | 남자 |      2.0 |

In [10]:

```
df4['이름'].fillna('희선',inplace=True)
df4
```

Out[10]:

|      | 이름 | 나이 | 성별 | 시험점수 |
| ---: | ---: | ---: | ---: | -------: |
|    0 | 길동 | 40.0 | 남자 |      NaN |
|    1 | 둘리 |  NaN |  NaN |     20.0 |
|    2 | 희선 | 15.0 | 여자 |     80.0 |
|    3 | 또치 |  NaN | 여자 |     10.0 |
|    4 | 희동 |  5.0 | 남자 |      2.0 |

In [11]:

```
df4['나이'].fillna(df4['나이'].mean(),inplace=True)
df4
```

Out[11]:

|      | 이름 | 나이 | 성별 | 시험점수 |
| ---: | ---: | ---: | ---: | -------: |
|    0 | 길동 | 40.0 | 남자 |      NaN |
|    1 | 둘리 | 20.0 |  NaN |     20.0 |
|    2 | 희선 | 15.0 | 여자 |     80.0 |
|    3 | 또치 | 20.0 | 여자 |     10.0 |
|    4 | 희동 |  5.0 | 남자 |      2.0 |

In [12]:

```
df4.groupby('성별')['시험점수'].transform('mean')
```

Out[12]:

```
0     2.0
1     NaN
2    45.0
3    45.0
4     2.0
Name: 시험점수, dtype: float64
```

In [15]:

```
df4['시험점수'].fillna(df4.groupby('성별')['시험점수'].transform('mean'),inplace=True)
df4
```

Out[15]:

|      | 이름 | 나이 | 성별 | 시험점수 |
| ---: | ---: | ---: | ---: | -------: |
|    0 | 길동 | 40.0 | 남자 |      2.0 |
|    1 | 둘리 | 20.0 |  NaN |     20.0 |
|    2 | 희선 | 15.0 | 여자 |     80.0 |
|    3 | 또치 | 20.0 | 여자 |     10.0 |
|    4 | 희동 |  5.0 | 남자 |      2.0 |

In [18]:

```
df
```

Out[18]:

|      | 이름 | 나이 | 성별 | 시험점수 | 수정요망 |
| ---: | ---: | ---: | ---: | -------: | -------: |
|    0 | 길동 | 40.0 | 남자 |      NaN |      NaN |
|    1 | 둘리 |  NaN |  NaN |     20.0 |      NaN |
|    2 |  NaN | 15.0 | 여자 |     80.0 |      NaN |
|    3 | 또치 |  NaN | 여자 |     10.0 |      NaN |
|    4 | 희동 |  5.0 | 남자 |      2.0 |      NaN |
|    5 |  NaN |  NaN |  NaN |      NaN |      NaN |

In [19]:

```
ck=df['나이'].fillna(df['나이'].mean())#전체 채우기
ck
```

Out[19]:

```
0    40.0
1    20.0
2    15.0
3    20.0
4     5.0
5    20.0
Name: 나이, dtype: float64
```

In [21]:

```
ck=df['시험점수'].fillna(df4.groupby('성별')['시험점수'].transform('mean'))#특정화 채우기
ck
```

Out[21]:

```
0     2.0
1    20.0
2    80.0
3    10.0
4     2.0
5     NaN
Name: 시험점수, dtype: float64
```

In [30]:

```
end_df=df
end_df.dropna(axis=0,thresh=1,inplace=True)
end_df.dropna(axis=1,thresh=1,inplace=True)
end_df['시험점수'].fillna(end_df.groupby('성별')['시험점수'].transform('mean'),inplace=True)
end_df['나이'].fillna(end_df.groupby('성별')['나이'].transform('mean'),inplace=True)
end_df['이름'].fillna('희선',inplace=True)
end_df.dropna(inplace=True)
end_df
```

Out[30]:

|      | 이름 | 나이 | 성별 | 시험점수 |
| ---: | ---: | ---: | ---: | -------: |
|    0 | 길동 | 40.0 | 남자 |      2.0 |
|    2 | 희선 | 15.0 | 여자 |     80.0 |
|    3 | 또치 | 15.0 | 여자 |     10.0 |
|    4 | 희동 |  5.0 | 남자 |      2.0 |

In [35]:

```
data2=pd.DataFrame({
    '점수':[0,1,2],
    '타겟':[2,3,2],
    '색':['red','blue','red'],
    '크기':[1,5,10]
})
data2
```

Out[35]:

|      | 점수 | 타겟 |   색 | 크기 |
| ---: | ---: | ---: | ---: | ---: |
|    0 |    0 |    2 |  red |    1 |
|    1 |    1 |    3 | blue |    5 |
|    2 |    2 |    2 |  red |   10 |

In [32]:

```
data2.dtypes
```

Out[32]:

```
점수     int64
타겟     int64
색     object
크기     int64
dtype: object
```

In [36]:

```
pd.get_dummies(data2)
```

Out[36]:

|      | 점수 | 타겟 | 크기 | 색_blue | 색_red |
| ---: | ---: | ---: | ---: | ------: | -----: |
|    0 |    0 |    2 |    1 |       0 |      1 |
|    1 |    1 |    3 |    5 |       1 |      0 |
|    2 |    2 |    2 |   10 |       0 |      1 |

In [38]:

```
pd.get_dummies(end_df)
```

Out[38]:

|      | 나이 | 시험점수 | 이름_길동 | 이름_또치 | 이름_희동 | 이름_희선 | 성별_남자 | 성별_여자 |
| ---: | ---: | -------: | --------: | --------: | --------: | --------: | --------: | --------: |
|    0 | 40.0 |      2.0 |         1 |         0 |         0 |         0 |         1 |         0 |
|    2 | 15.0 |     80.0 |         0 |         0 |         0 |         1 |         0 |         1 |
|    3 | 15.0 |     10.0 |         0 |         1 |         0 |         0 |         0 |         1 |
|    4 |  5.0 |      2.0 |         0 |         0 |         1 |         0 |         1 |         0 |

In [39]:

```
pd.get_dummies(end_df['성별'])
```

Out[39]:

|      | 남자 | 여자 |
| ---: | ---: | ---: |
|    0 |    1 |    0 |
|    2 |    0 |    1 |
|    3 |    0 |    1 |
|    4 |    1 |    0 |

In [47]:

```
add_data=pd.get_dummies(end_df[['성별']])
```

In [42]:

```
크기_표쥰 = {1:'아동용',5:'성인여성',10:'남성표준'}
data2['크기_표쥰']=data2['크기'].map(크기_표쥰)
data2
```

Out[42]:

|      | 점수 | 타겟 |   색 | 크기 | 크기_표쥰 |
| ---: | ---: | ---: | ---: | ---: | --------: |
|    0 |    0 |    2 |  red |    1 |    아동용 |
|    1 |    1 |    3 | blue |    5 |  성인여성 |
|    2 |    2 |    2 |  red |   10 |  남성표준 |

In [43]:

```
크기_확인변형=pd.get_dummies(data2['크기_표쥰'])
크기_확인변형
```

Out[43]:

|      | 남성표준 | 성인여성 | 아동용 |
| ---: | -------: | -------: | -----: |
|    0 |        0 |        0 |      1 |
|    1 |        0 |        1 |      0 |
|    2 |        1 |        0 |      0 |

In [44]:

```
data2
```

Out[44]:

|      | 점수 | 타겟 |   색 | 크기 | 크기_표쥰 |
| ---: | ---: | ---: | ---: | ---: | --------: |
|    0 |    0 |    2 |  red |    1 |    아동용 |
|    1 |    1 |    3 | blue |    5 |  성인여성 |
|    2 |    2 |    2 |  red |   10 |  남성표준 |

In [45]:

```
pd.concat([data2,크기_확인변형],axis=1)
```

Out[45]:

|      | 점수 | 타겟 |   색 | 크기 | 크기_표쥰 | 남성표준 | 성인여성 | 아동용 |
| ---: | ---: | ---: | ---: | ---: | --------: | -------: | -------: | -----: |
|    0 |    0 |    2 |  red |    1 |    아동용 |        0 |        0 |      1 |
|    1 |    1 |    3 | blue |    5 |  성인여성 |        0 |        1 |      0 |
|    2 |    2 |    2 |  red |   10 |  남성표준 |        1 |        0 |      0 |

In [57]:

```
add_data=pd.get_dummies(end_df[['성별']])
df2=pd.concat([end_df,add_data],axis=1)
df2
```

Out[57]:

|      | 이름 | 나이 | 성별 | 시험점수 | 성별_남자 | 성별_여자 |
| ---: | ---: | ---: | ---: | -------: | --------: | --------: |
|    0 | 길동 | 40.0 | 남자 |      2.0 |         1 |         0 |
|    2 | 희선 | 15.0 | 여자 |     80.0 |         0 |         1 |
|    3 | 또치 | 15.0 | 여자 |     10.0 |         0 |         1 |
|    4 | 희동 |  5.0 | 남자 |      2.0 |         1 |         0 |

In [49]:

```
bin_data=[0,10,20,50]
b_name=['아동','청소년','성인']
c_data = pd.cut(end_df['나이'],bins=bin_data,labels=b_name)
c_data
```

Out[49]:

```
0     성인
2    청소년
3    청소년
4     아동
Name: 나이, dtype: category
Categories (3, object): ['아동' < '청소년' < '성인']
```

In [58]:

```
data=pd.DataFrame(df2,columns=['나이','시험점수'])
data
```

Out[58]:

|      | 나이 | 시험점수 |
| ---: | ---: | -------: |
|    0 | 40.0 |      2.0 |
|    2 | 15.0 |     80.0 |
|    3 | 15.0 |     10.0 |
|    4 |  5.0 |      2.0 |

In [59]:

```
data['나이']-data['나이'].min()
```

Out[59]:

```
0    35.0
2    10.0
3    10.0
4     0.0
Name: 나이, dtype: float64
```

In [62]:

```
(data['나이']-data['나이'].min())/(data['나이'].max()-data['나이'].min())
```

Out[62]:

```
0    1.000000
2    0.285714
3    0.285714
4    0.000000
Name: 나이, dtype: float64
```

In [63]:

```
(data['시험점수']-data['시험점수'].mean()) /(data['시험점수'].std())
```

Out[63]:

```
0   -0.567957
2    1.492538
3   -0.356624
4   -0.567957
Name: 시험점수, dtype: float64
```



---

---



In [1]:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
```

In [2]:

```
f_l=[i for i in os.listdir() if '.csv' in i]
f_l.reverse()
f_l
```

Out[2]:

```
['train.csv', 'test.csv']
```

In [3]:

```
add_l=[]
for i in f_l:
    add_l.append(pd.read_csv(i))
new_df=pd.concat(add_l)
new_df
```

Out[3]:

|      | PassengerId | Survived | Pclass |                                              Name |    Sex |  Age | SibSp | Parch |             Ticket |     Fare | Cabin | Embarked |
| ---: | ----------: | -------: | -----: | ------------------------------------------------: | -----: | ---: | ----: | ----: | -----------------: | -------: | ----: | -------: |
|    0 |           1 |      0.0 |      3 |                           Braund, Mr. Owen Harris |   male | 22.0 |     1 |     0 |          A/5 21171 |   7.2500 |   NaN |        S |
|    1 |           2 |      1.0 |      1 | Cumings, Mrs. John Bradley (Florence Briggs Th... | female | 38.0 |     1 |     0 |           PC 17599 |  71.2833 |   C85 |        C |
|    2 |           3 |      1.0 |      3 |                            Heikkinen, Miss. Laina | female | 26.0 |     0 |     0 |   STON/O2. 3101282 |   7.9250 |   NaN |        S |
|    3 |           4 |      1.0 |      1 |      Futrelle, Mrs. Jacques Heath (Lily May Peel) | female | 35.0 |     1 |     0 |             113803 |  53.1000 |  C123 |        S |
|    4 |           5 |      0.0 |      3 |                          Allen, Mr. William Henry |   male | 35.0 |     0 |     0 |             373450 |   8.0500 |   NaN |        S |
|  ... |         ... |      ... |    ... |                                               ... |    ... |  ... |   ... |   ... |                ... |      ... |   ... |      ... |
|  413 |        1305 |      NaN |      3 |                                Spector, Mr. Woolf |   male |  NaN |     0 |     0 |          A.5. 3236 |   8.0500 |   NaN |        S |
|  414 |        1306 |      NaN |      1 |                      Oliva y Ocana, Dona. Fermina | female | 39.0 |     0 |     0 |           PC 17758 | 108.9000 |  C105 |        C |
|  415 |        1307 |      NaN |      3 |                      Saether, Mr. Simon Sivertsen |   male | 38.5 |     0 |     0 | SOTON/O.Q. 3101262 |   7.2500 |   NaN |        S |
|  416 |        1308 |      NaN |      3 |                               Ware, Mr. Frederick |   male |  NaN |     0 |     0 |             359309 |   8.0500 |   NaN |        S |
|  417 |        1309 |      NaN |      3 |                          Peter, Master. Michael J |   male |  NaN |     1 |     1 |               2668 |  22.3583 |   NaN |        C |

1309 rows × 12 columns

In [4]:

```
test_d=pd.read_csv('test.csv')
t_d=pd.read_csv('train.csv')

df= pd.concat([t_d,test_d])
df=df.reset_index(drop=True)
df.T
```

Out[4]:

|             |                       0 |                                                 1 |                      2 |                                            3 |                        4 |                5 |                       6 |                              7 |                                                 8 |                                   9 |  ... |                            1299 |                      1300 |                   1301 |                                            1302 |                           1303 |               1304 |                         1305 |                         1306 |                1307 |                     1308 |
| ----------: | ----------------------: | ------------------------------------------------: | ---------------------: | -------------------------------------------: | -----------------------: | ---------------: | ----------------------: | -----------------------------: | ------------------------------------------------: | ----------------------------------: | ---: | ------------------------------: | ------------------------: | ---------------------: | ----------------------------------------------: | -----------------------------: | -----------------: | ---------------------------: | ---------------------------: | ------------------: | -----------------------: |
| PassengerId |                       1 |                                                 2 |                      3 |                                            4 |                        5 |                6 |                       7 |                              8 |                                                 9 |                                  10 |  ... |                            1300 |                      1301 |                   1302 |                                            1303 |                           1304 |               1305 |                         1306 |                         1307 |                1308 |                     1309 |
|    Survived |                     0.0 |                                               1.0 |                    1.0 |                                          1.0 |                      0.0 |              0.0 |                     0.0 |                            0.0 |                                               1.0 |                                 1.0 |  ... |                             NaN |                       NaN |                    NaN |                                             NaN |                            NaN |                NaN |                          NaN |                          NaN |                 NaN |                      NaN |
|      Pclass |                       3 |                                                 1 |                      3 |                                            1 |                        3 |                3 |                       1 |                              3 |                                                 3 |                                   2 |  ... |                               3 |                         3 |                      3 |                                               1 |                              3 |                  3 |                            1 |                            3 |                   3 |                        3 |
|        Name | Braund, Mr. Owen Harris | Cumings, Mrs. John Bradley (Florence Briggs Th... | Heikkinen, Miss. Laina | Futrelle, Mrs. Jacques Heath (Lily May Peel) | Allen, Mr. William Henry | Moran, Mr. James | McCarthy, Mr. Timothy J | Palsson, Master. Gosta Leonard | Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg) | Nasser, Mrs. Nicholas (Adele Achem) |  ... | Riordan, Miss. Johanna Hannah"" | Peacock, Miss. Treasteall | Naughton, Miss. Hannah | Minahan, Mrs. William Edward (Lillian E Thorpe) | Henriksson, Miss. Jenny Lovisa | Spector, Mr. Woolf | Oliva y Ocana, Dona. Fermina | Saether, Mr. Simon Sivertsen | Ware, Mr. Frederick | Peter, Master. Michael J |
|         Sex |                    male |                                            female |                 female |                                       female |                     male |             male |                    male |                           male |                                            female |                              female |  ... |                          female |                    female |                 female |                                          female |                         female |               male |                       female |                         male |                male |                     male |
|         Age |                    22.0 |                                              38.0 |                   26.0 |                                         35.0 |                     35.0 |              NaN |                    54.0 |                            2.0 |                                              27.0 |                                14.0 |  ... |                             NaN |                       3.0 |                    NaN |                                            37.0 |                           28.0 |                NaN |                         39.0 |                         38.5 |                 NaN |                      NaN |
|       SibSp |                       1 |                                                 1 |                      0 |                                            1 |                        0 |                0 |                       0 |                              3 |                                                 0 |                                   1 |  ... |                               0 |                         1 |                      0 |                                               1 |                              0 |                  0 |                            0 |                            0 |                   0 |                        1 |
|       Parch |                       0 |                                                 0 |                      0 |                                            0 |                        0 |                0 |                       0 |                              1 |                                                 2 |                                   0 |  ... |                               0 |                         1 |                      0 |                                               0 |                              0 |                  0 |                            0 |                            0 |                   0 |                        1 |
|      Ticket |               A/5 21171 |                                          PC 17599 |       STON/O2. 3101282 |                                       113803 |                   373450 |           330877 |                   17463 |                         349909 |                                            347742 |                              237736 |  ... |                          334915 |        SOTON/O.Q. 3101315 |                 365237 |                                           19928 |                         347086 |          A.5. 3236 |                     PC 17758 |           SOTON/O.Q. 3101262 |              359309 |                     2668 |
|        Fare |                    7.25 |                                           71.2833 |                  7.925 |                                         53.1 |                     8.05 |           8.4583 |                 51.8625 |                         21.075 |                                           11.1333 |                             30.0708 |  ... |                          7.7208 |                    13.775 |                   7.75 |                                            90.0 |                          7.775 |               8.05 |                        108.9 |                         7.25 |                8.05 |                  22.3583 |
|       Cabin |                     NaN |                                               C85 |                    NaN |                                         C123 |                      NaN |              NaN |                     E46 |                            NaN |                                               NaN |                                 NaN |  ... |                             NaN |                       NaN |                    NaN |                                             C78 |                            NaN |                NaN |                         C105 |                          NaN |                 NaN |                      NaN |
|    Embarked |                       S |                                                 C |                      S |                                            S |                        S |                Q |                       S |                              S |                                                 S |                                   C |  ... |                               Q |                         S |                      Q |                                               Q |                              S |                  S |                            C |                            S |                   S |                        C |

12 rows × 1309 columns

In [5]:

```
n_of_t_d_dataset = df.Survived.notnull().sum()
n_of_t_d_dataset
```

Out[5]:

```
891
```

In [6]:

```
n_of_test_d_dataset = df.Survived.isnull().sum()
n_of_test_d_dataset
```

Out[6]:

```
418
```

In [7]:

```
Y_true=df.pop('Survived')[:n_of_t_d_dataset]
Y_true
```

Out[7]:

```
0      0.0
1      1.0
2      1.0
3      1.0
4      0.0
      ... 
886    0.0
887    1.0
888    0.0
889    1.0
890    0.0
Name: Survived, Length: 891, dtype: float64
```

In [8]:

```
d=df.head(2).T
```

In [9]:

```
df.isnull().sum()/len(df) *100
```

Out[9]:

```
PassengerId     0.000000
Pclass          0.000000
Name            0.000000
Sex             0.000000
Age            20.091673
SibSp           0.000000
Parch           0.000000
Ticket          0.000000
Fare            0.076394
Cabin          77.463713
Embarked        0.152788
dtype: float64
```

In [10]:

```
pd.options.display.float_format='{:.2f}'.format
```

In [11]:

```
df.isnull().sum()/len(df) *100
```

Out[11]:

```
PassengerId    0.00
Pclass         0.00
Name           0.00
Sex            0.00
Age           20.09
SibSp          0.00
Parch          0.00
Ticket         0.00
Fare           0.08
Cabin         77.46
Embarked       0.15
dtype: float64
```

In [12]:

```
df[df['Age'].notnull()].groupby(['Sex'])['Age'].mean()
```

Out[12]:

```
Sex
female   28.69
male     30.59
Name: Age, dtype: float64
```

In [13]:

```
df[df['Age'].notnull()].groupby(['Pclass'])['Age'].mean()
```

Out[13]:

```
Pclass
1   39.16
2   29.51
3   24.82
Name: Age, dtype: float64
```

In [14]:

```
df['Age'].fillna(df.groupby('Pclass')['Age'].transform('mean'),inplace=True)
df.isnull().sum()/len(df) *100
```

Out[14]:

```
PassengerId    0.00
Pclass         0.00
Name           0.00
Sex            0.00
Age            0.00
SibSp          0.00
Parch          0.00
Ticket         0.00
Fare           0.08
Cabin         77.46
Embarked       0.15
dtype: float64
```

In [15]:

```
df.loc[61,'Embarked']='S'
df.loc[829,'Embarked']='S'
df.isnull().sum()/len(df) *100
```

Out[15]:

```
PassengerId    0.00
Pclass         0.00
Name           0.00
Sex            0.00
Age            0.00
SibSp          0.00
Parch          0.00
Ticket         0.00
Fare           0.08
Cabin         77.46
Embarked       0.00
dtype: float64
```

In [16]:

```
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1309 entries, 0 to 1308
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  1309 non-null   int64  
 1   Pclass       1309 non-null   int64  
 2   Name         1309 non-null   object 
 3   Sex          1309 non-null   object 
 4   Age          1309 non-null   float64
 5   SibSp        1309 non-null   int64  
 6   Parch        1309 non-null   int64  
 7   Ticket       1309 non-null   object 
 8   Fare         1308 non-null   float64
 9   Cabin        295 non-null    object 
 10  Embarked     1309 non-null   object 
dtypes: float64(2), int64(4), object(5)
memory usage: 112.6+ KB
```

In [17]:

```
범주형=['PassengerId','Pclass','Name','Sex','Ticket','Cabin','Embarked']
숫자형=['Age','SibSp','Parch','Fare']
```

In [18]:

```
for i in 범주형:
    df[i]=df[i].astype(object)
for i in 숫자형:
    df[i]=df[i].astype(float)
df['SibSp']=df['SibSp'].astype(int)
df['Parch']=df['Parch'].astype(int)
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1309 entries, 0 to 1308
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  1309 non-null   object 
 1   Pclass       1309 non-null   object 
 2   Name         1309 non-null   object 
 3   Sex          1309 non-null   object 
 4   Age          1309 non-null   float64
 5   SibSp        1309 non-null   int32  
 6   Parch        1309 non-null   int32  
 7   Ticket       1309 non-null   object 
 8   Fare         1308 non-null   float64
 9   Cabin        295 non-null    object 
 10  Embarked     1309 non-null   object 
dtypes: float64(2), int32(2), object(7)
memory usage: 102.4+ KB
```

In [19]:

```
def f(ldf, rdf, on, how='inner',index=None):
    if index is True:
        return pd.merge(ldf,rdf,how=how,left_index=True,right_index=True)
    else:
         return pd.merge(ldf,rdf,how=how,on=on)
```

In [20]:

```
one_hot_df=f(
df,pd.get_dummies(df['Sex'],prefix='Sex'),on=None,index=True)
one_hot_df=f(
one_hot_df,pd.get_dummies(df['Pclass'],prefix='Pclass'),on=None,index=True)
one_hot_df=f(
one_hot_df,pd.get_dummies(df['Embarked'],prefix='Embarked'),on=None,index=True)
one_hot_df
```

Out[20]:

|      | PassengerId | Pclass |                                              Name |    Sex |   Age | SibSp | Parch |             Ticket |   Fare | Cabin | Embarked | Sex_female | Sex_male | Pclass_1 | Pclass_2 | Pclass_3 | Embarked_C | Embarked_Q | Embarked_S |
| ---: | ----------: | -----: | ------------------------------------------------: | -----: | ----: | ----: | ----: | -----------------: | -----: | ----: | -------: | ---------: | -------: | -------: | -------: | -------: | ---------: | ---------: | ---------: |
|    0 |           1 |      3 |                           Braund, Mr. Owen Harris |   male | 22.00 |     1 |     0 |          A/5 21171 |   7.25 |   NaN |        S |          0 |        1 |        0 |        0 |        1 |          0 |          0 |          1 |
|    1 |           2 |      1 | Cumings, Mrs. John Bradley (Florence Briggs Th... | female | 38.00 |     1 |     0 |           PC 17599 |  71.28 |   C85 |        C |          1 |        0 |        1 |        0 |        0 |          1 |          0 |          0 |
|    2 |           3 |      3 |                            Heikkinen, Miss. Laina | female | 26.00 |     0 |     0 |   STON/O2. 3101282 |   7.92 |   NaN |        S |          1 |        0 |        0 |        0 |        1 |          0 |          0 |          1 |
|    3 |           4 |      1 |      Futrelle, Mrs. Jacques Heath (Lily May Peel) | female | 35.00 |     1 |     0 |             113803 |  53.10 |  C123 |        S |          1 |        0 |        1 |        0 |        0 |          0 |          0 |          1 |
|    4 |           5 |      3 |                          Allen, Mr. William Henry |   male | 35.00 |     0 |     0 |             373450 |   8.05 |   NaN |        S |          0 |        1 |        0 |        0 |        1 |          0 |          0 |          1 |
|  ... |         ... |    ... |                                               ... |    ... |   ... |   ... |   ... |                ... |    ... |   ... |      ... |        ... |      ... |      ... |      ... |      ... |        ... |        ... |        ... |
| 1304 |        1305 |      3 |                                Spector, Mr. Woolf |   male | 24.82 |     0 |     0 |          A.5. 3236 |   8.05 |   NaN |        S |          0 |        1 |        0 |        0 |        1 |          0 |          0 |          1 |
| 1305 |        1306 |      1 |                      Oliva y Ocana, Dona. Fermina | female | 39.00 |     0 |     0 |           PC 17758 | 108.90 |  C105 |        C |          1 |        0 |        1 |        0 |        0 |          1 |          0 |          0 |
| 1306 |        1307 |      3 |                      Saether, Mr. Simon Sivertsen |   male | 38.50 |     0 |     0 | SOTON/O.Q. 3101262 |   7.25 |   NaN |        S |          0 |        1 |        0 |        0 |        1 |          0 |          0 |          1 |
| 1307 |        1308 |      3 |                               Ware, Mr. Frederick |   male | 24.82 |     0 |     0 |             359309 |   8.05 |   NaN |        S |          0 |        1 |        0 |        0 |        1 |          0 |          0 |          1 |
| 1308 |        1309 |      3 |                          Peter, Master. Michael J |   male | 24.82 |     1 |     1 |               2668 |  22.36 |   NaN |        C |          0 |        1 |        0 |        0 |        1 |          1 |          0 |          0 |

1309 rows × 19 columns

In [21]:

```
Y_true
```

Out[21]:

```
0     0.00
1     1.00
2     1.00
3     1.00
4     0.00
      ... 
886   0.00
887   1.00
888   0.00
889   1.00
890   0.00
Name: Survived, Length: 891, dtype: float64
```

In [28]:

```
ck1=['Sex','Pclass','Embarked']
for i in ck1:
    ck_df=pd.merge(one_hot_df[i],Y_true,left_index=True,right_index=True)
    sns.countplot(x='Survived',hue=i,data=ck_df)
    plt.show()
```

![img](data:image/png;base64,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)

![img](data:image/png;base64,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)

![img](data:image/png;base64,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)

In [32]:

```
ck1=['Sex','Pclass','Embarked']
ck_df2=pd.merge(one_hot_df[ck1],Y_true,left_index=True,right_index=True)
g=sns.catplot(x='Sex',hue='Pclass',col='Survived',kind='count' ,data=ck_df2)
```

![img](data:image/png;base64,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)

In [33]:

```
g=sns.catplot(x='Sex',hue='Embarked',col='Survived',kind='count' ,data=ck_df2)
```

![img](data:image/png;base64,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)

In [34]:

```
one_hot_df.columns.to_list()
```

Out[34]:

```
['PassengerId',
 'Pclass',
 'Name',
 'Sex',
 'Age',
 'SibSp',
 'Parch',
 'Ticket',
 'Fare',
 'Cabin',
 'Embarked',
 'Sex_female',
 'Sex_male',
 'Pclass_1',
 'Pclass_2',
 'Pclass_3',
 'Embarked_C',
 'Embarked_Q',
 'Embarked_S']
```

In [35]:

```
[name.split('_')[0] for name in one_hot_df.columns.to_list()]
```

Out[35]:

```
['PassengerId',
 'Pclass',
 'Name',
 'Sex',
 'Age',
 'SibSp',
 'Parch',
 'Ticket',
 'Fare',
 'Cabin',
 'Embarked',
 'Sex',
 'Sex',
 'Pclass',
 'Pclass',
 'Pclass',
 'Embarked',
 'Embarked',
 'Embarked']
```

In [37]:

```
[name for name in one_hot_df.columns.to_list() 
          if name.split('_')[0] in ck1 
          and '_' in name
         ]+['Sex']
```

Out[37]:

```
['Sex_female',
 'Sex_male',
 'Pclass_1',
 'Pclass_2',
 'Pclass_3',
 'Embarked_C',
 'Embarked_Q',
 'Embarked_S',
 'Sex']
```

In [38]:

```
new_ck_l=[name for name in one_hot_df.columns.to_list() 
          if name.split('_')[0] in ck1 
          and '_' in name
         ]+['Sex']

pt_df=pd.merge(one_hot_df[new_ck_l],Y_true,left_index=True,right_index=True)
pt_df
```

Out[38]:

|      | Sex_female | Sex_male | Pclass_1 | Pclass_2 | Pclass_3 | Embarked_C | Embarked_Q | Embarked_S |    Sex | Survived |
| ---: | ---------: | -------: | -------: | -------: | -------: | ---------: | ---------: | ---------: | -----: | -------: |
|    0 |          0 |        1 |        0 |        0 |        1 |          0 |          0 |          1 |   male |     0.00 |
|    1 |          1 |        0 |        1 |        0 |        0 |          1 |          0 |          0 | female |     1.00 |
|    2 |          1 |        0 |        0 |        0 |        1 |          0 |          0 |          1 | female |     1.00 |
|    3 |          1 |        0 |        1 |        0 |        0 |          0 |          0 |          1 | female |     1.00 |
|    4 |          0 |        1 |        0 |        0 |        1 |          0 |          0 |          1 |   male |     0.00 |
|  ... |        ... |      ... |      ... |      ... |      ... |        ... |        ... |        ... |    ... |      ... |
|  886 |          0 |        1 |        0 |        1 |        0 |          0 |          0 |          1 |   male |     0.00 |
|  887 |          1 |        0 |        1 |        0 |        0 |          0 |          0 |          1 | female |     1.00 |
|  888 |          1 |        0 |        0 |        0 |        1 |          0 |          0 |          1 | female |     0.00 |
|  889 |          0 |        1 |        1 |        0 |        0 |          1 |          0 |          0 |   male |     1.00 |
|  890 |          0 |        1 |        0 |        0 |        1 |          0 |          1 |          0 |   male |     0.00 |

891 rows × 10 columns

In [39]:

```
corr= pt_df.corr()
corr
```

Out[39]:

|            | Sex_female | Sex_male | Pclass_1 | Pclass_2 | Pclass_3 | Embarked_C | Embarked_Q | Embarked_S | Survived |
| ---------: | ---------: | -------: | -------: | -------: | -------: | ---------: | ---------: | ---------: | -------: |
| Sex_female |       1.00 |    -1.00 |     0.10 |     0.06 |    -0.14 |       0.08 |       0.07 |      -0.12 |     0.54 |
|   Sex_male |      -1.00 |     1.00 |    -0.10 |    -0.06 |     0.14 |      -0.08 |      -0.07 |       0.12 |    -0.54 |
|   Pclass_1 |       0.10 |    -0.10 |     1.00 |    -0.29 |    -0.63 |       0.30 |      -0.16 |      -0.16 |     0.29 |
|   Pclass_2 |       0.06 |    -0.06 |    -0.29 |     1.00 |    -0.57 |      -0.13 |      -0.13 |       0.19 |     0.09 |
|   Pclass_3 |      -0.14 |     0.14 |    -0.63 |    -0.57 |     1.00 |      -0.15 |       0.24 |      -0.02 |    -0.32 |
| Embarked_C |       0.08 |    -0.08 |     0.30 |    -0.13 |    -0.15 |       1.00 |      -0.15 |      -0.78 |     0.17 |
| Embarked_Q |       0.07 |    -0.07 |    -0.16 |    -0.13 |     0.24 |      -0.15 |       1.00 |      -0.50 |     0.00 |
| Embarked_S |      -0.12 |     0.12 |    -0.16 |     0.19 |    -0.02 |      -0.78 |      -0.50 |       1.00 |    -0.15 |
|   Survived |       0.54 |    -0.54 |     0.29 |     0.09 |    -0.32 |       0.17 |       0.00 |      -0.15 |     1.00 |

In [40]:

```
sns.set()
ax = sns.heatmap(corr,annot=True,linewidths=.5)
```

---

---

```
import pandas as pd
import numpy as np
data=pd.read_csv('data.csv')
data
```

Out[1]:

|      | Unnamed: 0 | D_length | D_weight |    y |
| ---: | ---------: | -------: | -------: | ---: |
|    0 |          0 |     25.4 |    242.0 |  0.0 |
|    1 |          1 |     26.3 |    290.0 |  0.0 |
|    2 |          2 |     26.5 |    340.0 |  0.0 |
|    3 |          3 |     29.0 |    363.0 |  0.0 |
|    4 |          4 |     29.0 |    430.0 |  0.0 |
|    5 |          5 |     29.7 |    450.0 |  0.0 |
|    6 |          6 |     29.7 |    500.0 |  0.0 |
|    7 |          7 |     30.0 |    390.0 |  0.0 |
|    8 |          8 |     30.0 |    450.0 |  0.0 |
|    9 |          9 |     30.7 |    500.0 |  0.0 |
|   10 |         10 |     31.0 |    475.0 |  0.0 |
|   11 |         11 |     31.0 |    500.0 |  0.0 |
|   12 |         12 |     31.5 |    500.0 |  0.0 |
|   13 |         13 |     32.0 |    340.0 |  0.0 |
|   14 |         14 |     32.0 |    600.0 |  0.0 |
|   15 |         15 |     32.0 |    600.0 |  0.0 |
|   16 |         16 |     33.0 |    700.0 |  0.0 |
|   17 |         17 |     33.0 |    700.0 |  0.0 |
|   18 |         18 |     33.5 |    610.0 |  0.0 |
|   19 |         19 |     33.5 |    650.0 |  0.0 |
|   20 |         20 |     34.0 |    575.0 |  0.0 |
|   21 |         21 |     34.0 |    685.0 |  0.0 |
|   22 |         22 |     34.5 |    620.0 |  0.0 |
|   23 |         23 |     35.0 |    680.0 |  0.0 |
|   24 |         24 |     35.0 |    700.0 |  0.0 |
|   25 |         25 |     35.0 |    725.0 |  0.0 |
|   26 |         26 |     35.0 |    720.0 |  0.0 |
|   27 |         27 |     36.0 |    714.0 |  0.0 |
|   28 |         28 |     36.0 |    850.0 |  0.0 |
|   29 |         29 |     37.0 |   1000.0 |  0.0 |
|   30 |         30 |     38.5 |    920.0 |  0.0 |
|   31 |         31 |     38.5 |    955.0 |  0.0 |
|   32 |         32 |     39.5 |    925.0 |  0.0 |
|   33 |         33 |     41.0 |    975.0 |  0.0 |
|   34 |         34 |     41.0 |    950.0 |  0.0 |
|   35 |         35 |      9.8 |      6.7 |  1.0 |
|   36 |         36 |     10.5 |      7.5 |  1.0 |
|   37 |         37 |     10.6 |      7.0 |  1.0 |
|   38 |         38 |     11.0 |      9.7 |  1.0 |
|   39 |         39 |     11.2 |      9.8 |  1.0 |
|   40 |         40 |     11.3 |      8.7 |  1.0 |
|   41 |         41 |     11.8 |     10.0 |  1.0 |
|   42 |         42 |     11.8 |      9.9 |  1.0 |
|   43 |         43 |     12.0 |      9.8 |  1.0 |
|   44 |         44 |     12.2 |     12.2 |  1.0 |
|   45 |         45 |     12.4 |     13.4 |  1.0 |
|   46 |         46 |     13.0 |     12.2 |  1.0 |
|   47 |         47 |     14.3 |     19.7 |  1.0 |
|   48 |         48 |     15.0 |     19.9 |  1.0 |

In [2]:

```
X = pd.DataFrame(data,columns=['D_length','D_weight'])
Y = pd.DataFrame(data,columns=['y'])
```

In [3]:

```
np_X=np.array(X)
np_Y=np.array(Y['y'], dtype=int)
```

In [4]:

```
np_Y
```

Out[4]:

```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1])
```

In [5]:

```
np.random.seed(10)
ck=np.arange(49)
np.random.shuffle(ck)
ck
```

Out[5]:

```
array([37, 23, 43, 41, 46, 20,  3, 30,  7,  6, 38,  2, 39, 32, 10, 21, 35,
       27, 18, 31,  1, 12, 34, 44, 26,  5, 13, 22, 19, 17, 14,  4, 40, 33,
       24, 11, 16, 47, 45, 48,  8, 42, 29, 25, 28,  0, 15, 36,  9])
```

In [6]:

```
t_x=np_X[ck[:35]]
tt_x=np_X[ck[35:]]
t_y=np_Y[ck[:35]]
tt_y=np_Y[ck[35:]]
```

In [7]:

```
import matplotlib.pyplot as plt
plt.scatter(t_x[:,0],t_x[:,1])
plt.scatter(tt_x[:,0],tt_x[:,1])
```

Out[7]:

```
<matplotlib.collections.PathCollection at 0x26d9da38880>
```

![img](data:image/png;base64,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)

In [8]:

```
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier().fit(t_x,t_y)
```

In [9]:

```
kn.score(tt_x,tt_y)
```

Out[9]:

```
1.0
```

In [10]:

```
kn.predict(tt_x)
```

Out[10]:

```
array([0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0])
```

In [11]:

```
tt_y
```

Out[11]:

```
array([0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0])
```

In [12]:

```
from sklearn.model_selection import train_test_split
sk_t_x, sk_tt_x, sk_t_y, sk_tt_y = train_test_split(X,Y,random_state=10)
sk_t_x
```

Out[12]:

|      | D_length | D_weight |
| ---: | -------: | -------: |
|   32 |     39.5 |    925.0 |
|   10 |     31.0 |    475.0 |
|   21 |     34.0 |    685.0 |
|   35 |      9.8 |      6.7 |
|   27 |     36.0 |    714.0 |
|   18 |     33.5 |    610.0 |
|   31 |     38.5 |    955.0 |
|    1 |     26.3 |    290.0 |
|   12 |     31.5 |    500.0 |
|   34 |     41.0 |    950.0 |
|   44 |     12.2 |     12.2 |
|   26 |     35.0 |    720.0 |
|    5 |     29.7 |    450.0 |
|   13 |     32.0 |    340.0 |
|   22 |     34.5 |    620.0 |
|   19 |     33.5 |    650.0 |
|   17 |     33.0 |    700.0 |
|   14 |     32.0 |    600.0 |
|    4 |     29.0 |    430.0 |
|   40 |     11.3 |      8.7 |
|   33 |     41.0 |    975.0 |
|   24 |     35.0 |    700.0 |
|   11 |     31.0 |    500.0 |
|   16 |     33.0 |    700.0 |
|   47 |     14.3 |     19.7 |
|   45 |     12.4 |     13.4 |
|   48 |     15.0 |     19.9 |
|    8 |     30.0 |    450.0 |
|   42 |     11.8 |      9.9 |
|   29 |     37.0 |   1000.0 |
|   25 |     35.0 |    725.0 |
|   28 |     36.0 |    850.0 |
|    0 |     25.4 |    242.0 |
|   15 |     32.0 |    600.0 |
|   36 |     10.5 |      7.5 |
|    9 |     30.7 |    500.0 |

In [13]:

```
sk_t_x, sk_tt_x, sk_t_y, sk_tt_y = train_test_split(np_X,np_Y,random_state=42)
sk_t_x
```

Out[13]:

```
array([[  30. ,  450. ],
       [  29. ,  363. ],
       [  29.7,  500. ],
       [  11.3,    8.7],
       [  11.8,   10. ],
       [  13. ,   12.2],
       [  32. ,  600. ],
       [  30.7,  500. ],
       [  33. ,  700. ],
       [  35. ,  700. ],
       [  41. ,  975. ],
       [  38.5,  920. ],
       [  25.4,  242. ],
       [  12. ,    9.8],
       [  39.5,  925. ],
       [  29.7,  450. ],
       [  37. , 1000. ],
       [  31. ,  500. ],
       [  10.5,    7.5],
       [  26.3,  290. ],
       [  34. ,  685. ],
       [  26.5,  340. ],
       [  10.6,    7. ],
       [   9.8,    6.7],
       [  35. ,  680. ],
       [  11.2,    9.8],
       [  31. ,  475. ],
       [  34.5,  620. ],
       [  33.5,  610. ],
       [  15. ,   19.9],
       [  34. ,  575. ],
       [  30. ,  390. ],
       [  11.8,    9.9],
       [  32. ,  600. ],
       [  36. ,  850. ],
       [  11. ,    9.7]])
```

In [14]:

```
kn1=KNeighborsClassifier().fit(sk_t_x,sk_t_y)
```

In [15]:

```
kn1.predict([[25,150]])
```

Out[15]:

```
array([1])
```

In [16]:

```
Y_name=np.array(['A','B'])
```

In [17]:

```
plt.scatter(np_X[:35,0],np_X[:35,1])#0
plt.scatter(np_X[35:,0],np_X[35:,1])#1
plt.scatter(25,150,marker='*')
```

Out[17]:

```
<matplotlib.collections.PathCollection at 0x26da06baa60>
```

![img](data:image/png;base64,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)

In [18]:

```
kn=KNeighborsClassifier().fit(t_x,t_y)
d,i=kn.kneighbors([[25,150]])
kn.predict([[25,150]])
```

Out[18]:

```
array([1])
```

In [19]:

```
plt.scatter(t_x[:,0],t_x[:,1])
plt.scatter(25,150,marker='*')
plt.scatter(t_x[i,0],t_x[i,1],marker='o')
```

Out[19]:

```
<matplotlib.collections.PathCollection at 0x26da0736790>
```

![img](data:image/png;base64,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)

In [20]:

```
kn1=KNeighborsClassifier().fit(np_X,np_Y)

plt.scatter(np_X[:35,0],np_X[:35,1])#0
plt.scatter(np_X[35:,0],np_X[35:,1])#1
plt.scatter(25,150,marker='*')
d1,i1=kn1.kneighbors([[25,150]])
plt.scatter(np_X[i1,0],np_X[i1,1],marker='o')
```

Out[20]:

```
<matplotlib.collections.PathCollection at 0x26da07a9880>
```

![img](data:image/png;base64,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)

In [21]:

```
print(np_X[i1])
[[[ 25.4 242. ]
  [ 15.   19.9]
  [ 14.3  19.7]
  [ 12.4  13.4]
  [ 13.   12.2]]]
```

In [22]:

```
print(np_Y[i1])
[[0 1 1 1 1]]
```

In [23]:

```
plt.scatter(np_X[:35,0],np_X[:35,1])#0
plt.scatter(np_X[35:,0],np_X[35:,1])#1
plt.scatter(25,150,marker='*')
plt.xlim((0,1000))
plt.scatter(np_X[i1,0],np_X[i1,1],marker='^')
```

Out[23]:

```
<matplotlib.collections.PathCollection at 0x26da0818f70>
```

![img](data:image/png;base64,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)

In [26]:

```
mean=np.mean(np_X,axis=0)
std=np.std(np_X,axis=0)
```

In [27]:

```
sc_t_X=(np_X-mean)/std
sc_d=([25,150]-mean)/std
```

In [28]:

```
plt.scatter(sc_t_X[:35,0],sc_t_X[:35,1])#0
plt.scatter(sc_t_X[35:,0],sc_t_X[35:,1])#1
plt.scatter(sc_d[0],sc_d[1],marker='*')
```

Out[28]:

```
<matplotlib.collections.PathCollection at 0x26da17eeee0>
```

![img](data:image/png;base64,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)

In [31]:

```
kn2=KNeighborsClassifier().fit(sc_t_X,np_Y)
d,i2=kn2.kneighbors([sc_d])
```

In [33]:

```
kn2.predict([sc_d])
```

Out[33]:

```
array([0])
```

In [34]:

```
plt.scatter(sc_t_X[:35,0],sc_t_X[:35,1])#0
plt.scatter(sc_t_X[35:,0],sc_t_X[35:,1])#1
plt.scatter(sc_d[0],sc_d[1],marker='*')
plt.scatter(sc_t_X[i2,0],sc_t_X[i2,1],marker='o')
```

```
<matplotlib.collections.PathCollection at 0x26da3ef6c40>
```

---

---



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#1.data의 수집
data=pd.read_csv('data.csv')
X = pd.DataFrame(data,columns=['D_length','D_weight'])
Y = pd.DataFrame(data,columns=['y'])
np_X=np.array(X)
np_Y=np.array(Y['y'], dtype=int)
#2.data 전처리
mean=np.mean(np_X,axis=0)
std=np.std(np_X,axis=0)
sc_t_X=(np_X-mean)/std
t_x,tt_x,t_y,tt_y = train_test_split(sc_t_X,np_Y,random_state=10)
#3.모델 생성 및 학습
kn=KNeighborsClassifier().fit(t_x,t_y)
#4.테스트 및 검증
kn.score(tt_x,tt_y)

출력 : 1.0