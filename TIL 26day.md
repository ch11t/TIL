# 머신러닝 day 6

```
import pandas as pd
import numpy as np
data=pd.read_csv('day6_data2.csv')
data
```

Out[1]:

|      |         age | income | student | credit_rating | class_buys_computer |
| ---: | ----------: | -----: | ------: | ------------: | ------------------: |
|    0 |       youth |   high |      no |          fair |                  no |
|    1 |       youth |   high |      no |     excellent |                  no |
|    2 | middle_aged |   high |      no |          fair |                 yes |
|    3 |      senior | medium |      no |          fair |                 yes |
|    4 |      senior |    low |     yes |          fair |                 yes |
|    5 |      senior |    low |     yes |     excellent |                  no |
|    6 | middle_aged |    low |     yes |     excellent |                 yes |
|    7 |       youth | medium |      no |          fair |                  no |
|    8 |       youth |    low |     yes |          fair |                 yes |
|    9 |      senior | medium |     yes |          fair |                 yes |
|   10 |       youth | medium |     yes |     excellent |                 yes |
|   11 | middle_aged | medium |      no |     excellent |                 yes |
|   12 | middle_aged |   high |     yes |          fair |                 yes |
|   13 |      senior | medium |      no |     excellent |                  no |

In [2]:

```
def get_info(df):
    buy = df.loc[df['class_buys_computer']=='yes']
    not_buy = df.loc[df['class_buys_computer']=='no']
    x=np.array([len(buy)/len(df),len(not_buy)/len(df)])
    y=np.log2(x[x!=0])
    info_all = -sum(x[x!=0]*y)
    return info_all
```

전체 엔트로피

In [3]:

```
get_info(data)
```

Out[3]:

```
0.9402859586706311
```

age 속성 정보 계산

In [4]:

```
youth = data.loc[data['age']=='youth']
middle_aged = data.loc[data['age']=='middle_aged']
senior = data.loc[data['age']=='senior']
```

In [5]:

```
get_info(youth)
```

Out[5]:

```
0.9709505944546686
```

In [6]:

```
get_info(middle_aged)
```

Out[6]:

```
-0.0
```

In [7]:

```
get_info(senior)
```

Out[7]:

```
0.9709505944546686
```

In [8]:

```
data['age'].unique()
```

Out[8]:

```
array(['youth', 'middle_aged', 'senior'], dtype=object)
```

In [9]:

```
def get_attribute_info(df,attribute_name):
    att_v = data[attribute_name].unique()
    get_infos = []
    for i in att_v:
        split_df = data.loc[data[attribute_name]==i]
        get_infos.append((len(split_df)/len(df)) * get_info(split_df))
    return sum(get_infos)
```

In [10]:

```
get_attribute_info(data,'age')
```

Out[10]:

```
0.6935361388961918
```

In [11]:

```
get_info(data) - get_attribute_info(data,'age')
```

Out[11]:

```
0.24674981977443933
```

In [12]:

```
get_info(data) - get_attribute_info(data,'income')
```

Out[12]:

```
0.02922256565895487
```

In [13]:

```
get_info(data) - get_attribute_info(data,'student')
```

Out[13]:

```
0.15183550136234159
```

In [14]:

```
get_info(data) - get_attribute_info(data,'credit_rating')
```

Out[14]:

```
0.04812703040826949
```

In [15]:

```
youth = data.loc[data['age']=='youth']
get_info(youth) - get_attribute_info(youth,'income')
```

Out[15]:

```
-1.580026905978025
```

In [16]:

```
get_info(youth) - get_attribute_info(youth,'student')
```

Out[16]:

```
-1.2367106860085422
```

In [17]:

```
get_info(youth) - get_attribute_info(youth,'credit_rating')
```

Out[17]:

```
-1.527094404679944
```

---

---



```
import pandas as pd
import numpy as np
data=pd.read_csv('day6_data1.csv')
data
```

Out[1]:

|      | alcohol | sugar |   pH | class |
| ---: | ------: | ----: | ---: | ----: |
|    0 |     9.4 |   1.9 | 3.51 |   0.0 |
|    1 |     9.8 |   2.6 | 3.20 |   0.0 |
|    2 |     9.8 |   2.3 | 3.26 |   0.0 |
|    3 |     9.8 |   1.9 | 3.16 |   0.0 |
|    4 |     9.4 |   1.9 | 3.51 |   0.0 |
|  ... |     ... |   ... |  ... |   ... |
| 6492 |    11.2 |   1.6 | 3.27 |   1.0 |
| 6493 |     9.6 |   8.0 | 3.15 |   1.0 |
| 6494 |     9.4 |   1.2 | 2.99 |   1.0 |
| 6495 |    12.8 |   1.1 | 3.34 |   1.0 |
| 6496 |    11.8 |   0.8 | 3.26 |   1.0 |

6497 rows × 4 columns

In [2]:

```
X=data[['alcohol','sugar','pH']].to_numpy()
Y=data['class'].to_numpy()
```

In [3]:

```
from sklearn.model_selection import train_test_split
t_x,tt_x,t_y,tt_y = train_test_split(X,Y,random_state=42,test_size=0.2)
```

In [4]:

```
s_t_x,v_t_x,s_t_y,v_t_y = train_test_split(t_x,t_y,random_state=42,test_size=0.2)
```

In [5]:

```
print(f'학습data:{s_t_x.shape}\n태스트data:{v_t_x.shape}\n검증data:{tt_x.shape}')
학습data:(4157, 3)
태스트data:(1040, 3)
검증data:(1300, 3)
```

In [6]:

```
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=42)
dt.fit(s_t_x,s_t_y)
dt.score(s_t_x,s_t_y),dt.score(v_t_x,v_t_y)
```

Out[6]:

```
(0.9971133028626413, 0.864423076923077)
```

In [7]:

```
from sklearn.model_selection import cross_validate
sc=cross_validate(dt,t_x,t_y)
sc
```

Out[7]:

```
{'fit_time': array([0.00719881, 0.00498176, 0.00498533, 0.00398779, 0.00398922]),
 'score_time': array([0.        , 0.00099874, 0.        , 0.00099754, 0.00099754]),
 'test_score': array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])}
```

In [8]:

```
np.mean(sc['test_score'])
```

Out[8]:

```
0.855300214703487
```

In [9]:

```
from sklearn.model_selection import StratifiedKFold
sc1=cross_validate(dt,t_x,t_y,cv=StratifiedKFold())
sc1
#np.mean(sc1['test_score'])
```

Out[9]:

```
{'fit_time': array([0.00598431, 0.00498605, 0.00498557, 0.00498676, 0.00498652]),
 'score_time': array([0.00099683, 0.00099826, 0.        , 0.0009985 , 0.        ]),
 'test_score': array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])}
```

In [10]:

```
sc_ck=StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
sc2=cross_validate(dt,t_x,t_y,cv=sc_ck)
np.mean(sc2['test_score'])
sc2
```

Out[10]:

```
{'fit_time': array([0.00598168, 0.00498629, 0.00498772, 0.00598311, 0.00498652,
        0.00499225, 0.00499034, 0.004987  , 0.00498772, 0.00499153]),
 'score_time': array([0.        , 0.00099683, 0.00099754, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.00099778, 0.00099754]),
 'test_score': array([0.83461538, 0.87884615, 0.85384615, 0.85384615, 0.84615385,
        0.87307692, 0.85961538, 0.85549133, 0.85163776, 0.86705202])}
```

In [11]:

```
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease':[0.0001,0.0002,0.0003,0.0004,0.0005]}
```

In [12]:

```
gs=GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1)
gs.fit(t_x,t_y)
```

Out[12]:

```
GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), n_jobs=-1,
             param_grid={'min_impurity_decrease': [0.0001, 0.0002, 0.0003,
                                                   0.0004, 0.0005]})
```

In [13]:

```
dt=gs.best_estimator_
dt.score(t_x,t_y),dt.score(tt_x,tt_y)
```

Out[13]:

```
(0.9615162593804117, 0.8653846153846154)
```

In [14]:

```
gs.best_params_
```

Out[14]:

```
{'min_impurity_decrease': 0.0001}
```

In [15]:

```
gs.cv_results_['mean_test_score']
```

Out[15]:

```
array([0.86819297, 0.86453617, 0.86492226, 0.86780891, 0.86761605])
```

In [16]:

```
gs.cv_results_
```

Out[16]:

```
{'mean_fit_time': array([0.00718117, 0.00518594, 0.00658107, 0.00518575, 0.0045876 ]),
 'std_fit_time': array([0.00193386, 0.00039883, 0.00214827, 0.00039916, 0.00048829]),
 'mean_score_time': array([0.00159602, 0.00099711, 0.0021934 , 0.00059843, 0.00099707]),
 'std_score_time': array([7.98728820e-04, 5.91739352e-07, 1.93381653e-03, 4.88616714e-04,
        1.09240577e-03]),
 'param_min_impurity_decrease': masked_array(data=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
              mask=[False, False, False, False, False],
        fill_value='?',
             dtype=object),
 'params': [{'min_impurity_decrease': 0.0001},
  {'min_impurity_decrease': 0.0002},
  {'min_impurity_decrease': 0.0003},
  {'min_impurity_decrease': 0.0004},
  {'min_impurity_decrease': 0.0005}],
 'split0_test_score': array([0.86923077, 0.87115385, 0.86923077, 0.86923077, 0.86538462]),
 'split1_test_score': array([0.86826923, 0.86346154, 0.85961538, 0.86346154, 0.86923077]),
 'split2_test_score': array([0.8825794 , 0.87680462, 0.87584216, 0.88161694, 0.8825794 ]),
 'split3_test_score': array([0.86717998, 0.85466795, 0.85081809, 0.84889317, 0.84985563]),
 'split4_test_score': array([0.85370549, 0.85659288, 0.86910491, 0.87584216, 0.87102984]),
 'mean_test_score': array([0.86819297, 0.86453617, 0.86492226, 0.86780891, 0.86761605]),
 'std_test_score': array([0.00915386, 0.00843731, 0.0087452 , 0.01125985, 0.01056953]),
 'rank_test_score': array([1, 5, 4, 2, 3])}
```

In [17]:

```
i=np.argmax(gs.cv_results_['mean_test_score'])
```

In [18]:

```
gs.cv_results_['params'][i]
```

Out[18]:

```
{'min_impurity_decrease': 0.0001}
```

In [19]:

```
params={'max_depth':range(5,20,1),
       'min_impurity_decrease':np.arange(0.0001,0.001,0.0001),
        'min_samples_split':range(2,100,10)
       }
gs1=GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1)
gs1.fit(t_x,t_y)
```

Out[19]:

```
GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), n_jobs=-1,
             param_grid={'max_depth': range(5, 20),
                         'min_impurity_decrease': array([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008,
       0.0009]),
                         'min_samples_split': range(2, 100, 10)})
```

In [20]:

```
gs1.best_params_
```

Out[20]:

```
{'max_depth': 14, 'min_impurity_decrease': 0.0004, 'min_samples_split': 12}
```

In [21]:

```
np.max(gs1.cv_results_['mean_test_score'])
```

Out[21]:

```
0.8683865773302731
```

랜덤 서치

In [22]:

```
from scipy.stats import uniform, randint
```

In [23]:

```
d=randint(0,10)
d.rvs(5)
```

Out[23]:

```
array([5, 3, 0, 4, 5])
```

In [24]:

```
np.unique(d.rvs(1000),return_counts=True)
```

Out[24]:

```
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
 array([ 90,  87, 108,  93, 100, 103, 108, 114,  94, 103], dtype=int64))
```

In [25]:

```
d1=uniform(0,1)
d1.rvs(5)
```

Out[25]:

```
array([0.45827588, 0.620739  , 0.47735397, 0.46901499, 0.93417612])
```

In [26]:

```
params={'max_depth':randint(20,50),
       'min_impurity_decrease':uniform(0.0001,0.001),
        'min_samples_split':randint(2,25),
        'min_samples_leaf':randint(1,25)
       }
```

In [27]:

```
from sklearn.model_selection import RandomizedSearchCV
rs=RandomizedSearchCV(DecisionTreeClassifier(random_state=42),params,n_iter=100
                      ,n_jobs=-1,random_state=42)
rs.fit(t_x,t_y)
```

Out[27]:

```
RandomizedSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                   n_iter=100, n_jobs=-1,
                   param_distributions={'max_depth': <scipy.stats._distn_infrastructure.rv_frozen object at 0x000001D781147100>,
                                        'min_impurity_decrease': <scipy.stats._distn_infrastructure.rv_frozen object at 0x000001D78111B340>,
                                        'min_samples_leaf': <scipy.stats._distn_infrastructure.rv_frozen object at 0x000001D78111BB20>,
                                        'min_samples_split': <scipy.stats._distn_infrastructure.rv_frozen object at 0x000001D7FFF46580>},
                   random_state=42)
```

In [28]:

```
rs.best_params_
```

Out[28]:

```
{'max_depth': 39,
 'min_impurity_decrease': 0.00034102546602601173,
 'min_samples_leaf': 7,
 'min_samples_split': 13}
```



---

---

<div><br class="Apple-interchange-newline">numpy</div>

```
import pandas as pd
import numpy as np
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_id = train_df["PassengerId"].values
test_id = test_df["PassengerId"].values

all_df = train_df.append(test_df).set_index('PassengerId')
all_df["Sex"] = all_df["Sex"].replace({"male":0,"female":1})

all_df["Age"].fillna(
    all_df.groupby("Pclass")["Age"].transform("mean"), inplace=True)
all_df["cabin_count"] = all_df["Cabin"].map(
         lambda x : len(x.split()) if type(x) == str else 0)
def transform_status(x):
    if "Mrs" in x or "Ms" in x:
        return "Mrs"
    elif "Mr" in x:
        return "Mr"
    elif "Miss" in x:
        return "Miss"
    elif "Master" in x:
        return "Master"
    elif "Dr" in x:
        return "Dr"
    elif "Rev" in x:
        return "Rev"
    elif "Col" in x:
        return "Col"
    else:
        return "0"

all_df["social_status"] = all_df["Name"].map(lambda x : transform_status(x))
all_df["social_status"].value_counts()
#all_df[all_df["Embarked"].isnull()]
all_df = all_df.drop([62,830])
train_id =np.delete(train_id, [62-1,830-1])
#all_df[all_df["Fare"].isnull()]
all_df.groupby(["Pclass","Sex"])["Fare"].mean()
all_df.loc[all_df["Fare"].isnull(), "Fare"] = 12.415462
all_df["cabin_type"] = all_df["Cabin"].map(lambda x : x[0] if type(x) == str else "99")
del all_df["Cabin"]
del all_df["Name"]
del all_df["Ticket"]
y = all_df.loc[train_id, "Survived"].values
del all_df["Survived"]
X_df = pd.get_dummies(all_df)
X = X_df.values
from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
X=mm.fit_transform(X)
```

In [6]:

```
y.shape,X.shape
```

Out[6]:

```
((889,), (1307, 27))
```

In [7]:

```
t_x=X[:len(train_id)]
tt_x=X[len(train_id):]
```

In [12]:

```
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
t_l=[]
tt_l=[]
for i in range(3,20):
    dt = DecisionTreeClassifier(min_samples_leaf=i)
    acc = cross_val_score(dt,t_x,y,scoring='accuracy',cv=5).mean()
    t_l.append(accuracy_score(dt.fit(t_x,y).predict(t_x),y))
    tt_l.append(acc)
r = pd.DataFrame(t_l,index=range(3,20),columns=['train'])
r['test']=tt_l
r.plot()
```

Out[12]:

```
<AxesSubplot:>
```

![img]()

---

---

```
import pandas as pd
import numpy as np
data=pd.read_csv('day6_data1.csv')
X=data[['alcohol','sugar','pH']].to_numpy()
Y=data['class'].to_numpy()
from sklearn.model_selection import train_test_split
t_x,tt_x,t_y,tt_y = train_test_split(X,Y,random_state=42,test_size=0.2)
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(random_state=42,n_jobs=-1)
sc=cross_validate(rf,t_x,t_y,return_train_score=True,n_jobs=-1)
np.mean(sc['train_score']),np.mean(sc['test_score'])
```

Out[1]:

```
(0.9973541965122431, 0.8905151032797809)
```

In [2]:

```
rf.fit(t_x,t_y)
#rf.feature_importances_
rf.score(t_x,t_y),rf.score(tt_x,tt_y)
```

Out[2]:

```
(0.996921300750433, 0.8892307692307693)
```

In [3]:

```
rf1= RandomForestClassifier(oob_score=True,random_state=42,n_jobs=-1)
rf1.fit(t_x,t_y)
#rf.feature_importances_
rf1.score(t_x,t_y),rf.score(tt_x,tt_y)
rf1.oob_score_
```

Out[3]:

```
0.8934000384837406
```

In [4]:

```
from sklearn.ensemble import ExtraTreesClassifier
et=ExtraTreesClassifier(random_state=42,n_jobs=-1)
sc=cross_validate(et,t_x,t_y,return_train_score=True,n_jobs=-1)
np.mean(sc['train_score']),np.mean(sc['test_score'])
et.fit(t_x,t_y)
rf.feature_importances_,et.feature_importances_
```

Out[4]:

```
(array([0.23167441, 0.50039841, 0.26792718]),
 array([0.20183568, 0.52242907, 0.27573525]))
```

In [5]:

```
from sklearn.ensemble import GradientBoostingClassifier
gd = GradientBoostingClassifier(random_state=42)
sc=cross_validate(gd,t_x,t_y,return_train_score=True,n_jobs=-1)
np.mean(sc['train_score']),np.mean(sc['test_score'])
```

Out[5]:

```
(0.8881086892152563, 0.8720430147331015)
```

In [6]:

```
gd = GradientBoostingClassifier(random_state=42,n_estimators=500,learning_rate=0.2)
sc=cross_validate(gd,t_x,t_y,return_train_score=True,n_jobs=-1)
np.mean(sc['train_score']),np.mean(sc['test_score'])
```

Out[6]:

```
(0.9464595437171814, 0.8780082549788999)
```