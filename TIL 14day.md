## 2.Numpy 연산

연산을 통한 결과

In [1]:

```
import numpy as np
t_a=np.arange(1,11)
t_a.sum()
```

Out[1]:

```
55
```

In [2]:

```
t_a=np.arange(1,13).reshape(3,4)
t_a
```

Out[2]:

```
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])
```

고차원에서의 출발로 고차원의 축을 이야기한다.

In [3]:

```
t_a.sum(axis=0) # 2차원의 축
```

Out[3]:

```
array([15, 18, 21, 24])
```

In [4]:

```
t_a.sum(axis=1) # 1차원의 축
```

Out[4]:

```
array([10, 26, 42])
```

In [6]:

```
t_a_2=np.array([t_a,t_a,t_a])
t_a_2
```

Out[6]:

```
array([[[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]],

       [[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]],

       [[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]]])
```

In [7]:

```
t_a_2.shape
```

Out[7]:

```
(3, 3, 4)
```

In [8]:

```
t_a_2.sum(axis=0) # 3차원의 축
```

Out[8]:

```
array([[ 3,  6,  9, 12],
       [15, 18, 21, 24],
       [27, 30, 33, 36]])
```

In [9]:

```
t_a_2.sum(axis=1) # 2차원의 축
```

Out[9]:

```
array([[15, 18, 21, 24],
       [15, 18, 21, 24],
       [15, 18, 21, 24]])
```

In [10]:

```
t_a_2.sum(axis=2) # 1차원의 축
```

Out[10]:

```
array([[10, 26, 42],
       [10, 26, 42],
       [10, 26, 42]])
```

In [14]:

```
v_a=np.array([[1,2,3],[1,2,3]])
v_a_1=np.array([4,5,6])
np.vstack((v_a,v_a_1)).T # 뒤집어짐
```

Out[14]:

```
array([[1, 1, 4],
       [2, 2, 5],
       [3, 3, 6]])
```

In [18]:

```
v_a=np.array([1,2,3])
h_1=v_a.reshape(-1,1)
h_2=v_a_1.reshape(-1,1)
np.hstack((h_1,h_2))
```

Out[18]:

```
array([[1, 4],
       [2, 5],
       [3, 6]])
```

In [19]:

```
x=np.arange(1,7).reshape(2,3)
x
```

Out[19]:

```
array([[1, 2, 3],
       [4, 5, 6]])
```

In [20]:

```
x+x
```

Out[20]:

```
array([[ 2,  4,  6],
       [ 8, 10, 12]])
```

In [21]:

```
x-x
```

Out[21]:

```
array([[0, 0, 0],
       [0, 0, 0]])
```

In [22]:

```
x/x
```

Out[22]:

```
array([[1., 1., 1.],
       [1., 1., 1.]])
```

In [23]:

```
x*x
```

Out[23]:

```
array([[ 1,  4,  9],
       [16, 25, 36]])
```

In [24]:

```
x%x
```

Out[24]:

```
array([[0, 0, 0],
       [0, 0, 0]], dtype=int32)
```

In [29]:

```
heights=[1.83,1.76,1.69,1.86,1.77,1.73]
weights=[86, 74, 59, 95, 80, 68]
np_heights=np.array(heights)
np_weights=np.array(weights)
bmi=np_weights/(np_heights**2)
bmi
```

Out[29]:

```
array([25.68007405, 23.88946281, 20.65754   , 27.45982194, 25.53544639,
       22.72043837])
```

In [30]:

```
y=np.arange(1,7).reshape(2,3)
y=np.arange(1,7).reshape(3,2)
np.dot(x,y)
x.dot(y)
```

Out[30]:

```
array([[22, 28],
       [49, 64]])
```

In [31]:

```
x+1
```

Out[31]:

```
array([[2, 3, 4],
       [5, 6, 7]])
```

In [32]:

```
x=np.arange(1,13).reshape(4,3)
x
```

Out[32]:

```
array([[ 1,  2,  3],
       [ 4,  5,  6],
       [ 7,  8,  9],
       [10, 11, 12]])
```

In [33]:

```
v=np.array([10,20,30])
v
```

Out[33]:

```
array([10, 20, 30])
```

In [34]:

```
x+v
```

Out[34]:

```
array([[11, 22, 33],
       [14, 25, 36],
       [17, 28, 39],
       [20, 31, 42]])
```

In [37]:

```
v2=np.array([1,2,3,4]).reshape(-1,1)
v2
```

Out[37]:

```
array([[1],
       [2],
       [3],
       [4]])
```

In [38]:

```
x+v2
```

Out[38]:

```
array([[ 2,  3,  4],
       [ 6,  7,  8],
       [10, 11, 12],
       [14, 15, 16]])
```

In [39]:

```
v+v2
```

Out[39]:

```
array([[11, 21, 31],
       [12, 22, 32],
       [13, 23, 33],
       [14, 24, 34]])
```

In [40]:

```
x=np.arange(3,10)
x
```

Out[40]:

```
array([3, 4, 5, 6, 7, 8, 9])
```

In [41]:

```
(x>5)
```

Out[41]:

```
array([False, False, False,  True,  True,  True,  True])
```

In [42]:

```
(x>5).all()
```

Out[42]:

```
False
```

In [43]:

```
(x>5).any()
```

Out[43]:

```
True
```

In [44]:

```
(x<1).all()
```

Out[44]:

```
False
```

In [45]:

```
(x<1).any()
```

Out[45]:

```
False
```

In [46]:

```
(x>2).all()
```

Out[46]:

```
True
```

In [47]:

```
(x>2).any()
```

Out[47]:

```
True
```

In [49]:

```
# x=np.random.randint()
```

In [50]:

```
score=np.random.randint(0,100,100)
```

In [54]:

```
if np.mean(score)>=50:
    print("전원통과")
```

In [ ]:

```
?
```

In [ ]:

```
?
```

In [59]:

```
x=np.array([1,2,3,4,5,1])
x==1
```

Out[59]:

```
array([ True, False, False, False, False,  True])
```

In [60]:

```
np.where(x==1)
```

Out[60]:

```
(array([0, 5], dtype=int64),)
```

In [61]:

```
d=np.array([1,5,3,7,2,8,4]) 
np.argsort(d)#작은 순서부터 인덱스 출력
```

Out[61]:

```
array([0, 4, 2, 6, 1, 3, 5], dtype=int64)
```

In [62]:

```
np.argmax(d) # 가장 큰 값 출력
```

Out[62]:

```
5
```

In [63]:

```
np.argmin(d) # 가장 작은 값 출력
```

Out[63]:

```
0
```

In [64]:

```
d>5
```

Out[64]:

```
array([False, False, False,  True, False,  True, False])
```

In [65]:

```
d[d>=5]
```

Out[65]:

```
array([5, 7, 8])
```

In [66]:

```
type(d>=5)
```

Out[66]:

```
numpy.ndarray
```

In [67]:

```
(d>=5).shape
```

Out[67]:

```
(7,)
```

In [68]:

```
d.shape
```

Out[68]:

```
(7,)
```

In [69]:

```
x=np.arange(1,11)
x
```

Out[69]:

```
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
```

In [70]:

```
l=np.array([0,5,9])
x[l] # 해당 인덱스
```

Out[70]:

```
array([ 1,  6, 10])
```

In [71]:

```
l=np.array([0,5,9,0,0,0])
x[l]
```

Out[71]:

```
array([ 1,  6, 10,  1,  1,  1])
```

In [74]:

```
x=x.reshape(2,-1)
x
```

Out[74]:

```
array([[ 1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10]])
```

In [77]:

```
i_1=np.array([0,1])
i_1
```

Out[77]:

```
array([0, 1])
```

In [80]:

```
i_0=np.array([1,0])
i_0
```

Out[80]:

```
array([1, 0])
```

In [83]:

```
x[i_0,i_1]
```

Out[83]:

```
array([6, 2])
```

In [84]:

```
x
```

Out[84]:

```
array([[ 1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10]])
```

In [86]:

```
x_l=np.array([x,x,x])
x_l
```

Out[86]:

```
array([[[ 1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10]],

       [[ 1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10]],

       [[ 1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10]]])
```

In [90]:

```
x_l[:,:1,::2]
```

Out[90]:

```
array([[[1, 3, 5]],

       [[1, 3, 5]],

       [[1, 3, 5]]])
```

In [92]:

```
x_l[:,:,:4]
```

Out[92]:

```
array([[[1, 2, 3, 4],
        [6, 7, 8, 9]],

       [[1, 2, 3, 4],
        [6, 7, 8, 9]],

       [[1, 2, 3, 4],
        [6, 7, 8, 9]]])
```

---

## 3. 판다스

판다스 시리즈

In [3]:

```
import pandas as pd # pandas 호출
import numpy as np
from pandas import Series, DataFrame

data_l=[1,2,3,4,5]
e_obj=Series([data_l])
e_obj
```

Out[3]:

```
0    [1, 2, 3, 4, 5]
dtype: object
```

In [4]:

```
idx_l=[1,2,3,4,1]
e_obj=Series(data=data_l,index=idx_l)
e_obj
```

Out[4]:

```
1    1
2    2
3    3
4    4
1    5
dtype: int64
```

In [5]:

```
idx_l=['a','b','c','d','e']
e_obj=Series(data=data_l,index=idx_l)
e_obj
```

Out[5]:

```
a    1
b    2
c    3
d    4
e    5
dtype: int64
```

In [28]:

```
e_obj=Series(data=data_l,index=idx_l)
e_obj
```

Out[28]:

```
a    1
b    2
c    3
d    4
e    5
dtype: int64
```

In [29]:

```
e_obj.index
```

Out[29]:

```
Index(['a', 'b', 'c', 'd', 'e'], dtype='object')
```

In [30]:

```
e_obj.values
```

Out[30]:

```
array([1, 2, 3, 4, 5], dtype=int64)
```

In [31]:

```
type(e_obj.values)
```

Out[31]:

```
numpy.ndarray
```

In [32]:

```
e_obj.name="data_s"
e_obj.index.name="id"
e_obj
```

Out[32]:

```
id
a    1
b    2
c    3
d    4
e    5
Name: data_s, dtype: int64
```

In [35]:

```
dic={'a':1,"b":2,"c":3,"d":4,"e":5}
e_obj=Series(dic,dtype=np.float32,name="data_s")
e_obj
```

Out[35]:

```
a    1.0
b    2.0
c    3.0
d    4.0
e    5.0
Name: data_s, dtype: float32
```

In [39]:

```
idx_d=['a','b','c','d','e','f','h']
dic={'a':1,"b":2,"c":3,"d":4,"e":5,'g':6}
e_obj=Series(dic,index=idx_d)
e_obj
```

Out[39]:

```
a    1.0
b    2.0
c    3.0
d    4.0
e    5.0
f    NaN
h    NaN
dtype: float64
```

In [57]:

```
data={
    '이름':["홍길동","도우너","둘리"],
    '계좌번호':["1234","4321","4567"],
    '금액':[10000,100,500]
}
e_obj=DataFrame(data,index=["1번고객","2번고객","3번고객"]) #2차원
e_obj.index.name='고객번호'
e_obj.reset_index()

```

Out[57]:

|      | 고객번호 |   이름 | 계좌번호 |  금액 |
| ---: | -------: | -----: | -------: | ----: |
|    0 |  1번고객 | 홍길동 |     1234 | 10000 |
|    1 |  2번고객 | 도우너 |     4321 |   100 |
|    2 |  3번고객 |   둘리 |     4567 |   500 |

In [59]:

```
e_obj.reset_index(drop=True,inplace=True)
e_obj
```

Out[59]:

|      |   이름 | 계좌번호 |  금액 |
| ---: | -----: | -------: | ----: |
|    0 | 홍길동 |     1234 | 10000 |
|    1 | 도우너 |     4321 |   100 |
|    2 |   둘리 |     4567 |   500 |

In [62]:

```
e_obj.set_index('이름',inplace=True)
e_obj.sort_index(ascending=False)
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
~\AppData\Local\Temp/ipykernel_11544/4248934613.py in <module>
----> 1 e_obj.set_index('이름',inplace=True)
      2 e_obj.sort_index(ascending=False)

~\anaconda3\lib\site-packages\pandas\util\_decorators.py in wrapper(*args, **kwargs)
    309                     stacklevel=stacklevel,
    310                 )
--> 311             return func(*args, **kwargs)
    312 
    313         return wrapper

~\anaconda3\lib\site-packages\pandas\core\frame.py in set_index(self, keys, drop, append, inplace, verify_integrity)
   5449 
   5450         if missing:
-> 5451             raise KeyError(f"None of {missing} are in the columns")
   5452 
   5453         if inplace:

KeyError: "None of ['이름'] are in the columns"
```

In [64]:

```
df_data=pd.read_csv('housing.data',sep='\s+',header=None) # header : 타이틀 

df=pd.DataFrame(df_data)
df
```

Out[64]:

|      |       0 |    1 |     2 |    3 |     4 |     5 |    6 |      7 |    8 |     9 |   10 |     11 |   12 |   13 |
| ---: | ------: | ---: | ----: | ---: | ----: | ----: | ---: | -----: | ---: | ----: | ---: | -----: | ---: | ---: |
|    0 | 0.00632 | 18.0 |  2.31 |    0 | 0.538 | 6.575 | 65.2 | 4.0900 |    1 | 296.0 | 15.3 | 396.90 | 4.98 | 24.0 |
|    1 | 0.02731 |  0.0 |  7.07 |    0 | 0.469 | 6.421 | 78.9 | 4.9671 |    2 | 242.0 | 17.8 | 396.90 | 9.14 | 21.6 |
|    2 | 0.02729 |  0.0 |  7.07 |    0 | 0.469 | 7.185 | 61.1 | 4.9671 |    2 | 242.0 | 17.8 | 392.83 | 4.03 | 34.7 |
|    3 | 0.03237 |  0.0 |  2.18 |    0 | 0.458 | 6.998 | 45.8 | 6.0622 |    3 | 222.0 | 18.7 | 394.63 | 2.94 | 33.4 |
|    4 | 0.06905 |  0.0 |  2.18 |    0 | 0.458 | 7.147 | 54.2 | 6.0622 |    3 | 222.0 | 18.7 | 396.90 | 5.33 | 36.2 |
|  ... |     ... |  ... |   ... |  ... |   ... |   ... |  ... |    ... |  ... |   ... |  ... |    ... |  ... |  ... |
|  501 | 0.06263 |  0.0 | 11.93 |    0 | 0.573 | 6.593 | 69.1 | 2.4786 |    1 | 273.0 | 21.0 | 391.99 | 9.67 | 22.4 |
|  502 | 0.04527 |  0.0 | 11.93 |    0 | 0.573 | 6.120 | 76.7 | 2.2875 |    1 | 273.0 | 21.0 | 396.90 | 9.08 | 20.6 |
|  503 | 0.06076 |  0.0 | 11.93 |    0 | 0.573 | 6.976 | 91.0 | 2.1675 |    1 | 273.0 | 21.0 | 396.90 | 5.64 | 23.9 |
|  504 | 0.10959 |  0.0 | 11.93 |    0 | 0.573 | 6.794 | 89.3 | 2.3889 |    1 | 273.0 | 21.0 | 393.45 | 6.48 | 22.0 |
|  505 | 0.04741 |  0.0 | 11.93 |    0 | 0.573 | 6.030 | 80.8 | 2.5050 |    1 | 273.0 | 21.0 | 396.90 | 7.88 | 11.9 |

506 rows × 14 columns

In [67]:

```
data={
    '이름':["홍길동","도우너","둘리",'고길동'],
    '계좌번호':["1234","4321","4567","7894"],
    '금액':[10000,100,500,50000],
    '은행':["국민","하나","우리",'신한']
}
df=DataFrame(data,index=["1번고객","2번고객","3번고객","4번고객"]) #2차원
df
```

Out[67]:

|         |   이름 | 계좌번호 |  금액 | 은행 |
| ------: | -----: | -------: | ----: | ---: |
| 1번고객 | 홍길동 |     1234 | 10000 | 국민 |
| 2번고객 | 도우너 |     4321 |   100 | 하나 |
| 3번고객 |   둘리 |     4567 |   500 | 우리 |
| 4번고객 | 고길동 |     7894 | 50000 | 신한 |

## CSV 파일로 저장

In [68]:

```
df.to_csv("data_1.csv",encoding='utf-8-sig')#index포함하여 저장
```

In [69]:

```
df.to_csv("data_2.csv",encoding='utf-8-sig',index=False) #index 제외하여 저장
```

## 텍스트(.txt)저장

In [72]:

```
df.to_csv('data3.txt' ,sep='\t')
```

## 엑셀파일 저장

In [73]:

```
df.to_excel("data4.xlsx")
```

## 열기

In [81]:

```
df=pd.read_csv('data_2.csv',skiprows=[0,1],header=None)
df
```

Out[81]:

|      |      0 |    1 |     2 |    3 |
| ---: | -----: | ---: | ----: | ---: |
|    0 | 도우너 | 4321 |   100 | 하나 |
|    1 |   둘리 | 4567 |   500 | 우리 |
|    2 | 고길동 | 7894 | 50000 | 신한 |

In [82]:

```
df=pd.read_csv('data_2.csv',skiprows=[1,3],nrows=2)
df
```

Out[82]:

|      |   이름 | 계좌번호 |  금액 | 은행 |
| ---: | -----: | -------: | ----: | ---: |
|    0 | 도우너 |     4321 |   100 | 하나 |
|    1 | 고길동 |     7894 | 50000 | 신한 |

In [84]:

```
df=pd.read_csv("data3.txt",sep='\t',index_col='이름')
df
```

Out[84]:

|        | Unnamed: 0 | 계좌번호 |  금액 | 은행 |
| -----: | ---------: | -------: | ----: | ---: |
|   이름 |            |          |       |      |
| 홍길동 |    1번고객 |     1234 | 10000 | 국민 |
| 도우너 |    2번고객 |     4321 |   100 | 하나 |
|   둘리 |    3번고객 |     4567 |   500 | 우리 |
| 고길동 |    4번고객 |     7894 | 50000 | 신한 |

In [85]:

```
df=pd.read_csv("data3.txt",sep='\t')
df.set_index('이름',inplace=True)
df
```

Out[85]:

|        | Unnamed: 0 | 계좌번호 |  금액 | 은행 |
| -----: | ---------: | -------: | ----: | ---: |
|   이름 |            |          |       |      |
| 홍길동 |    1번고객 |     1234 | 10000 | 국민 |
| 도우너 |    2번고객 |     4321 |   100 | 하나 |
|   둘리 |    3번고객 |     4567 |   500 | 우리 |
| 고길동 |    4번고객 |     7894 | 50000 | 신한 |

In [87]:

```
df=pd.read_excel('data4.xlsx',index_col='이름')
df
```

Out[87]:

|        | Unnamed: 0 | 계좌번호 |  금액 | 은행 |
| -----: | ---------: | -------: | ----: | ---: |
|   이름 |            |          |       |      |
| 홍길동 |    1번고객 |     1234 | 10000 | 국민 |
| 도우너 |    2번고객 |     4321 |   100 | 하나 |
|   둘리 |    3번고객 |     4567 |   500 | 우리 |
| 고길동 |    4번고객 |     7894 | 50000 | 신한 |

In [89]:

```
data={
    '이름':["홍길동","도우너","둘리",'고길동'],
    '계좌번호':["1234","4321","4567","7894"],
    '금액':[10000,100,500,50000],
    '은행':["국민","하나","우리",'신한']
}
df2=DataFrame(data,columns=['이름','은행','적금유무']) #2차원
df2
```

Out[89]:

|      |   이름 | 은행 | 적금유무 |
| ---: | -----: | ---: | -------: |
|    0 | 홍길동 | 국민 |      NaN |
|    1 | 도우너 | 하나 |      NaN |
|    2 |   둘리 | 우리 |      NaN |
|    3 | 고길동 | 신한 |      NaN |

In [98]:

```
data={
    '이름':["홍길동","도우너","둘리",'고길동'],
    '계좌번호':["1234","4321","4567","7894"],
    '금액':[10000,100,500,50000],
    '은행':["국민","하나","우리",'신한']
}
df_t=DataFrame(data)
df_t[:2]
#df2=DataFrame(data,columns=['이름','은행','적금유무']) #2차원
#ex_df.head(3).T
```

Out[98]:

|      |   이름 | 계좌번호 |  금액 | 은행 |
| ---: | -----: | -------: | ----: | ---: |
|    0 | 홍길동 |     1234 | 10000 | 국민 |
|    1 | 도우너 |     4321 |   100 | 하나 |

In [102]:

```
df_t.index=df_t['이름']
del df_t['이름']
df_t.loc['도우너':,['금액']]
```

Out[102]:

|        |  금액 |
| -----: | ----: |
|   이름 |       |
| 도우너 |   100 |
|   둘리 |   500 |
| 고길동 | 50000 |

In [ ]:

```
data={
    '이름':["홍길동","도우너","둘리",'고길동'],
    '계좌번호':["1234","4321","4567","7894"],
    '금액':[10000,100,500,50000],
    '은행':["국민","하나","우리",'신한']
}
df_t=DataFrame(data)
df_t.index=df_t['이름']
del df_t['금액']
df_t.iloc[:2,:2]
```

---

# 판다스

In [5]:

```
import pandas as pd
from pandas import DataFrame
df=pd.read_csv('경찰청 강원도경찰청_음주교통사고 발생 현황_20201231.csv',encoding="euc-kr")
df1=DataFrame(df,columns=["연도","발생"])
df1
```

Out[5]:

|      | 연도 | 발생 |
| ---: | ---: | ---: |
|    0 | 2020 |  620 |
|    1 | 2019 |  493 |
|    2 | 2018 |  680 |
|    3 | 2017 |  780 |
|    4 | 2016 |  708 |

In [7]:

```
df2=df.set_index('연도')
df2
```

Out[7]:

|      | 발생 | 사망 | 부상 |
| ---: | ---: | ---: | ---: |
| 연도 |      |      |      |
| 2020 |  620 |   11 | 1053 |
| 2019 |  493 |   18 |  797 |
| 2018 |  680 |   14 | 1165 |
| 2017 |  780 |   18 | 1338 |
| 2016 |  708 |   18 | 1266 |

In [8]:

```
df2.iloc[:2,:2] # 축별 내용을 인덱스로 취급하여 동작 가능하도록 설정하는 함수
```

Out[8]:

|      | 발생 | 사망 |
| ---: | ---: | ---: |
| 연도 |      |      |
| 2020 |  620 |   11 |
| 2019 |  493 |   18 |

In [9]:

```
df.loc[0] # 인덱스를 통한 data 호출이 가능하도록 설정하는 함수(모든 data가 적용될 수 있다.)
```

Out[9]:

```
연도    2020
발생     620
사망      11
부상    1053
Name: 0, dtype: int64
```

In [12]:

```
df.index=df['연도']
df.head()
```

Out[12]:

|      | 연도 | 발생 | 사망 | 부상 |
| ---: | ---: | ---: | ---: | ---: |
| 연도 |      |      |      |      |
| 2020 | 2020 |  620 |   11 | 1053 |
| 2019 | 2019 |  493 |   18 |  797 |
| 2018 | 2018 |  680 |   14 | 1165 |
| 2017 | 2017 |  780 |   18 | 1338 |
| 2016 | 2016 |  708 |   18 | 1266 |

In [13]:

```
del df['연도']
df.head()

```

Out[13]:

|      | 발생 | 사망 | 부상 |
| ---: | ---: | ---: | ---: |
| 연도 |      |      |      |
| 2020 |  620 |   11 | 1053 |
| 2019 |  493 |   18 |  797 |
| 2018 |  680 |   14 | 1165 |
| 2017 |  780 |   18 | 1338 |
| 2016 |  708 |   18 | 1266 |

In [14]:

```
df.loc[2020]
```

Out[14]:

```
발생     620
사망      11
부상    1053
Name: 2020, dtype: int64
```

In [17]:

```
n_df=df.reset_index()
print(int(n_df.loc[0,'연도']))
2020
```

In [22]:

```
print(list(n_df.loc[0]))
n_df.drop(1)
[0, 0, 2020, 620, 11, 1053]
```

Out[22]:

|      | level_0 | index | 연도 | 발생 | 사망 | 부상 |
| ---: | ------: | ----: | ---: | ---: | ---: | ---: |
|    0 |       0 |     0 | 2020 |  620 |   11 | 1053 |
|    2 |       2 |     2 | 2018 |  680 |   14 | 1165 |
|    3 |       3 |     3 | 2017 |  780 |   18 | 1338 |
|    4 |       4 |     4 | 2016 |  708 |   18 | 1266 |

In [23]:

```
n_df.drop(1)
```

Out[23]:

|      | level_0 | index | 연도 | 발생 | 사망 | 부상 |
| ---: | ------: | ----: | ---: | ---: | ---: | ---: |
|    0 |       0 |     0 | 2020 |  620 |   11 | 1053 |
|    2 |       2 |     2 | 2018 |  680 |   14 | 1165 |
|    3 |       3 |     3 | 2017 |  780 |   18 | 1338 |
|    4 |       4 |     4 | 2016 |  708 |   18 | 1266 |

Type *Markdown* and LaTeX: 𝛼2

---

# 3. 데이터 그룹처리

In [3]:

```
import numpy as np
import pandas as pd

ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings','kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
'Year': [2014, 2015, 2014, 2015, 2014, 2015, 2016, 2017, 2016, 2014, 2015, 2017],
'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df=pd.DataFrame(ipl_data)
df
```

Out[3]:

|      |   Team | Rank | Year | Points |
| ---: | -----: | ---: | ---: | -----: |
|    0 | Riders |    1 | 2014 |    876 |
|    1 | Riders |    2 | 2015 |    789 |
|    2 | Devils |    2 | 2014 |    863 |
|    3 | Devils |    3 | 2015 |    673 |
|    4 |  Kings |    3 | 2014 |    741 |
|    5 |  kings |    4 | 2015 |    812 |
|    6 |  Kings |    1 | 2016 |    756 |
|    7 |  Kings |    1 | 2017 |    788 |
|    8 | Riders |    2 | 2016 |    694 |
|    9 | Royals |    4 | 2014 |    701 |
|   10 | Royals |    1 | 2015 |    804 |
|   11 | Riders |    2 | 2017 |    690 |

In [6]:

```
df.groupby("Team")['Points'].sum()
```

Out[6]:

```
Team
Devils    1536
Kings     2285
Riders    3049
Royals    1505
kings      812
Name: Points, dtype: int64
```

In [8]:

```
c_df=df.groupby(["Team",'Year'])['Points'].sum()
```

In [9]:

```
c_df['Devils':'Kings']
```

Out[9]:

```
Team    Year
Devils  2014    863
        2015    673
Kings   2014    741
        2016    756
        2017    788
Name: Points, dtype: int64
```

In [10]:

```
c_df.unstack()
```

Out[10]:

|   Year |  2014 |  2015 |  2016 |  2017 |
| -----: | ----: | ----: | ----: | ----: |
|   Team |       |       |       |       |
| Devils | 863.0 | 673.0 |   NaN |   NaN |
|  Kings | 741.0 |   NaN | 756.0 | 788.0 |
| Riders | 876.0 | 789.0 | 694.0 | 690.0 |
| Royals | 701.0 | 804.0 |   NaN |   NaN |
|  kings |   NaN | 812.0 |   NaN |   NaN |

In [12]:

```
c_df.swaplevel().sort_index()
```

Out[12]:

```
Year  Team  
2014  Devils    863
      Kings     741
      Riders    876
      Royals    701
2015  Devils    673
      Riders    789
      Royals    804
      kings     812
2016  Kings     756
      Riders    694
2017  Kings     788
      Riders    690
Name: Points, dtype: int64
```

In [13]:

```
dc_df=df.groupby('Team')
dc_df.agg(min)
```

Out[13]:

|        | Rank | Year | Points |
| -----: | ---: | ---: | -----: |
|   Team |      |      |        |
| Devils |    2 | 2014 |    673 |
|  Kings |    1 | 2014 |    741 |
| Riders |    1 | 2014 |    690 |
| Royals |    1 | 2014 |    701 |
|  kings |    4 | 2015 |    812 |

In [14]:

```
dc_df.agg(np.max)
```

Out[14]:

|        | Rank | Year | Points |
| -----: | ---: | ---: | -----: |
|   Team |      |      |        |
| Devils |    3 | 2015 |    863 |
|  Kings |    3 | 2017 |    788 |
| Riders |    2 | 2017 |    876 |
| Royals |    4 | 2015 |    804 |
|  kings |    4 | 2015 |    812 |

In [15]:

```
f=lambda x:(x - x.mean())/x.std()
dc_df.transform(f)
```

Out[15]:

|      |      Rank |      Year |    Points |
| ---: | --------: | --------: | --------: |
|    0 | -1.500000 | -1.161895 |  1.284327 |
|    1 |  0.500000 | -0.387298 |  0.302029 |
|    2 | -0.707107 | -0.707107 |  0.707107 |
|    3 |  0.707107 |  0.707107 | -0.707107 |
|    4 |  1.154701 | -1.091089 | -0.860862 |
|    5 |       NaN |       NaN |       NaN |
|    6 | -0.577350 |  0.218218 | -0.236043 |
|    7 | -0.577350 |  0.872872 |  1.096905 |
|    8 |  0.500000 |  0.387298 | -0.770596 |
|    9 |  0.707107 | -0.707107 | -0.707107 |
|   10 | -0.707107 |  0.707107 |  0.707107 |
|   11 |  0.500000 |  1.161895 | -0.815759 |

In [16]:

```
dc_df.filter(lambda x : len(x)>=3)
```

Out[16]:

|      |   Team | Rank | Year | Points |
| ---: | -----: | ---: | ---: | -----: |
|    0 | Riders |    1 | 2014 |    876 |
|    1 | Riders |    2 | 2015 |    789 |
|    4 |  Kings |    3 | 2014 |    741 |
|    6 |  Kings |    1 | 2016 |    756 |
|    7 |  Kings |    1 | 2017 |    788 |
|    8 | Riders |    2 | 2016 |    694 |
|   11 | Riders |    2 | 2017 |    690 |

In [17]:

```
dc_df.filter(lambda x : x['Points'].max()>=800)
```

Out[17]:

|      |   Team | Rank | Year | Points |
| ---: | -----: | ---: | ---: | -----: |
|    0 | Riders |    1 | 2014 |    876 |
|    1 | Riders |    2 | 2015 |    789 |
|    2 | Devils |    2 | 2014 |    863 |
|    3 | Devils |    3 | 2015 |    673 |
|    5 |  kings |    4 | 2015 |    812 |
|    8 | Riders |    2 | 2016 |    694 |
|    9 | Royals |    4 | 2014 |    701 |
|   10 | Royals |    1 | 2015 |    804 |
|   11 | Riders |    2 | 2017 |    690 |

In [19]:

```
ck = dc_df.agg(np.max)
ck
```

Out[19]:

|        | Rank | Year | Points |
| -----: | ---: | ---: | -----: |
|   Team |      |      |        |
| Devils |    3 | 2015 |    863 |
|  Kings |    3 | 2017 |    788 |
| Riders |    2 | 2017 |    876 |
| Royals |    4 | 2015 |    804 |
|  kings |    4 | 2015 |    812 |

In [29]:

```
data={
    "Team":["Devils","Kings","Riders","Royals","kings",'A'],
    "num":[3,4,5,7,1,9],
    "id":['d1','d2','d3','d4','d5','d6']
}
ck2=pd.DataFrame(data)
ck2=ck2.set_index("Team")
ck2
```

Out[29]:

|        |  num |   id |
| -----: | ---: | ---: |
|   Team |      |      |
| Devils |    3 |   d1 |
|  Kings |    4 |   d2 |
| Riders |    5 |   d3 |
| Royals |    7 |   d4 |
|  kings |    1 |   d5 |
|      A |    9 |   d6 |

In [30]:

```
pd.merge(left=ck,right=ck2,how="inner",on="Team")
```

Out[30]:

|        | Rank | Year | Points |  num |   id |
| -----: | ---: | ---: | -----: | ---: | ---: |
|   Team |      |      |        |      |      |
| Devils |    3 | 2015 |    863 |    3 |   d1 |
|  Kings |    3 | 2017 |    788 |    4 |   d2 |
| Riders |    2 | 2017 |    876 |    5 |   d3 |
| Royals |    4 | 2015 |    804 |    7 |   d4 |
|  kings |    4 | 2015 |    812 |    1 |   d5 |

In [31]:

```
pd.merge(ck,ck2,on="Team",how='right')
```

Out[31]:

|        | Rank |   Year | Points |  num |   id |
| -----: | ---: | -----: | -----: | ---: | ---: |
|   Team |      |        |        |      |      |
| Devils |  3.0 | 2015.0 |  863.0 |    3 |   d1 |
|  Kings |  3.0 | 2017.0 |  788.0 |    4 |   d2 |
| Riders |  2.0 | 2017.0 |  876.0 |    5 |   d3 |
| Royals |  4.0 | 2015.0 |  804.0 |    7 |   d4 |
|  kings |  4.0 | 2015.0 |  812.0 |    1 |   d5 |
|      A |  NaN |    NaN |    NaN |    9 |   d6 |

In [32]:

```
zip()
pd.merge(ck,ck2,on="Team",how='left')
```

Out[32]:

|        | Rank | Year | Points |  num |   id |
| -----: | ---: | ---: | -----: | ---: | ---: |
|   Team |      |      |        |      |      |
| Devils |    3 | 2015 |    863 |    3 |   d1 |
|  Kings |    3 | 2017 |    788 |    4 |   d2 |
| Riders |    2 | 2017 |    876 |    5 |   d3 |
| Royals |    4 | 2015 |    804 |    7 |   d4 |
|  kings |    4 | 2015 |    812 |    1 |   d5 |

In [33]:

```
l1=[1,2,3,4]
l2=[1,2,3]
a_l=list(zip(l1,l2))
a_l
```

Out[33]:

```
[(1, 1), (2, 2), (3, 3)]
```

In [34]:

```
pd.merge(ck,ck2,on="Team",how='outer')
```

Out[34]:

|        | Rank |   Year | Points |  num |   id |
| -----: | ---: | -----: | -----: | ---: | ---: |
|   Team |      |        |        |      |      |
| Devils |  3.0 | 2015.0 |  863.0 |    3 |   d1 |
|  Kings |  3.0 | 2017.0 |  788.0 |    4 |   d2 |
| Riders |  2.0 | 2017.0 |  876.0 |    5 |   d3 |
| Royals |  4.0 | 2015.0 |  804.0 |    7 |   d4 |
|  kings |  4.0 | 2015.0 |  812.0 |    1 |   d5 |
|      A |  NaN |    NaN |    NaN |    9 |   d6 |

In [38]:

```
t1=ck.reset_index(drop=True)
t1
```

Out[38]:

|      | Rank | Year | Points |
| ---: | ---: | ---: | -----: |
|    0 |    3 | 2015 |    863 |
|    1 |    3 | 2017 |    788 |
|    2 |    2 | 2017 |    876 |
|    3 |    4 | 2015 |    804 |
|    4 |    4 | 2015 |    812 |

In [39]:

```
t2=ck2.reset_index(drop=True)
t2
```

Out[39]:

|      |  num |   id |
| ---: | ---: | ---: |
|    0 |    3 |   d1 |
|    1 |    4 |   d2 |
|    2 |    5 |   d3 |
|    3 |    7 |   d4 |
|    4 |    1 |   d5 |
|    5 |    9 |   d6 |

In [41]:

```
pd.concat([t1,t2],axis=0).reset_index(drop=True)
```

Out[41]:

|      | Rank |   Year | Points |  num |   id |
| ---: | ---: | -----: | -----: | ---: | ---: |
|    0 |  3.0 | 2015.0 |  863.0 |  NaN |  NaN |
|    1 |  3.0 | 2017.0 |  788.0 |  NaN |  NaN |
|    2 |  2.0 | 2017.0 |  876.0 |  NaN |  NaN |
|    3 |  4.0 | 2015.0 |  804.0 |  NaN |  NaN |
|    4 |  4.0 | 2015.0 |  812.0 |  NaN |  NaN |
|    5 |  NaN |    NaN |    NaN |  3.0 |   d1 |
|    6 |  NaN |    NaN |    NaN |  4.0 |   d2 |
|    7 |  NaN |    NaN |    NaN |  5.0 |   d3 |
|    8 |  NaN |    NaN |    NaN |  7.0 |   d4 |
|    9 |  NaN |    NaN |    NaN |  1.0 |   d5 |
|   10 |  NaN |    NaN |    NaN |  9.0 |   d6 |

In [42]:

```
pd.concat([t1,t2],axis=1).reset_index(drop=True)
```

Out[42]:

|      | Rank |   Year | Points |  num |   id |
| ---: | ---: | -----: | -----: | ---: | ---: |
|    0 |  3.0 | 2015.0 |  863.0 |    3 |   d1 |
|    1 |  3.0 | 2017.0 |  788.0 |    4 |   d2 |
|    2 |  2.0 | 2017.0 |  876.0 |    5 |   d3 |
|    3 |  4.0 | 2015.0 |  804.0 |    7 |   d4 |
|    4 |  4.0 | 2015.0 |  812.0 |    1 |   d5 |
|    5 |  NaN |    NaN |    NaN |    9 |   d6 |

In [43]:

```
end_df=t1.append(t2).reset_index(drop=True)
end_df
```

Out[43]:

|      | Rank |   Year | Points |  num |   id |
| ---: | ---: | -----: | -----: | ---: | ---: |
|    0 |  3.0 | 2015.0 |  863.0 |  NaN |  NaN |
|    1 |  3.0 | 2017.0 |  788.0 |  NaN |  NaN |
|    2 |  2.0 | 2017.0 |  876.0 |  NaN |  NaN |
|    3 |  4.0 | 2015.0 |  804.0 |  NaN |  NaN |
|    4 |  4.0 | 2015.0 |  812.0 |  NaN |  NaN |
|    5 |  NaN |    NaN |    NaN |  3.0 |   d1 |
|    6 |  NaN |    NaN |    NaN |  4.0 |   d2 |
|    7 |  NaN |    NaN |    NaN |  5.0 |   d3 |
|    8 |  NaN |    NaN |    NaN |  7.0 |   d4 |
|    9 |  NaN |    NaN |    NaN |  1.0 |   d5 |
|   10 |  NaN |    NaN |    NaN |  9.0 |   d6 |

In [45]:

```
end_df=t2.append(t1).reset_index(drop=True)
end_df
```

Out[45]:

|      |  num |   id | Rank |   Year | Points |
| ---: | ---: | ---: | ---: | -----: | -----: |
|    0 |  3.0 |   d1 |  NaN |    NaN |    NaN |
|    1 |  4.0 |   d2 |  NaN |    NaN |    NaN |
|    2 |  5.0 |   d3 |  NaN |    NaN |    NaN |
|    3 |  7.0 |   d4 |  NaN |    NaN |    NaN |
|    4 |  1.0 |   d5 |  NaN |    NaN |    NaN |
|    5 |  9.0 |   d6 |  NaN |    NaN |    NaN |
|    6 |  NaN |  NaN |  3.0 | 2015.0 |  863.0 |
|    7 |  NaN |  NaN |  3.0 | 2017.0 |  788.0 |
|    8 |  NaN |  NaN |  2.0 | 2017.0 |  876.0 |
|    9 |  NaN |  NaN |  4.0 | 2015.0 |  804.0 |
|   10 |  NaN |  NaN |  4.0 | 2015.0 |  812.0 |

In [ ]:

```

```