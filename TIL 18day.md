# 알고리즘 2일차



# 1.문제

N개의 숫자가 공백 없이 쓰여있다. 이 숫자를 모두 합해서 출력하는 프로그램을 작성하시오.

입력 첫째 줄에 숫자의 개수 N (1 ≤ N ≤ 100)이 주어진다. 둘째 줄에 숫자 N개가 공백없이 주어진다.

출력 입력으로 주어진 숫자 N개의 합을 출력한다.

In [ ]:

```
l=[]

Count=int(input())
Num=input()
for i in range(Count):
    l.append(int(Num[i]))
print(sum(l))

'''
_ = input()
print=sum(map(int, input())) # sum 은 itrable/iterator 한 값을 받으면 됨

'''
```

# 2.문제

N개의 수가 주어졌을 때, 이를 오름차순으로 정렬하는 프로그램을 작성하시오.

입력 첫째 줄에 수의 개수 N(1 ≤ N ≤ 1,000)이 주어진다. 둘째 줄부터 N개의 줄에는 수 주어진다. 이 수는 절댓값이 1,000보다 작거나 같은 정수이다. 수는 중복되지 않는다.

출력 첫째 줄부터 N개의 줄에 오름차순으로 정렬한 결과를 한 줄에 하나씩 출력한다.

In [ ]:

```
l=[]

Count=int(input())
for i in range(Count):
    l.append(int(input()))


for i in sorted(l):
    print(i)
    
# print(*sorted(numbers),sep="\n")    # * : unpacking : 낱개로 반환됨
```

# 3.문제

정수 n을 입력받아, 1부터 n까지의 제곱수를 리스트로 만들어 출력하는 프로그램을 작성하라.

리스트 내포를 이용한다.

In [ ]:

```
n = input())

squared_numbers=[]
for i in range(1, n+1):
    squared_numbers.append(i**2)
  
'''
# List comprehension
squared_numbers = [i ** 2 for i in range(1,n+1)]
print(squared+numbers)
'''
```

# 4.문제

리스트 내포를 이용하여 1번부터 입력받은 숫자까지의 숫자에 "No."를 붙인 문자열을 원소로 하는 리스트를 만들어 출력하시오.

In [1]:

```
Count = int(input())

l = [f"No.{i}" for i in range(1,Count+1)]
print(l)
5
['No.1', 'No.2', 'No.3', 'No.4', 'No.5']
```

# 5.문제

9개의 서로 다른 자연수가 주어질 때, 이들 중 최댓값을 찾고 그 최댓값이 몇 번째 수인지를 구하는 프로그램을 작성하시오.

예를 들어, 서로 다른 9개의 자연수

3, 29, 38, 12, 57, 74, 40, 85, 61

이 주어지면, 이들 중 최댓값은 85이고, 이 값은 8번째 수이다.

입력 첫째 줄부터 아홉 번째 줄까지 한 줄에 하나의 자연수가 주어진다. 주어지는 자연수는 100 보다 작다.

출력 첫째 줄에 최댓값을 출력하고, 둘째 줄에 최댓값이 몇 번째 수인지를 출력한다.

In [2]:

```
l=[int(input()) for _ in range(9)]

print(max(l))
print(l.index(max(l))+1)
1
2
3
4
5
6
7
4
5
7
6
```

# 6.문제

2행 4열의 리스트 두 개를 만들어서 입력을 받고 두 리스트의 곱을 구하여 출력하는 프로그램을 작성 하시오.

In [ ]:

```
L1 = [list(map(int, input())) for _ in range(2)]
L2 = [list(map(int, input())) for _ in range(2)]

for i in range(2):
    for j in range(4):
        print(L1[i][j]*L2[i][j],end=" ")
    print()
```

# 7.문제

2행 3열 리스트 두 개에 각각의 값을 입력 받은 후 두 배열의 같은 위치끼리 곱하여 새로운 리스트에 저장한 후 출력하는 프로그램을 작성하시오.

In [ ]:

```
L1 = [list(map(int, input())) for _ in range(2)]
L2 = [list(map(int, input())) for _ in range(2)]


L3=[list(L1[j][i]*L2[j][i] for i in range(3)) for j in range(2)]
for i in range(2):
    print(*L3[i])
```

# 8.문제

체스판은 8×8크기이고, 검정 칸과 하얀 칸이 번갈아가면서 색칠되어 있다. 가장 왼쪽 위칸 (0,0)은 하얀색이다. 체스판의 상태가 주어졌을 때, 하얀 칸 위에 말이 몇 개 있는지 출력하는 프로그램을 작성하시오.

입력 첫째 줄부터 8개의 줄에 체스판의 상태가 주어진다. ‘.’은 빈 칸이고, ‘F’는 위에 말이 있는 칸이다.

출력 첫째 줄에 문제의 정답을 출력한다.

In [ ]:

```
L1 = [list(input()) for _ in range(8)]
num=0
count=0

for i in L1:
    for j in i:
         if j == "F":
            if num%2==0:
                count+=1
         num+=1

print(count)
```