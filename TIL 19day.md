# 1.문제

딕셔너리를 이용하여,

"Pokemon"을 입력하면 "Pikachu",

"Digimon"을 입력하면 "Agumon",

"Yugioh"를 입력하면 "Black Magician",

그 외의 문자열이 입력되면 "I don't know"를 출력하는 프로그램을 작성하시오.

In [ ]:

```
animation = input()

animations = {
    "Pokemon": "Pikachu",
    "Digimon": "Agumon",
    "Yugioh": "Black Magician"
}

print(animations.get(animation, "I don't know"))
```

# 2.문제

정수를 입력받아 입력받은 정수만큼 반복하면서, 각 줄에 나라의 이름과 그 나라의 수도를 공백을 사이에 두고 입력받는다. 그 후에, 나라의 이름을 입력받아 그 나라의 수도를 출력하는 프로그램을 작성하라. 만약 나라가 입력된 적이 없으면, Unknown Country을 출력한다.

In [ ]:

```
N=int(input())

Dic={}
for _ in range(N):
    x,y=input().split()
    Dic[x]=y

Name=input()
print(Dic.get(Name, "Unknown Country"))

'''
n = int(input())
countries = {}

for _ in range(n):
    country, city = input().split()
    countries[country] = city

checked_country = input()
print(countries.get(checked_country, "Unknown Country"))
''''''
```

# 3.문제

야구선수들의 파울 기록을 입력 받아, 파울을 가장 적게 한 선수를 모두 출력하고, 최소 파울 횟수를 마지막에 출력하는 프로그램을 작성하시오.

In [ ]:

```
players = input().split()
fouls = {}

for player in players:
    # 1. 파울 목록에 이름이 이미 있다
    if player in fouls:
        fouls[player] += 1
    # 2. 파울 목록에 이름이 없다
    else:
        fouls[player] = 1

min_foul = min(fouls.values()) # 2

for player, foul in fouls.items():
    if foul == min_foul:
        print(player)

print(min_foul)

```

# 4.문제

네오와 프로도가 숫자놀이를 하고 있습니다. 네오가 프로도에게 숫자를 건넬 때 일부 자릿수를 영단어로 바꾼 카드를 건네주면 프로도는 원래 숫자를 찾는 게임입니다.

다음은 숫자의 일부 자릿수를 영단어로 바꾸는 예시입니다.

1478 → "one4seveneight" 234567 → "23four5six7" 10203 → "1zerotwozero3" 이렇게 숫자의 일부 자릿수가 영단어로 바뀌어졌거나, 혹은 바뀌지 않고 그대로인 문자열 s가 매개변수로 주어집니다. s가 의미하는 원래 숫자를 return 하도록 solution 함수를 완성해주세요.

In [ ]:

```
def solution(s):
    numbers = {
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9'
    }
    
    for word, digit in numbers.items():
        s = s.replace(word, digit)
    
    return int(s)

'''
l=[0,1,2,3,4,5,6,7,8,9]
l1=['zero','one','two','three','four','five','six','seven','eight','nine']

def solution(s):
    answer1=""
    tt=""
    for i in s:
        for m in l:
            if i == str(m):
                answer1+=i
                i = ""
                break
        tt+=i
        for j in l1:
            if tt==j:
                tt=l[l1.index(j)]
                answer1+=str(tt)
                tt=""
    answer=int(answer1)
    return answer

print(solution(input()))
'''
```