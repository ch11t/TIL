# BeautifulSoup

> HTML 문서를 원하는 부분만 쉽게 뽑아낼 수 있는 파이썬 라이브러리



```python
import requests
from bs4 import BeautifulSoup

webpage = requests.get("https://www.daangn.com/hot_articles")
soup = BeautifulSoup(webpage.content, "html.parser")

print(soup)
```

> ` from bs4 import BeautifulSoup 로 라이브러리 호출
>
> ` 웹페이지를 요청한 뒤 받아낸 문서를 .content로 지정한 후 BeautifulSoup를 통해 soup라는 객체로 저장



* 태그(Tag) 탐색

  ```python
  print(soup.p)
  ```

  `<p>네이버 웹에서 어떤 - - -  입니다.<p>`

  ```python
  print(soup.p.string)
  ```

  `네이버 웹에서 어떤 - - - 입니다.`

  ```python
  print(soup.h1) # h1 태그 출력
  ```

  ```python
  for child in soup.ul.children:
      print(child)
  # 태그는 보통 트리구조로 위계가 있기 때문에 하위 항목을 모두 가져오고 싶다면 `.children` 을 사용
  ```

  ```python
  for parent in soup.ul.parents:
      print(parent)
  # 지정된 태그의 상위 항목을 가져오고 싶다면 `.parents' 사용
  # ul 상위에 있는 body 태그를 출력한 후 전체 html까지 추가로 출력
  ```

  ```python
  for d in soup.div.children:
      print(d)
  # div 태그 하위에 있는 요소들을 하나씩 출력한다
  ```

* **'find_all'** 을 통해 원하는 부분 가져오기

  ```python
  print(soup.find_all("h2"))
  # h2 태그만을 모두 찾아온다.
  # 정규식, html 속성, 함수 등을 사용해서 내가 원하는 부분을 추출할 수 있다.
  ```

* 정규식 활용

  ```python
  import re
  soup.find_all(re.compile("[ou]l"))
  # <ol> 이나 <ul> 이나 뭐든 포함된 리스트 추출
  ```

  ```python
  import re
  soup.find_all(re.compile("h[1-9]"))
  # h1 부터 h9까지 헤딩만 추출
  ```

* 리스트 활용

  ```python
  soup.find_all(['h1','p'])
  # .find_all()을 사용해 리스트로 원하는 태그를 지정해서 추출
  ```

* html 속성 활용

  > .find_all() 괄호 안에 **attrs** 파라미터를 **딕셔너리** 형태로 지정

  ```python
  soup.find_all(attrs=('class':'card-title'))
  # or
  soup.find_all(attrs=('class':footer-list', 'id':'footer-address-list'))
  ```

* 함수 활용

  > 원하는 부분을 가져오고 싶은데 조건이나 규칙이 까다로울 때

  ```python
  def search_function(tag):
      return tag.attr('class') == "card-title" and tag.strin == "Hello World"
  
  soup.find_all(search_function)
  ```

* CSS 선택자를 통해 원하는 부분 가져오기

  ```python
  soup.select(".card-region-name")
  # card-region-name 이라는 클래스를 가진 요소들만 추출
  # class 앞에는 .(dot) 찍어줄 것
  ```

  ```python
  soup.select("#hot-articles-go-download")
  # hot-aritcles-go-donwnload라는 id 를 가진 요소들만 추출
  # id 앞에는 # 찍어줄 것
  ```

* 텍스트만 읽어오기

  ```python
  for x in range(0, 10):
      print(soup.select(".card-title")[x].get_text())
   
  # 텍스트 추출을 위해 .get_text() 사용
  ```

  



