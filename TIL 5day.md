# 파이썬 크롤링과 스크레이핑



## Code

* ```python
  import re
  #.문자
  #^시작
  #$끝
  l=['abcd','adcd',"accd","abdc","casdfd","cabcdd","c1234d","cddddd"]
  ck=re.compile("^c....d$")
  def print_t(str):
      print(str)
      if str:
          print("출력정보 분석")
          print("일치문자",str.group())
          print("입력문자", str.string)
          print("일치문자 시작", str.start())
          print("일치문자 끝", str.end())
          print("일치문자 시작,끝", str.span())
      else:
          print("일치 없음")
  for i in l:
      str=ck.match(i)
      print_t(str)
      str=ck.search(i)
      print_t(str)
      print("all_data",ck.findall(i))
  
  #print(ck.search(""))
  #print(ck.findall(""))
  #원하는 형태에 따른 문자열 선택:정규식
  #match("문자열"):처음부터 일치
  #search("문자열"):일치하는 문자 있는지 확인
  #findall("문자열"): 일치 하는 모든것의 리스트 출력
  #.문자
  #^시작
  #$끝
  ```

* ```python
  from html.parser import HTMLParser
  class 추출(HTMLParser):
      def __init__(self):
          HTMLParser.__init__(self)
          self.is_strong =False
      def handle_starttag(self, tag, attrs):
          if tag =='strong':
              self.is_strong = True
      def handle_endtag(self, tag):
          if tag =='strong':
              self.is_strong = False
      def handle_data(self, data):
          if self.is_strong:
              print(data)
  with open("data.html") as f:
      parser=추출()
      parser.feed(f.read())
  ```

* ```python
  class A:
      __x=0
      def __init__(self):
          print("A생성자 동작")
      def f(self):
          print("A의 메소드")
  class Sub_A(A):
      def __init__(self):
          A.__init__(self)
          print("자식 A 생성")
  Sub_A()
  Sub_A()
  #1. 호출된 클래스의 객체가 메모리에 생성 -> 2. 초기자(인스턴스 메소드) 동작 *목적:객체의 초기화(필드,메소드)
  #A.__x#호출이 main
  ```

* ```python
  import requests
  r=requests.get('http://www.hanbit.co.kr/store/books/full_book_list.html')
  r.raise_for_status()
  #print(r.text)
  with open("data1.html","w",encoding="utf-8") as f:
      f.write(r.text)
  ```

* ```python
  import sys
  import re
  from urllib.request import urlopen
  f = urlopen('http://www.hanbit.co.kr/store/books/full_book_list.html')
  bytes_content = f.read()
  scanned_text = bytes_content[:1024].decode('ascii', errors='replace')
  #print(scanned_text)
  match = re.search(r'charset=["\']?([\w-]+)', scanned_text)
  
  if match:
      encoding = match.group(1)
  else:
      encoding = 'utf-8'
  print('encoding:', encoding, file=sys.stderr)
  text = bytes_content.decode(encoding)
  print(text)
  ```

* ```python
  import sys
  from urllib.request import urlopen
  f = urlopen('http://www.hanbit.co.kr/store/books/full_book_list.html')
  encoding = f.info().get_content_charset(failobj="utf-8")
  print('encoding:', encoding, file=sys.stderr)
  text = f.read().decode(encoding)
  print(text)
  
  ```

* ```python
  import requests
  from bs4 import BeautifulSoup
  url = 'http://www.hanbit.co.kr/store/books/full_book_list.html'
  r=requests.get(url)#html 갖고오기
  soup=BeautifulSoup(r.text,"html.parser")#정리
  """
  print(soup)
  t1=soup.find("a")#<a></a>1개
  print(t1)
  print(type(t1))
  l_t1=list(t1)
  print(len(l_t1))
  for i in t1:
      print(i)
  print("-"*20)
  
  t2=soup.find("div")#<div></div>1개
  print(t2)
  print(type(t2))
  #l_t2=list(t2)
  #print(len(l_t2))
  #print(l_t2)
  #for i in t2:
      #print(i)
  print("-"*20)
  #print(soup.a.get_text())
  #print(t1.get_text())
  """
  t2=soup.find("div")#<div></div>1개
  #print(type(t2))
  #print(t2.a.get_text())
  #print("-"*20)
  t3=soup.find_all("div")
  sub_t3=list(t3)[:2]
  for i in sub_t3:
      print(i.get_text())
      print("-"*20)
  #print(t3)
  #l_t3=list(t3)
  #print(len(l_t3))
  #print(l_t3[0])
  #print(type(l_t3[0]))
  #print(l_t3[0].a.get_text())
  #for i in t3:
      #print(i.a.get_text())
  #1,1,1,1,1,1,1,1
  ```

* ```python
  import requests
  from bs4 import BeautifulSoup
  url = 'http://www.hanbit.co.kr/store/books/full_book_list.html'
  r=requests.get(url)
  soup=BeautifulSoup(r.text,"html.parser")
  data1=soup.find("li")
  #print(data1.a['href'])
  data2=data1.next_sibling.next_sibling
  #print(data2)
  data3=data2.next_sibling.next_sibling
  #print(data3.get_text())
  data_all=soup.find_all("li")
  #for i in data_all:
      #print(i.get_text())
  a=soup.find('a',text="한빛미디어")
  print(a)
  
  ```

* ```python
  import requests
  from bs4 import BeautifulSoup
  url = 'https://www.hanbit.co.kr/store/books/full_book_list.html'
  r=requests.get(url)
  soup=BeautifulSoup(r.text,"html.parser")
  """
  data=soup.find_all("td",attrs={"class":"title"})
  #print(data)
  for i in data:
      print(i.a.get_text())
  """
  #print(soup.select('td[class=left]'))
  for i in soup.select('td[class=left]'):
      if i.a:
          print(i.a.text)
  for i in soup.find_all("td",attrs={"class":"title"}):
      print(i)
  ```

* ```python
  import requests
  from bs4 import BeautifulSoup
  url = 'https://search.naver.com/search.naver?query=%EB%A7%8C%EC%9A%B0%EC%A0%88&nso=&where=view&sm=tab_nmr&mode=normal'
  r=requests.get(url)
  soup=BeautifulSoup(r.text,"html.parser")
  data=soup.select("원하는 타이틀")
  for i in data:
      t=i.get_text()
      print(t)
  ```

* ```python
  import requests
  from bs4 import BeautifulSoup
  url = 'https://movie.naver.com/movie/point/af/list.naver'
  r=requests.get(url)
  soup=BeautifulSoup(r.text,"html.parser")
  data=soup.select("td.title")
  #for i in data:
      #print(i.br.next_sibling.text.strip())
  #데이터 특성을 이용한 코드
  for i in data:
      l=list(i)
      print(l[6].strip())
  """
  import requests
  from bs4 import BeautifulSoup
  url = 'https://movie.naver.com/movie/point/af/list.naver'
  r=requests.get(url)
  soup=BeautifulSoup(r.text,"html.parser")
  data=soup.select("td.title")
  for i in data:
      l=list(i)
      for j in range(len(i)):
          if j==6:
              print(l[j].strip())
  """
  ```

* ```python
  import requests
  from bs4 import BeautifulSoup
  url = 'https://finance.naver.com/sise/sise_rise.naver'
  r=requests.get(url)
  soup=BeautifulSoup(r.text,"html.parser")
  data=soup.select("a.tltle")
  for i in data:
      print(i.text)
  ```

* ```python
  import requests
  from bs4 import BeautifulSoup
  url = 'https://finance.naver.com/sise/sise_rise.naver'
  r=requests.get(url)
  soup=BeautifulSoup(r.text,"html.parser")
  data=soup.select("td")
  def f(n,x):
      for i in range(x):
          n=n.next_sibling
      return n.text
  for i in data:
      if i.a:
          print(f"종목명:{i.a.text},현재가:{f(i,2)},PER{f(i,18)}")
  ```

  