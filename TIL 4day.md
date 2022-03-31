# 파이썬 크롤링과 스크레이핑

## < Def >

* 크롤링

  > 웹 페이지의 하이퍼링크를 순회하면서 웹 페이지를 다운로드하는 작업

* 스크레이핑

  > 다운로드한 웹 페이지에서 필요한 정보를 추출하는 작업

* HTTP

  > 서버와 클라이언트는 프로토콜이라는 정해진 규약에 따라 통신하는데,  HTTP는 HTML 문서와 같은 리소스들을 가져올 수 있도록 해주는 프로토콜

  > ` Client의 요청(Request)이 있을 때만 서버가 응답(Response)하는 단방향 통신
  >
  > ` 계속해서 서버와 브라우저가 연결되어 있지는 않다.
  >
  > ` 서버는 클라이언트가 요청한 정보를 전송하고 곧바로 연결 종료
  >
  > ` 서버가 요구하는 API에 맞게 요청해야 응답을 받음

* URL

  > Uniform Resource Locator(인터넷에서 자원 위치) 를 나타낸다.
  >
  > ` 이론적으로 각각의 유일한 URL은 유일한 자원을 가리킴
  >
  > ` 자원으로 HTML 페이지, CSS 문서, 이미지 등이 될 수 있음

* HTTP 메서드 - GET/POST

  * GET

    > 주소와 함께 메세지를 남김
    >
    > 파일 업로드 불가
    >
    > 주로 조회 요청시 사용

  * POST

    > 주소와 함께 메세지, 파일업로드 지원
    >
    > 주로 추가/수정/삭제 요청시 사용

* HTTP 요청 / 응답 패킷 형식

  * 요청패킷

    > `요청헤더 : 클라이언트에서 필요한 헤더 Key/Value를 세팅한 후 요청, 전달
    >
    > `첫번째 빈줄 : Header와 Body 구분자
    >
    > ` Body : 클라이언트에서 필요한 Body를 세팅한 후 요청, 전달

  * 응답패킷

    > ` 응답헤더 : 서버에 필요한 Key/Value를 세팅한 후 응답, 전달
    >
    > ` 첫번째 빈줄 : Header와 Body 구분자
    >
    > ` Body : 서버에서 필요한 Body를 세팅한 후 요청, 전달

* 헤더/바디

  * 헤더

    > HTTP요청/응답 시에 헤더 정보가 Key/Value 형식으로 세팅됨
    >
    > `User-Agent : 브라우저의 종류
    >
    > ` Referer : 이전 페이지 URL ( 어떤 페이지를 거쳐서 왔는가 ? )
    >
    > ` Accept-Language : 어떤 언어로 응답을 원하는가?
    >
    > ` Authorization : 인증 정보

  * 바디

    > HTTP 요청시에는 바디가 없고, 응답에만 있음
    >
    > Ex) HTML코드, 이미지 데이터, JavaScript 코드, CSS코드, 비디오 데이터 등

* 웹

  > HTML ( Hyper Text Markup Language )
  >
  > CSS ( Cascading Style Sheet )
  >
  > 자바스크립트 ( Java Script )

* HTML

  * 하이퍼 텍스트

    > 비순차적으로 검색 할 수 있는 문서를 의미

  * 마크업 언어

    > 태그(tag)와 같은 구분자를 사용해서 데이터의 구조를 기술

    

    < HTML 구조 >

    > ![image-20220331234843387](C:\Users\yoon\AppData\Roaming\Typora\typora-user-images\image-20220331234843387.png)

* JSON

  > ` JavaScript Objet Notation (JSON)은 JavaScript 객체 문법으로 구조화된 데이터를 표현하기    위한 문자 기반의 표준 포맷
  >
  > ` 웹 어플리케이션에서 데이터를 전송할 때 일반적으로 사용
  >
  > ` JavaScript 객체 문법을 따르는 문자 기반의 데이터 포맷
  >
  > ` 문자열 형태로 존재





### < Code >

---

* ```python
  import numpy as np
  import sys
  import pandas as pd
  import matplotlib
  import requests
  import lxml
  import bs4
  
  print(f"python 버전 : {sys.version}")
  print(f"pandas 버전 : {pd.__version__}")
  print(f"matplotlib 버전 : {matplotlib.__version__}")
  print(f"numpy 버전 : { np.__version__}")
  print(f"requests 버전 : {requests.__version__}")
  print(f"lxml 버전 : {lxml.__version__}")
  print(f"beautifulsoup4: {bs4.__version__}")
  ```

* ```python
  j_data = {
      "key1": "data1",
      "key2": "data2",
      "key3": "data3",
      "key4": [100,355]
  } # 딕셔너리
  j_data2 = {
      "key4": "data1",
      "key5": "data2",
      "key6": "data3",
      "key7": [100, "data"]
  }
  
  import json  # json 불러오기
  
  with open("data.json", 'w') as f:# 텍스트 기반 통로 ( json은 문자열기반 )
      json.dump(j_data,f)
      # json.dump(j_data2,f) # 뒤에다 붙임
  
  ```

* ```python
  import json  # json은 문자 기반의 데이터 포맷
  with open("data.json", "r") as f: # 불러오기
      in_data=json.load(f)
      #in_data2=json.load(f)
  print(in_data)
  #print(in_data2)
  
  -----------------------------------------
  {"key1": "data1", "key2": "data2", "key3": "data3", "key4": [100, 355]}
  
  ```

* ```python
  from urllib.request import urlopen # 웹페이지 추출할 때  urllib.request 모듈 사용
  
  f = urlopen("http://www.hanbit.co.kr") # urlopen 함수에 url을 지정하면 웹 페이지 추출
  f1 = urlopen("http://www.naver.com") # 파이썬은 객체임을 인지
  print(f)
  #print(type(f)) HTTPResponse 자료형의 객체를 반환
  #print(f.read()) 추가로 HTTP 연결은 자동으로 닫히므로 따로 close()함수 호출하지 않아도 됨
  print(f.status) # 상태코드 추출
  print(f1)
  print(f1.status)
  print(f.getheader("Content-Type")) # HTTP 헤더의 값을 추출
  
  ```

* ```python
  import sys
  from urllib.request import urlopen
  f = urlopen('http://www.hanbit.co.kr/store/books/full_book_list.html')
  
  encoding = f.info().get_content_charset(failobj="utf-8")
  # HTTP 헤더를 기반으로 인코딩 방식을 추출(명시하지 않을 시 utf-8)
  print('encoding:', encoding, file=sys.stderr)
  # 인코딩 방식을 표준 오류에 출력
  
  text = f.read().decode(encoding) # 추출한 인코딩 방식으로 디코딩
  print(text) # 웹 페이지의 내용을 표준 출력에 출력
  ```

* ```python
  import sys
  import re
  from urllib.request import urlopen
  
  f = urlopen('http://www.hanbit.co.kr/store/books/full_book_list.html')
  bytes_content = f.read() # bytes 자료형의 응답 본문을 일단 변수에 저장
  
  #charset은 HTML의 앞부분에 적혀 있는 경우가 많으므로 응답 본문의 앞부분 1024바이트를 ASCII 문자로
  #디코딩해둠, ASCII 범위 이외의 문자는 U+FFFD(REPLACEMENT CHARACTER)로 변환되어 예외 발생치 않음
  scanned_text = bytes_content[:1024].decode('ascii', errors='replace')
  #print(scanned_text)
  
  # 디코딩한 문자열에서 정규 표현식으로 charset 값을 추출
  match = re.search(r'charset=["\']?([\w-]+)', scanned_text)
  #print(match)
  
  if match:
      encoding = match.group(1)
  else :
      encoding = 'utf-8' #charset이 명시돼 있지 않으면 UTF-8을 사용
      
  print('encoding:', encoding, file=sys.stderr) # 추출한 인코딩을 표준 오류에 출력
  
  #text = bytes_content.decode(encoding) #추출한 인코딩으로 다시 디코딩
  text = f.read().decode(encoding)
  print(text) # 응답 본문을 표준 출력에 출력
  ```

* ```python
  import csv #csv 파일 만드는 방법
  
  with open("data.csv",'w',newline='') as f: #파일을 연다, newline=''은 줄바꿈 제어
      wd = csv.DictWriter(f,["key1","key2","key3"]) #딕셔너리 요소 출력시 사용
      #wd.writeheader() #키값이 헤더가 됨
      wd.writerow({"key1":10,"key2":20,"key3":30}) # 키값의 명시를 위해 딕셔너리 형태여야함
      wd.writerows([{"key1":10,"key2":20,"key3":30},{"key1":10,"key2":20,"key3":30},{"key1":10,"key2":20,"key3":30}]) # writerows() 로 여러개의 데이터를 딕셔너리 형태로 작성
  
      '''
      wd = csv.writer(f)
      for i in range(5):
          wd.writerow(["data1","data2","data3"]) #1차원
      wd.writerows([[10,20,30],[10,20,30],[10,20,30],[10,20,30],[10,20,30]]) #2차원
      '''
      '''
      wd = csv.writer(f)
      for i in range(5):
          wd.writerow([[10]])
          wd.writerow([10])
          wd.writerow("10")
          wd.writerow([[10],"data",{1,2,3}])
      '''
  -------------------------------------------------------------------------
  10,20,30
  10,20,30
  10,20,30
  10,20,30
  
  ```

* ```python
  import csv
  with open("data.csv",'w',newline='') as f:
      wd = csv.DictWriter(f,["key1","key2","key3"])
      wd.writeheader() #키 값이 헤더가 됨
      wd.writerow({"key1":10,"key2":20,"key3":30})
      wd.writerows(({"key1":10,"key2":20,"key3":30},
                   {"key1":10,"key2":20,"key3":30},
                   {"key1":10,"key2":20,"key3":30},
                   {"key1":10,"key2":20,"key3":30}))
  ----------------------------------------------------
  key1,key2,key3
  10,20,30
  10,20,30
  10,20,30
  10,20,30
  10,20,30
  
  ```

* ```python
  import sqlite3
  conn = sqlite3.connect("data.db") # data.db 파일을 열고 연결을 변수에 저장
  c=conn.cursor() #커서를 추출
  c.execute('DROP TABLE IF EXISTS cities') 
  # execute() 메서드로 SQL 구문을 실행, 스크립트를 여러 번 사용해도 같은 결과를 출력할 수 있게
  # cities 테이블이 존재하는 경우 제거
  c.execute('''
      CREATE TABLE cities (
          rank integer,
          city text,
          population integer
      ) 
  ''') # cities 테이블 생성
  c.execute('INSERT INTO cities VALUES (?, ?, ?)', (1, '상하이', 24150000))
  #execute  두번째 매개변수에는 파라미터를 지정할 수 있음
  #SQL 내부에서 파라미터로 변경할 부분(플레이스홀더)은 ?로 지정
  
  c.execute('INSERT INTO cities VALUES (:rank, :city, :population)',
            {'rank': 2, 'city': '카라치', 'population': 23500000})
  #파라미터가 딕셔너리일 경우 플레이스홀더를 :<이름> 형태로 지정
  
  c.executemany('INSERT INTO cities VALUES (:rank, :city, :population)', [
      {'rank': 3, 'city': '베이징', 'population': 21516000},
      {'rank': 4, 'city': '텐진', 'population': 14722100},
      {'rank': 5, 'city': '이스탄불', 'population': 14160467},
  ]) # executemany() 메서드를 사용하면 여러 개의 파라미터를 리스트로 지정해서
     # 여러 개(현재 예제에서는 3개)의 SQL 구문을 실행할 수 있음
      
  '''
  c.executemany('INSERT INTO cities VALUES (?, ?, ?)', [(1, '상하이', 24150000),(1, '상하이', 24150000),(1, '상하이', 24150000),(1, '상하이', 24150000)])
  '''
  
  conn.commit() # 변경사항을 커밋(저장)
  
  c.execute('SELECT * FROM cities') # 저장한 데이터를 추출
  for i in c.fetchall(): # 추출한 데이터를 출력
      print(i)
      print(type(i))
  conn.close() # 연결 닫음
  ```

* ```python
  import html
  data="<span class = veta_bd_t> data </span>"
  out_data=html.escape(data) # 문자화
  print(out_data)
  c_data=html.unescape(out_data)
  print(c_data)
  ```

* ```python
  from urllib.request import urlopen
  import re
  from html import unescape
  import sqlite3
  
  def 추출(url): # 웹페이지 추출, 인코딩 형식은 Content-Type헤더를 통해알아냄. 반환값: str 자료형의 HTML
      f = urlopen(url)
      # HTTP 헤더를 기반으로 인코딩 형식을 추출
      encoding = f.info().get_content_charset(failobj="utf-8")
  	# 추출한 인코딩 형식을 기반으로 문자열을 디코딩
      html = f.read().decode(encoding)
      return html
  
  def 정규화(html): # HTML을 기반으로 정규 표현식을 사용해 정보 추출, 반환값 리스트
      data=[]
      
      # re.findall()을 사용해 해당하는 HTML을 추출
      for i in re.findall(r'<td class="left"><a.*?</td>', html, re.DOTALL):
          url = re.search(r'<a href="(.*?)">',i).group(1) # URL 추출
          url="http://www.hanbit.co.kr"+url
          title = re.sub(r'<.*?>', '',i) # 태그를 제거해서 제목 추출
          title = unescape(title)
          data.append({'url':url,"title":title})
      return data
  
  def 저장(db,data): 
      # 매개변수로 전달된 목록을 SQLite 데이터베이스에 저장
      # 데이터 베이스의 경로는 매개변수 db로 지정
      # 반환값: None(없음)
      
      conn = sqlite3.connect(db) # 데이터베이스를 열고 연결을 확립
      c=conn.cursor() # 커서를 추출
      c.execute('DROP TABLE IF EXISTS data') # 메서드로 SQL 실행, 기존 data 테이블 제거
      c.execute('''
              CREATE TABLE data (
                  title text,
                  url text
              )
          ''') # data 테이블 생성
      c.executemany('INSERT INTO data VALUES (:title, :url)', data)
      # 메서드를 사용하면 매개변수로 리스트 지정 가능
      conn.commit() # 변경사항을 커밋(저장)
      conn.close() # 연결 종료
      
  def 출력(db):
      conn = sqlite3.connect(db) # 데이터베이스를 열고 연결을 확립
      c = conn.cursor() #커서를 추출
      c.execute('SELECT * FROM data') # 저장한 데이터를 추출
      for i in c.fetchall():
          print(i) # 출력
  
  if __name__=="__main__":
      html=추출("http://www.hanbit.co.kr/store/books/full_book_list.html")
      print(html)
      #data=정규화(html)
      #print(data)
      #저장("data.db",data)
      #출력("data.db")
  
  ```

* ```python
  from html.parser import HTMLParser
  class 추출(HTMLParser):
      def __init__(self):
          HTMLParser.__init__(self) # 셀프가 주어지면 인스턴스로서의 접근이 되기에 호출이 가능해짐
          self.is_strong = False
      def handle_starttag(self, tag, attrs):
          if tag == 'strong':
              self.is_strong = True
      def handle_endtag(self, tag):
          if tag == 'strong':
              self.is_strong = False
      def handle_data(self, data: str):
          if self.is_strong:
              print(data)
  
  with open("data.html") as f:
      parser = 추출()
      parser.feed(f.read())
  ```

  