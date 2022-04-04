# 복습 및 실습

1. ```python
   # 복습
   import requests
   from bs4 import BeautifulSoup
   
   url = 'https://finance.naver.com/sise/sise_quant.naver' # 데이터의 진짜 위치 ( 주소 )
   r = requests.get(url) # 특정 주소에 위치하는 data 요청을 통한 HTML 정보 수집
   # print(r) # 결과 : <Response [200]>
   html_data = r.text
   soup = BeautifulSoup(html_data,'html.parser') # 정립된 정보 # 정적 도구
   data=soup.select("a.tltle")
   print(data)
   for i in data:
       print(i.text)
   ```

2. ```python
   # 복습
   import requests
   from bs4 import BeautifulSoup
   
   url = 'https://finance.naver.com/sise/sise_quant.naver' # 데이터의 진짜 위치 ( 주소 )
   r = requests.get(url) # 특정 주소에 위치하는 data 요청을 통한 HTML 정보 수집
   html_data = r.text
   soup = BeautifulSoup(html_data,'html.parser') # 정립된 정보 # 정적 도구
   
   data=soup.select("td")
   #data = soup.find_all("td",attrs="class":"number") # td 안에서 넘버라는 클래스 요소만 가져오곗다
   def f(n,x): #내가 입력한 숫자만큼 이동하겠다 (x)
       for i in range(x):
           n=n.next_sibling
       return n.text.strip() # 공백 제거 : .strip()
   
   for i in data:
       if i.a: # a 태그
           print(f"종목명:{i.a.text},현재가 : {f(i,2)}, PER : {f(i,18)}")
           #break #한번만
   
   ```

3. ```python
   import sqlite3
   #sqlite3.connect("파일의 이름") #db 파일 연결
   f=sqlite3.connect("Ex3_data.db") #db 파일 연결
   c=f.cursor() # 시작지점확인
   def f(data1, data2, data3):
       pass
   f(1,2,3)
   f(data1=10, data=20, data3=30)
   #c.execute("DROP TABLE IF EXISTS 테이블의 이름") #한줄 작성
   c.execute("DROP TABLE IF EXISTS data") #한줄작성
   c.execute("CREATE TABLE data(data1 text, data2 text, data3 text)") # db의 테이블 생성
   c.execute("INSERT INTO data VALUES(:data1, :data2, :data3)",{"data1":10,"data2":20,"data3":30})
   #c.executemany()#여러줄 작성(제공된 data(list)의 길이로 반복횟수 결정) / 딕셔너리를 포함하는 리스트를 입력받는다
   c.executemany("INSERT INTO data VALUES(:data1,:data2,:data3)",[{"data1":10,"data2":20,"data3":30},{"data1":10,"data2":20,"data3":30},{"data1":10,"data2":20,"data3":30}])
   f.commit()
   f.closes()
   
   #출력
   f=sqlite3.connect("Ex2_data.db")
   c=f.cursor()
   c.execute('SELECT * FROM data')
   for i in c.fetchall(): # 튜플로 가져오는 거 인지
       print(f"data1:{i[0]},data2:{i[1]},data3:{i[2]}")
   ```

4. ```python
   #Ex3 연관 복습
   import sqlite3
   
   import requests
   from bs4 import BeautifulSoup
   url = 'https://finance.naver.com/sise/sise_quant.naver'
   r = requests.get(url)
   html_data = r.text
   soup = BeautifulSoup(html_data,'html.parser')
   datas = soup.select("td")
   
   def f(n,x):
       for i in range(x):
           n=n.next_sibling
       return n.text.strip()
   def d(data):
       for i in datas:
           if i.a:
               a = {"종목명":i.a.text, "현재가" : f(i,2), "전일비" : f(i,4),"등락률" : f(i,6),"거래량" : f(i,8),"거래대금" : f(i,10),"매수호가" : f(i,12),"매도호가" : f(i,14),"시가총액" : f(i,16),"PER" : f(i,18),"ROE" : f(i,20)}
               data.append(a)
   
   #data = [{"종목":"에이비프로바이오" , "per":"-21.48"}]
   data = []
   db="data.db"
   d(data)
   
   def 저장(db, data):
       conn=sqlite3.connect(db) # db 는 주소값
       c = conn.cursor()
       c.execute('DROP TABLE IF EXISTS data') # db_data(data) 테이블 초기화
       c.execute('''
                   CREATE TABLE data(
                       종목명 text,
                       현재가 text,
                       전일비 text,
                       등락률 text,
                       거래량 text,
                       거래대금 text,
                       매수호가 text,
                       매도호가 text,
                       시가총액 text,
                       PER text,
                       ROE text
                   ) 
               ''')
       c.executemany('INSERT INTO data VALUES (:종목명, :현재가, :전일비, :등락률, :거래량, :거래대금, :매수호가, :매도호가, :시가총액, :PER, :ROE)', data)
       conn.commit()
       conn.close()
   
   def 출력(db):
       conn = sqlite3.connect(db)
       c = conn.cursor()
       c.execute('SELECT * FROM data')
       for i in c.fetchall():
           print(i)
   
   저장(db,data)
   출력(db)
   ```

5. ```python
   # 정리
   import sqlite3
   import requests # 수집
   from bs4 import BeautifulSoup # 정리
   url = 'https://movie.naver.com/movie/point/af/list.naver?&page=' # 파일 이름
   page = 10
   r = requests.get(url+str(page))
   r.raise_for_status() # 접속 상태 확인 / 체크 해주고 넘어가야함 ( 접속 코드 200 아닐시 예외 발생 )
   soup = BeautifulSoup(r.text,"html.parser") # 정리
   
   data = soup.find_all("td", attrs={"class":"title"})
   data_l=[]
   # print(type(data))
   # print(data.find("a").text)
   '''
   for i in data:
       if i.a:
           print("영화명",i.a.text)
           print("평점",i.em.text)
           print("리뷰",i.br.next_sibling.strip())
   '''
   for i in data:
       if i.a:
           data_l.append({"영화명":i.a.text,
            "평점":i.em.text,
            "리뷰":i.br.next_sibling.strip()})
   
   conn=sqlite3.connect("data_ex4.db") # db 는 주소값
   c = conn.cursor()
   c.execute('DROP TABLE IF EXISTS data') # db_data(data) 테이블 초기화
   c.execute('''
               CREATE TABLE data (
               영화명 text,
               평점 text,
               리뷰 text
               ) 
           ''')
   c.executemany('INSERT INTO data VALUES (:영화명,:평점,:리뷰)', data_l)
   conn.commit()
   conn.close()
   
   def 출력(db):
       conn=sqlite3.connect(db)
       c = conn.cursor()
       c.execute('SELECT * FROM data')
       for i in c.fetchall():
           print(f"영화명 : {i[0]}, 평점 : {i[1]}. 리뷰 : {i[2]}")
   
   
   ```

6. ```python
   import time
   from random import randint
   print("출력")
   for i in range(10):
       p=randint(3,10)
       print(p)
       time.sleep(p)
   print("종료")
   
   ```

7. ```python
   import time
   from random import randint
   import csv
   import requests # 수집
   from bs4 import BeautifulSoup # 정리
   url = 'https://movie.naver.com/movie/point/af/list.naver?&page=' # 파일 이름
   #파일의 내용 정리
   title = "영화명", "평점", "리뷰"
   f = open("save.csv","w",encoding='utf-8-sig', newline="") # sig : 한글 쓸려면 붙여야함
   writer=csv.writer(f)
   writer.writerow(title)
   in_data = []
   #data 수집
   for page in range(1,6):   #1개의 page를 로드(스크래핑)
       print(f"page{page} 크롤링중")
       r = requests.get(url+str(page))
       r.raise_for_status() # 접속 상태 확인 / 체크 해주고 넘어가야함 ( 접속 코드 200 아닐시 예외 발생 )
       soup = BeautifulSoup(r.text,"html.parser") # 정리
       data = soup.find_all("td", attrs={"class":"title"})
       #파일 정리
       for i in data:
           if i.a:
               # 단일 입력
               # in_data=[i.a.text,i.em.text,i.br.next_sibling.strip()]
               # writer.writerow(in_data)
               in_data.append([i.a.text,i.em.text,i.br.next_sibling.strip()])
       time.sleep(randint(5,10)) # 필수 !!
   #저장
   writer.writerows(in_data)
   ```

   