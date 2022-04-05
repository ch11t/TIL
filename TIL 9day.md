# 크롤링 / 스크래핑 실습 코딩

1. ```python
   import requests
   from bs4 import BeautifulSoup
   import csv
   
   headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36"}
   url = "https://news.naver.com/main/list.naver?mode=LS2D&sid2=228&sid1=105&mid=shm&date=20220404&page="
   title = "제목", "내용"
   f = open("Q5.csv", "w", encoding='utf-8-sig', newline="")
   writer=csv.writer(f)
   writer.writerow(title)
   r=requests.get(url+str(1),headers=headers)
   r.raise_for_status()
   html=r.text
   soup=BeautifulSoup(html, 'html.parser')
   
   data=soup.select("dt")
   data1=soup.select("dd")
   data_l=[] # 1번 data
   data1_l=[] # 2번 data
   for i in data:
       if i.a:
           if len(i.a.text.strip()) != 0:
               data_l.append(i.a.text.strip())
   
   for i in data1:
       data1_l.append(i.span.text)
   print(len(data_l),len(data1_l))
   data_all=list(zip(data_l,data1_l))
   print(len(data_all))
   for i in data_all:
       print(i)
   
   ```

2. ```python
   import requests
   from bs4 import BeautifulSoup
   import csv
   
   headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36"}
   url = "https://news.naver.com/main/list.naver?mode=LS2D&sid2=228&sid1=105&mid=shm&date=20220404&page="
   title = "제목", "내용"
   f = open("Q5.csv", "w", encoding='utf-8-sig', newline="")
   writer=csv.writer(f)
   writer.writerow(title)
   r=requests.get(url+str(1),headers=headers)
   r.raise_for_status()
   html=r.text
   soup=BeautifulSoup(html, 'html.parser')
   data=soup.select("dl")
   for i in data:
       if i.a:
           print(i.dd.previous_sibling.previous_sibling.a.text.strip())
           print(i.dd.span.text.strip())
   ```

3. ```python
   #추가 실습
   # http://land.naver.com/news/region.naver?city_no=1100000000&dvsn_no=&page=1 에서 뉴스 5페이지 까지
   # randint(3,6) sleep 함수 사용
   # 개인에게 맞는 User_Agent 정보를 찾아서 헤더를 이용하여 접속
   # csv 파일로 저장
   # 뉴스의 제목, 내용 2가지를 이용하여 입력
   # 저장된 csv 파일을 이용하여 내용 출력
   import requests
   from bs4 import BeautifulSoup
   import csv
   from random import randint
   import time
   
   headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36"}
   url = "http://land.naver.com/news/region.naver?city_no=1100000000&dvsn_no=&page="
   title = "제목", "내용"
   f = open("Q.csv", "w", encoding='utf-8-sig', newline="")
   writer=csv.writer(f)
   writer.writerow(title)
   in_data = []
   #data 수집
   for page in range(1,6):   #
       print(f"page{page} 크롤링중")
       r = requests.get(url+str(page))
       r.raise_for_status()
       soup = BeautifulSoup(r.text,"html.parser")
       data=soup.select("dl")
       #파일 정리
       for i in data:
           if i.a:
               #print(i.dd.previous_sibling.previous_sibling.a.text.strip())
               #print(i.span.previous_sibling.strip())
               in_data.append([i.dd.previous_sibling.previous_sibling.a.text.strip(),i.span.previous_sibling.strip()])
       time.sleep(randint(3,6))
   #저장
   writer.writerows(in_data)
   
   with open('Q.csv', 'r', encoding='utf-8-sig') as f:
       reader = csv.reader(f)
       for i in reader:
           print(i)
   ```

4. ```python
   import time
   from selenium import webdriver
   from selenium.webdriver.common.keys import Keys
   browser=webdriver.Chrome()
   browser.get("http://www.google.com")
   #n=browser.find_element_by_css_selector("input.gLFyf.gsfi")
   n=browser.find_element_by_xpath("/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input")
   n.send_keys("뉴스")
   n.send_keys(Keys.RETURN)
   browser.implicitly_wait(10)
   browser.execute_script("window.scrollTo(0,500);") # 스크롤을 500 만큼 내려라
   ck=browser.find_element_by_xpath("//*[@id='rso']/div[2]/div/div/div[1]/div/a/h3")
   ck.click()
   browser.implicitly_wait(10)
   for i in range(1,10):
       k=0
       g=k*i+100
       browser.execute_script(f"window.scrollTo({k},{g};")
       time.sleep(3)
       g=k
   time.sleep(10)
   ```

5. ```python
   from selenium import webdriver
   from selenium.webdriver.common.keys import Keys
   import pyperclip
   id="id" # 로그인 할 id
   pw="pw" # 로그인 할 pw
   b=webdriver.Chrome()
   b.implicitly_wait(10)
   b.get("http://naver.com")
   b.implicitly_wait(10)
   lc=b.find_element_by_class_name("link_login")
   lc.click()
   b.implicitly_wait(10)
   in_id=b.find_element_by_id('id')
   in_id.click()
   pyperclip.copy(id) # 복사
   in_id.send_keys(Keys.CONTROL,'v') #붙여넣기
   b.implicitly_wait(10)
   in_pw=b.find_element_by_id('pw')
   in_pw.click()
   pyperclip.copy(pw)
   in_pw.send_keys(Keys.CONTROL,'v') #붙여넣기
   b.implicitly_wait(10)
   b.find_element_by_id("log.login").click()
   b.implicitly_wait(10)
   ```

6. ```python
   import time
   from selenium import webdriver
   from selenium.webdriver.common.keys import Keys
   browser=webdriver.Chrome()
   browser.get("http://www.naver.com")
   #n=browser.find_element_by_css_selector("input.gLFyf.gsfi")
   browser.find_element_by_xpath("//*[@id='NM_NEWSSTAND_HEADER']/div[2]/a[1]").click()
   browser.implicitly_wait(10)
   browser.find_element_by_xpath("/html/body/section/header/div[2]/div/div/div[1]/div/div/ul/li[6]/a/span").click()
   browser.implicitly_wait(10)
   browser.find_element_by_xpath("//*[@id='snb']/ul/li[4]/a").click()
   browser.implicitly_wait(10)
   data_l=[]
   for i in range(1,11):
       data_l.append(browser.find_element_by_xpath(f'//*[@id="main_content"]/div[2]/ul[1]/li[1]/dl/dt[2]/a').text)
       browser.implicitly_wait(10)
   print(data_l)
   ```

   