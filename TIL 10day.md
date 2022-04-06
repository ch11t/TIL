# 크롤링 / 스크래핑 실습 코딩

1. ```python
   import warnings
   warnings.filterwarnings('ignore')
   from selenium import webdriver
   b=webdriver.Chrome() # 컨트롤러 실행
   b.implicitly_wait(10) # 활성화 될때까지 대기
   #time.sleep(초단위) # 스레드를 정지
   b.get("http://naver.com") # 요청( 페이지 이동 )
   b.implicitly_wait(10)
   in_t=b.find_element_by_xpath('//*[@id="query"]')
   in_t.send_keys("뉴스\n") # 키보드 입력
   b.find_element_by_xpath('//*[@id="search_btn"]/span[2]').click()
   b.implicitly_wait(10)
   ```

2. ```python
   #실습
   #구글 키고, 파이튜터 검색, 파이튜터 클릭, 코딩 창 까지 띄우기
   import time
   import warnings
   warnings.filterwarnings('ignore')
   from selenium import webdriver
   from selenium.webdriver.common.keys import Keys
   b=webdriver.Chrome() # 컨트롤러 실행
   b.implicitly_wait(10) # 활성화 될때까지 대기
   b.get("http://google.com") # 요청( 페이지 이동 )
   b.implicitly_wait(10)
   b.find_element_by_xpath('/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input').send_keys("파이튜터",Keys.ENTER)
   b.implicitly_wait(10)
   b.find_element_by_xpath('//*[@id="rso"]/div[1]/div/div/div/div/div/div[1]/a/h3').click()
   b.implicitly_wait(10)
   b.find_element_by_xpath('//*[@id="startLink"]/a').click()
   
   ```

3. ```python
   import time
   from selenium import webdriver
   from selenium.webdriver.common.keys import Keys
   from bs4 import BeautifulSoup
   browser=webdriver.Chrome()
   browser.implicitly_wait(10)
   browser.get("http://naver.com")
   browser.find_element_by_xpath('//*[@id="NM_NEWSSTAND_HEADER"]/div[2]/a[1]').click()
   browser.implicitly_wait(10)
   browser.find_element_by_xpath('/html/body/section/header/div[2]/div/div/div[1]/div/div/ul/li[6]/a/span').click()
   browser.implicitly_wait(10)
   browser.find_element_by_xpath('//*[@id="snb"]/ul/li[4]/a').click()
   browser.implicitly_wait(10)
   
   for i in range(1,6):
       print(f"{i}페이지")
       browser.implicitly_wait(10)
       browser.execute_script("window.scrollTo(0, document.body.scrollHeight)") # 0 에서 부터 구성요소의 가장 하위단 까지
       browser.implicitly_wait(10)
       html=browser.page_source
       browser.implicitly_wait(10)
       browser.find_element_by_xpath(f'//*[@id="main_content"]/div[3]/a[{i}]').click()
       #컴퓨터에서 동작
       s=BeautifulSoup(html,'html.parser')
       data=s.select('dl')
       for i in data:
           if i.a:
               print(i.dd.previous_sibling.previous_sibling.a.text.strip())
               print(i.dd.span.text.strip())
       print("출력완료")
   ```

4. ```python
   import time
   from selenium import webdriver
   from selenium.webdriver.common.keys import Keys
   from bs4 import BeautifulSoup
   b=webdriver.Chrome()
   b.implicitly_wait(10)
   b.get("http://naver.com")
   b.implicitly_wait(10)
   in_t=b.find_element_by_xpath('//*[@id="query"]').send_keys("2021 챔스\n")
   b.implicitly_wait(10)
   b.execute_script("window.scrollTo(0, document.body.scrollHeight)")
   b.find_element_by_xpath('//*[@id="main_pack"]/div[3]/div/div/a[2]').click()
   for i in range(3,7):
       b.implicitly_wait(10)
       b.execute_script("window.scrollTo(0, document.body.scrollHeight)")
       html=b.page_source
       b.find_element_by_xpath(f'//*[@id="main_pack"]/div[2]/div/div/a[{i}]').click()
       s=BeautifulSoup(html,'html.parser')
       data=s.select('a.link_tit')
       for i in data:
           print(i.text)
   ```

5. ```python
   # 1.네이버 뉴스에 암호화폐 검색 후 뉴스 제목과 내용 스크래핑하기
   # 2.csv 파일로 저장
   # 3. 저장된 파일을 이용하여 출력
   
   from selenium import webdriver
   from selenium.webdriver.common.keys import Keys
   from bs4 import BeautifulSoup
   import csv
   
   title = "제목","내용"
   f=open("Q.csv", "w", encoding='utf-8-sig', newline="")
   writer=csv.writer(f)
   writer.writerow(title)
   in_data=[]
   
   b=webdriver.Chrome()
   b.implicitly_wait(10)
   b.get("http://naver.com")
   b.implicitly_wait(10)
   in_t=b.find_element_by_xpath('//*[@id="query"]').send_keys("암호화폐\n")
   b.implicitly_wait(10)
   b.execute_script("window.scrollTo(0, document.body.scrollHeight)")
   b.find_element_by_xpath('//*[@id="main_pack"]/div[4]/div/div/a[2]').click()
   data_all=[]
   for i in range(3,7):
       print(f"page{i-1} 크롤링중")
       b.implicitly_wait(10)
       b.execute_script("window.scrollTo(0, document.body.scrollHeight)")
       html=b.page_source
       b.find_element_by_xpath(f'//*[@id="main_pack"]/div[2]/div/div/a[{i}]').click()
       s=BeautifulSoup(html,'html.parser')
       data=s.select('a.link_tit') # a.link_tit , a.dsc_txt
       data1=s.select('a.total_dsc')
       data_l=[]
       data1_l=[]
       for i in data:
           if len(i.text.strip()) != 0:
               data_l.append(i.text.strip())
       for i in data1:
           data1_l.append(i.text.strip())
       data_all.append(list(zip(data_l,data1_l)))
       #for i in data_all:
       #    print(i)
   
   writer.writerows(data_all)
   
   with open('Q.csv', 'r', encoding='utf-8-sig') as f:
       reader = csv.reader(f)
       for i in reader:
           print(i)
   
   ```

6. ```python
   import time
   import warnings
   warnings.filterwarnings('ignore')
   from selenium import webdriver
   from selenium.webdriver.common.keys import Keys
   from bs4 import BeautifulSoup
   browser=webdriver.Chrome()
   browser.maximize_window() #크기 결정
   browser.implicitly_wait(10)
   browser.get("http://google.com")
   browser.find_element_by_xpath('/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input').send_keys("뉴스\n")
   browser.implicitly_wait(10)
   browser.execute_script("window.scrollTo(0, 500)")
   browser.find_element_by_xpath('//*[@id="rso"]/div[2]//div/div/div[1]/div/a/h3').click()
   browser.implicitly_wait(10)
   # 모든 내용을 담고있는 페이지 최하단
   info_n=browser.execute_script("return document.body.scrollHeight")
   while True:
       browser.execute_script("window.scrollTo(0, document.body.scrollHeight)") # 불러온 페이지의 최하단
       time.sleep(2)
   #확장
       next_n=browser.execute_script("return document.body.scrollHeight")
       if info_n==next_n: # 모든 내용을 담고있는 페이지 최하단
           break
   
   browser.implicitly_wait(10)
   ```

7. ```python
   from selenium import webdriver
   from bs4 import BeautifulSoup
   op=webdriver.ChromeOptions()
   op.headless=True # 페이지를 열지 않고 실행하겠다
   op.add_argument("window-size=1920x1080") #구성요소 설정 / 사이즈 크기
   op.add_argument('user-agent=Mozilla/5.0 (Window NT 10.0; Win64; x64) AppleWebKit/537.36 - -')
   b=webdriver.Chrome(options=op)
   b.maximize_window()
   b.get("https://search.naver.com/search.naver?display=15&f=&filetype=0&page=2&query=%EC%95%94%ED%98%B8%ED%99%94%ED%8F%90&research_url=&sm=tab_pge&start=1&where=web")
   s=BeautifulSoup(b.page_source,'html.parser')
   print(s.text)
   print(b.page_source)
   
   ```

8. ```python
   #동적 크롤링
   import time
   #import selenium # 웹 자동화
   from selenium import webdriver # 웹페이지 컨트롤러
   from selenium.webdriver.common.keys import Keys
   from bs4 import BeautifulSoup
   op=webdriver.ChromeOptions() # 웹 컨트롤러의 옵션을 설정
   op.add_argument("window-size=1920x1080") # 지정된 크기로 페이지 오픈
   op.add_argument('User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36')
   # 봇으로 인식되지 않도록 설정. ( 단, 직접적으로 페이지를 연다면  사용하지 않아도 됨)
   b=webdriver.Chrome(options=op) #상위에 지정한 옵션으로 크롬 실행
   b.maximize_window()
   url="http://google.com"
   b.get(url)
   b.implicitly_wait(10) # 활성화 대기 (웹 창 전부가 완성되었을 때 동작 그리고 완성이 진행되지 않으면 예외 발생)
   in_d=b.find_elements_by_class_name('/html/body/div[1]/div[3]/form/div[1]/div[1]/div/div[2]/input')
   in_d.send_keys("구글뮤비\n") # 값 입력
   b.implicitly_wait(10) # 활성화대기
   b.execute_script("window.scrollTo(0, 500)")
   link=b.find_element_by_xpath('//*[@id="tads"]/div/div/div/div/div[1]/a/div[1]/span') # 링크 선택
   link.click() # 클릭 동작
   b.implicitly_wait(10)# 활성화대기
   while True: # 모든 내용 로드를 위한 동작
       info_n = b.execute_script("return document.body.scrollHeight") # 로드된 내용의 최하단 크기확인
       b.execute_script("window.scrollTo(0, document.body.scrollHeight)") # 로드된 내용의 촤하단으로 이동
       # 확 장
       time.sleep(2)
       next_n = b.execute_script("return document.body.scrollHeight")
       if info_n==next_n: # 모든내용을 담고 있는 페이지 최하단
           break
   html=b.page_source # 3. 페이지 크롤링
   s=BeautifulSoup(html, 'html.parser') # 정렬도구 ( 단수 )
   data=s.select('div.Epkrse')
   for i in data:
       print(i.text)
   
   ```

