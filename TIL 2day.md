# 2일차

1. [**.gitignore **]

   : 특정 파일 혹은 폴더에 대해 Git이 버전 관리를 하지 못하도록 지정하는 것

   > 제외 하고 싶은 파일은 반드시 git add 전에 .gitignore 에 작성

   ___

   

   

2. 원격 저장소 가져오기

* **git clone**

  > 원격 저장소의 커밋 내역을 모두 가져와서, 로컬 저장소를 생성하는 명령어
  >
  > git clone < 원격 저장소 주소 >
  >
  > 생성된 로컬 저장소는 git init과 git remote add 가 이미 수행된 상태

* **git pull**

  > 원격 저장소의 변경 사항을 가져와서, 로컬 저장소를 업데이트하는 명령어
  >
  > git pull <저장소 이름> <브랜치 이름>

  ___





3. **Branch**

​		: 나뭇가지처럼 여러 갈래로 작업 공간을 나누어 독립적으로 작업할 수 있도록 도와주는 Git의 도구

​		> 브랜치는 독립 공간을 형성하기 때문에 원본(master)에 대해 안전

​		> 하나의 작업은 하나의 브랜치로 나누어 진행되므로 체계적인 개발이 가능

​		> Git은 브랜치를 만드는 속도가 빠르고, 용량도 적게 듬

* **git branch**

  > 브랜치 조회, 생성, 삭제 등 브랜치와 관련된 Git 명령어

  ```python
  # 브랜치 목록 확인
  $ git branch
  
  # 원격 저장소의 브랜치 목록 확인
  $ git branch -r
  
  # 새로운 브랜치 생성
  $ git branch <브랜치 이름>
  
  # 특정 커밋 기준으로 브랜치 생성
  $ git branch <브랜치 이름> <커밋 ID>
  
  # 특정 브랜치 삭제
  $ git branch -d < 브랜치 이름 > # 병합된 브랜치만 삭제 가능
  $ git branch -D < 브랜치 이름 > # 강제 삭제 ( 병합되지 않은 브랜치도 삭제 가능 )
  ```

  

* **git switch**

  > 현재 브랜치에서 다른 브랜치로 HEAD를 이동시키는 명령어

  ```python
  # 다른 브랜치로 이동
  $ git switch < 다른 브랜치 이름 >
  
  # 브랜치를 새로 생성과 동시에 이동
  $ git switch -c < 브랜치 이름 >
  
  # 특정 커밋 기준으로 브랜치 생성과 동시에 이동
  $ git switch -c < 브랜치 이름 > < 커밋 ID >
  ```

  ^ 주의할 점 : `브랜치 이동하기 전엔 꼭 커밋을 완료` 해야 함

  ___





4. **Branch Merge**

   > 브랜치를 합침( `병합` )

* **git merge**

  > 분기된 브랜치들을 하나로 `합치는` 명령어
  >
  > git merge < 합칠 브랜치 이름 >
  >
  > Merge 하기 전에 다른 브랜치를 합치려고 하는, 메인 브랜치로 switch 해야함

  ``` python
  # branch1 , branch2 / HEAD는 branch1
  $ git branch
  * branch1
    branch2
      
  # 2. branch2를 branch1에 합치려면
  $ git merge branch2
  
  # 3. branch1을 branch2에 합치려면
  $ git switch branch2
  $ git merge branch1
  ```

  

5. Merge 의 세 종류

   * Fast-forward

     > 브랜치를 병합할 때 `빨리감기`처럼 브랜치가 가리키는 커밋을 앞으로 이동
     >
     > `병합된 브랜치는 필요없으므로 삭제`

     

   * 3-Way Merge

     > 브랜치를 병합할 때 `각 브랜치의 커밋 두개와 공통 조상 하나`를 사용, 병합
     >
     > 두 브랜치에서 `다른 파일` 혹은 같은 `파일의 다른 부분`을 수정했을 때 가능

     

   * Merge Conflict

     > 병합하는 두 브랜치에서 `같은 파일의 같은 부분`을 수정한 경우, 
     >
     > Git이 어느 브랜치의 내용으로 작성해야 하는지 판단치 못해 `충돌 현상`
     >
     > 결국은 `사용자가 직접 내용을 선택`해서 충돌(Conflict)를 해결

   

   

   