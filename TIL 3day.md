# 3일차

### Undoing

* git restore  

  > git restore <파일 이름> 의 형식을 사용
  >
  > Git의 추적이 되고 있는, 즉 버전 관리가 되고 있는 파일만 되돌리기 가능

  ==> 이미 버전 관리가 되고 있는 파일을 변경 후 저장, modified 상태가 될 것이고, 이를 git restore.

  ---

  

* **파일 상태를 Unstage로 되돌리기**

  : Staging Area와 Working Directory 사이

  

  * git rm --cached

    1. 새 폴더에서 git 초기화 후 진행 파일을 생성하고 git add를 진행

       `$ touch test.md`

       `$ git add test.md `

       `$ git status`

    

    2. Staging Area에 올라간 파일을 다시 내리기 (unstage)

       ```python
       $ git rm --cached test.md
       rm 'test.md'
       ```

    

  * git restore --staged

    * 두번째 상황 전 사전 준비

      ```python
      $ git add .
      $ git commit -m "first commit"
      ```

      1. test.md의 내용을 변경하고 git add 를 진행

         ```python
         test.md 파일 변경 후
         $ git add test.md
         ```

         

      2. Staging Area에 올라간 test.md 다시 내리기 (unstage)

         ```python
         $ git restore --staged test.md
         ```

​		/// **차이점**

​			`git rm --cached <file>`

​				> 기존에 커밋이 없는 경우

​				> "to unstage and remove paths only from the staging area"

​			`git restore --staged<file>`

​				> 기존에 커밋이 존재하는 경우

​				> "the contents are restored from HEAD"

---



* **바로 직전 커밋 수정하기**

  * git commit --amend

    * 두가지 역할

      1. Staging Area에 새로 올라온 내용이 없다면, 단순히 `직전 커밋의 메세지만` `수정`함

         (즉, 커밋하자마자 바로 이 명령어를 실행하는 경우)

      2.  Staging Area 에 새로 올라온 내용이 있다면, 직전 커밋 내역에 같이 묶어서 재 커밋됨

    1.1 커밋 메시지만 수정하는 경우

     1. A 기능을 완성하고 커밋합니다.

        `$ git commit -m 'B feature completed'`

     2. . 현재 커밋 해시 값 확인해두기

        `$ git log`

     3.  커밋 메시지 수정을 위해 다음과 같이 입력합니다.

        ```python
        $ git commit --amend
        
        hint : Waitinf for your editor to close the file..[master c01f908] Add no.txt ...
        ```

     4. Vim 편집기가 열리면서 직전 커밋 메시지를 수정할 수 있다.

        ```python
        B feature completed
        
        # Pleasse enter the commit message for your changes. Lines starting
        # with '#' will be ignored, and an empty message aborts the commit.
        .
        .
        .
        ```

     5. 커밋 메세지를 수정하고 저장하면 새로운 메세지로 변경되며 커밋 해시 값 또한 변경됨

        `$ git log`

        

    1-2 커밋 재작성

     1. 실수로 bar.txt를 빼면 커밋 해버린 상황까지 만들어 본다.

        ```python
        $ touch foo.txt bar.txt
        $ git add foo.txt
        ```

        ```python
        $ git status
        on branch master
        changes to be committed:
            (use "git restore --staged<file>..." to unstage)
        .
        .
        .
        
        ```

        ```python
        $ git commit -m 'foo & bar'
        
        [master 4221af6] foo & bar
        1 file changed, 0 insertions(+), 0 deletions(-)
        create mode 100644 foo.txt
        ```

        ```python
        $ git status
        
        on branch master
        untracked files:
            (use "git add <file>..." to include in what will be committed)
            	bar.txt
        ```

     2. 누락된 파일을 staging area로 이동시킴

        ```python
        $ git add bar.txt
        
        $ git status
        On branch master
        Changes to be committed:
          (use "git restore --staged <file>..." to unstage)
                new file:   bar.txt
        ```

     3. `git commit --amend` 를 입력

        ```python
        $ git commit --amend
        ```

     4. Vim 편집기가 열림( 마찬가지로 커밋 메세지 수정 가능)

        ````python
        ```bash
        foo & bar
        
        # Please enter the commit message for your changes. Lines starting
        # with '#' will be ignored, and an empty message aborts the commit.
        #
        # Date:      Mon Jun 7 22:32:58 2021 +0900
        #
        # On branch master
        # Changes to be committed:
        #       new file:   bar.txt
        #       new file:   foo.txt
        ```
        ````

     5. Vim 편집기를 저장 후 종료하면 직전 커밋이 덮어 씌워짐 (커밋 새로 추가 x)

        마찬가지로 커밋 `해시 값 또한 변경`

        ```python
        $ git commit --amend
        
        [master 7f6c24c] foo & bar
         Date: Mon Jun 7 22:32:58 2021 +0900
         2 files changed, 0 insertions(+), 0 deletions(-)
         create mode 100644 bar.txt
         create mode 100644 foo.txt
        ```

     6. git log -p를 사용하여 직전 커밋의 변경 내용을 살펴봄



<aside> 💡 **Vim 간단 사용법**

1. 입력 모드

   ```
   i
   ```

   - 문서 편집 가능

     

2. 명령 모드

   ```
   esc
   ```

   - `dd` : 해당 줄 삭제

   - ```
     :wq
     ```

      : **저장 및 종료**

     - `w` : write (저장)
     - `q` : quit (종료)

   - ```
     :q!
     ```

      : **강제 종료**

     - `q` : quit
     - `!` : 강제 </aside>

---





## reset, revert

* git reset

  > `git reset [옵션] <커밋 ID>`의 형태로 사용
  >
  > `시계를 마치 과거로 돌리는 듯한 행위`, 특정 커밋 상태로 돌아감
  >
  > 특정 커밋으로 돌아갔을 때 , 해당 커밋 이후로 쌓아 놨던 커밋은 전부 사라짐

  * `옵션`은 아래와 같이 세종류, 생략 시 `--mixed` 가 기본 값

    1. --soft

       * 돌아가려는 커밋으로 되돌아가고,
       * 이후의 commit된 파일들을 `staging area`로 돌려놓음 (commit 하기 전 상태)
       * 즉, 다시 커밋할 수 있는 상태가 됨

    2. --mixed

       * 돌아가려는 커밋으로 되돌아가고,
       * 이후의 commit된 파일들을 `working directory`로 돌려놓음(add 하기 전 상태)
       * 즉, unstage 된 상태로 남아있음
       * 기본 값

    3. --hard

       * 돌아가려는 커밋으로 되돌아가고,
       * 이후의 commit된 파일들(`tracked 파일들` ) 은 모두 working directory에서 삭제
       * 단, Untracked 파일은 Untracked 로 남음
       * 혹시나 이미 삭제한 커밋으로 다시 돌아가고 싶다면 `git reflog`를 사용

       예시

       ```python
       $ git log --oneline
       d56a232 (HEAD -> master) hello
       7f6c24c foo & bar
       006dc87 rename commit message
       3551584 asdasd
       71ccbf1 first
       
       
       $ git reset --hard 3551584
       HEAD is now at 3551584 asdasd
       
       
       # 3551584 커밋까지만 살아있고, 나머지 커밋은 모두 사라짐
       $ git log --oneline
       3551584 (HEAD -> master) asdasd
       71ccbf1 first
       
       
       $ git status
       On branch master
       nothing to commit, working tree clean
       ```

* git revert

  > `git revert < 커밋 아이디 `> 의 형태로 사용
  >
  > **특정 사건을 없었던 일로 만드는 행위**로써, `이전 커밋을 취소한다는 새로운 커밋`을 만듬
  >
  > git reset은 커밋 내역을 삭제하는 반면, git revert는 `새로 커밋을 쌓는다`는 차이

  * 예시

    ```python
    $ git log --oneline
    7f6c24c (HEAD -> master) foo & bar
    006dc87 rename commit message
    3551584 asdasd
    71ccbf1 first
    
    # revert commit 편집기 실행
    $ git revert 71ccbf1
    Removing foo.txt
    Removing bar.txt
    [master 3b55051] Revert "first"
     2 files changed, 0 insertions(+), 0 deletions(-)
     delete mode 100644 bar.txt
     delete mode 100644 foo.txt
    
    $ git log --oneline
    3b55051 (HEAD -> master) Revert "foo & bar" # 새로 쌓인 커밋
    7f6c24c foo & bar # 히스토리는 남아있음
    006dc87 rename commit message
    3551584 asdasd
    71ccbf1 first
    ```

    // git reset과 비슷하다는 이유로 다음 사항이 혼동 될 수 있음

    `git reset --hard 5sd2f42` 라고 작성한다면 5sd242라는 `커밋`으로 돌아간다는 뜻

    `it revert 5s2f42` 라고 작성한다면 5sd2f42라는 커밋`을` 되돌린다는 뜻



