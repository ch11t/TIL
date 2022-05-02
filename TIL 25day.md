# 데이터 전처리



* 데이터 전처리 ( data preprocessing )

  > 머신러닝 모델에 훈련 데이터를 입력하기 전에 데이터를 가공

* 머신러닝 기초 수식

  > y = f ( x )
  >
  > 데이터 x 는 훈련 데이터 ( train data ) 와 테스트 데이터 ( test data ) 가 모두 같은 구조를 갖는 피쳐 ( feature ) 이어야 함

* 기수형 데이터와 서수형 데이터

  > 기수형 데이터와 서수형 데이터는 일반적으로 숫자로 표현되지 않음
  >
  > 컴퓨터가 이해할 수 있는 숫자 형태의 정보로 변형

* 결측치

  > 실제로 존재하지만 데이터베이스 등에 기록되지 않는 데이터
  >
  > 해당 데이터를 빼고 모델을 돌릴 수 없기 때문에 결측치 처리 전략을 세워 데이터를 채워 넣음

* 이상치 ( outlier )

  > 극단적으로 크거나 작은 값
  >
  > 단순히 데이터 분포의 차이와는 다름
  >
  > 데이터 오기입이나 특이 현상 때문에 나타남

* 결측치 처리하기

  > 드롭과 채우기
  >
  > .
  >
  > 데이터를 삭제하거나 데이터를 채움
  >
  > 데이터가 없으면 해당 행이나 열을 삭제
  >
  > .
  >
  > 평균 값, 최빈 값, 중간 값 등으로 데이터를 채움
  >
  > .
  >
  > 결측치를 확인할 때 isnull 함수 사용
  >
  > * NaN 값이 존재할 경우 True, 그렇지 않을 경우 False 출력
  > * sum 함수로 True 인 경우 모두 더하고 전체 데이터 개수로 나누어 열별 데이터 결측치 비율을 구함

* 드롭 ( drop )

  > 결측치가 나온 열이나 행을 삭제
  >
  > dropna 사용하여 NaN 이 있는 모든 데이터의 행을 제거
  >
  > .
  >
  > 드롭과 관련된 대부분의 명령어들은 실제 드롭한 결과를 반환하거나 객체에 드롭 결과를 저장 하지는 않음
  >
  > .
  >
  > 드롭의 결과물을 저장하려면 다른 변수에 재할당 또는 매개변수 inplace=True 사용

* 채우기 ( fill )

  > 비어있는 값을 채움
  >
  > 일반적으로 드롭한 후에 남은 값들을 채우기 처리
  >
  > 평균, 최빈 값 등 데이터의 분포를 고려해서 채움
  >
  > 함수 fillna 사용

* 범주형 데이터로 변환하여 처리하기

  * 바인딩( binding )

    > 연속형 데이터를 범주형 데이터로 변환

  * 함수 cut 사용

    > bins 리스트에 구간의 시작 값 끝 값을 넣고 구간의 이름을 리스트로 나열
    >
    > cut 함수로 나눌 시리즈 객체와 구간, 구간의 이름을 넣어주면 해당 값을 바인딩하여 표시해줌