# fashion_how
[대회 홈페이지](https://fashion-how.org/ETRI/fashion_how.html)

## 과제 설명
유저와 코디 에이전트의 대화로 주어진 패션 아이템 코디의 순서를 메기는 과제 

<img width="699" alt="스크린샷 2021-09-28 오후 2 12 09" src="https://user-images.githubusercontent.com/22863750/135026999-1788e5d9-1212-4f91-800c-20fe2c3879f9.png">

## 사용 모델
*	모델은 코디 단위로 요약을 하고 요약된 결과로 코디의 순위를 매김
![전체 구조](https://user-images.githubusercontent.com/22863750/135027202-d5af6b41-7bf5-4cc6-8b14-1e227a840874.png)
* 코디 단위의 요약
  * 다이얼로그랑 패션 아이템 메타데이터는 transformer에 문장 단위로 적용해서 문장 단위의 요약을 얻음
  * 이미지 특징 정보는 linear을 통과시켜서 다이얼로그랑 패션 아이템 메타데이터랑 같은 사이즈로 변환시킴
  * 문장 단위로 요약된 다이얼로그, 패션 아이템 메타데이터랑 이미지 특징 정보를 하나의 tranformer에 넣어서 코디 단위의 요약
![코디 단위 요약](https://user-images.githubusercontent.com/22863750/135027215-516f1b2d-4856-4fad-b315-e2f6c8c61a82.png)
* 요약된 결과는 linear을 사용해 코디의 순위 얻음

