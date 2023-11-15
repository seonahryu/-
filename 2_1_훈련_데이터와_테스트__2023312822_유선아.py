"""
## 훈련 세트와 테스트 세트

- 머신러닝 알고리즘의 성능을 제대로 평가하려면 훈련데이터와 평가에 사용할 데이터가 달라야함.
- feature -> x, target -> y
- 훈련 데이터 train set, 평가 데이터 test set
"""

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
# 생선의 길이(도미, 빙어)

fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
# 생선의 무게(도미, 빙어)

fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
# 각 생선에 대해서 길이와 무게를 짝지어줌. -> 하나의 생선 데이터 = 샘플

fish_target = [1]*35 + [0]*14
# 도미(35마리):1, 빙어(14마리):0

from sklearn.neighbors import KNeighborsClassifier
# 사이킷런에서 KNeighborsClassifier import해줌

kn = KNeighborsClassifier()
# KNeightborsClassifier 클래스에서 객체 생성

print(fish_data[4])
# 예시 : 인덱스 번호가 4인 데이터 출력 즉, 5번째 데이터 출력

print(fish_data[0:5])
# 예시 : 인덱스 번호가 0~4인 데이터 출력

print(fish_data[:5])
# 예시 : 인덱스 번호가 0~4인 데이터 출력(0 생략가능)

print(fish_data[44:])
# 인덱스 번호가 44~마지막 인덱스까지 출력

# 훈련 셋과 테스트 셋 나누기

train_input = fish_data[:35]
# 훈련 세트로 입력값 x 중 0~34번째 인덱스까지 사용

train_target = fish_target[:35]
# 훈련 세트로 타깃값 y 중 0~34번째 인덱스까지 사용

test_input = fish_data[35:]
# 테스트 세트로 입력값 x 중 35~마지막 인덱스까지 사용

test_target = fish_target[35:]
# 테스트 세트로 타깃값 y 중 35~마지막 인덱스까지 사용

kn.fit(train_input, train_target)
# fit() 모델 훈련

kn.score(test_input, test_target)
# score() 성능 평가
# 0 출력 -> 최악의 성능 발생
# 데이터 믹싱X 도미(0~34), 빙어(35~49)였는데 training 0~34, test 35~49 를 선택했기 때문 => numpy library 활용

"""## 넘파이"""

import numpy as np
# 넘파이 import (파이썬의 대표 "배열" 라이브러리)

input_arr = np.array(fish_data)
# fish_data를 넘파이 배열 형태로 변환

target_arr = np.array(fish_target)
# fish_target을 넘파이 배열 형태로 변환

print(input_arr)
# input_arr 배열 출력

print(input_arr.shape)
# input_arr의 배열 형태를 shape을 통해 알 수 있음
# 49개(도미, 빙어)(행)의 샘플과 2개(길이, 무게)(열)

np.random.seed(42)
# 넘파이에서 무작위 결과를 만드는 함수는 실행마다 다른 결과 도출
# 일정한 결과를 얻기 위해 랜덤시드(고정핀 역할, 임의의 수) 지정

index = np.arange(49)
# 0~48까지 1씩 증가하는 인덱스 생성 -> x, y 한 쌍으로 묶어서 둘다 섞어주기 위해

np.random.shuffle(index)
# shuffle() 함수로 배열 무작위로 섞음

print(index)
# 섞여진 배열 출력

print(input_arr[[1,3]])
# 넘파이의 배열 인덱싱
# [1,3]은 인덱스 번호 1, 3에 해당하는 데이터(2, 4번째 데이터) 출력
# 리스트 대신 넘파이 배열을 인덱스로 전달할 수 있음

train_input = input_arr[index[:35]]
# input_arr(x)에 index의 0~34번째 인덱스에 해당하는 리스트 슬라이싱
# train_input이 해당 값을 할당 받음
# 훈련 세트(x) 준비

train_target = target_arr[index[:35]]
# target_arr(y)에 index의 0~34번째 인덱스에 해당하는 리스트 슬라이싱
# train_target이 해당 값을 할당 받음
# 훈련 세트(y) 준비

print(input_arr[13], train_input[0])
# 섞인 index에서 첫번째 값 13이었고, 이 값이 할당된 train_input은 0번째 인덱스 -> 같은지 확인 작업

test_input = input_arr[index[35:]]
# input_arr(x)에 index 35~마지막 인덱스에 해당하는 리스트 슬라이싱
# teat_input에 해당 값을 할당 받음
# 테스트 세트(x) 준비

test_target = target_arr[index[35:]]
# target_arr(y)에 index 35~마지막 인덱스에 해당하는 리스트 슬라이싱
# test_target에 해당 값을 할당 받음
# 테스트 세트(y) 준비

import matplotlib.pyplot as plt
# 시각화를 위해 matplotlib에서 pyplot을 import -> 잘 섞였는지 데이터 분포 보고자함

plt.scatter(train_input[:, 0], train_input[:, 1])
# 훈련 데이터 산점도

plt.scatter(test_input[:, 0], test_input[:, 1])
# 테스트 데이터 산점도

plt.xlabel('length')
# x축 라벨

plt.ylabel('weight')
# y축 라벨

plt.show()
# 출력

"""## 두 번째 머신러닝 프로그램"""

kn.fit(train_input, train_target)
# 섞인 훈련 세트와 테스트 세트로 k-최근접 이웃 모델 훈련

kn.score(test_input, test_target)
# score() 성능 평가
# 1.0으로 100%의 정확도

kn.predict(test_input)
# test_input의 예측값(모델이 예측해서 내놓은 값) 출력

test_target
# 정답 출력
# test_input의 예측값과 전부 일치

"""##데이터 믹싱의 중요성
##사이킷런에서 fit(모델 훈련), score(성능 평가), predict(정확도뿐만 아니라 직접 값을 보고 성능 확인)
"""
