#!/usr/bin/env python
# coding: utf-8

# #기계학습 입문 2023-2 중간 프로젝트 코드
# #### 주제: Breast Cancer Wisconsin (Diagnostic) Data Set 분석
# ##### 목표: 유방암 악성 판단에 유의한 변수 파악하기
# ##### 데이터 개수는 적더라도 변수가 많은 데이터를 활용

# Breast cancer is the most common cancer amongst women in the world. It accounts for 25% of all cancer cases and affected over 2.1 Million people in 2015 alone. It starts when cells in the breast begin to grow out of control. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area.
# 

# ##데이터 불러오기

# In[65]:


from google.colab import drive
drive.mount('/content/drive')


# In[112]:


import pandas as pd
import numpy as np

data=pd.read_csv('/content/drive/MyDrive/data.csv')


# In[67]:


from google.colab.data_table import DataTable
DataTable.max_columns = 40


# In[68]:


data.head()


# ###결측치 확인 및 기술 통계

# In[69]:


pd.isnull(data).sum()


# In[70]:


#데이터 크기 출력
print(data.shape)
#데이터 크기는 비록 569로 작지만 독립변수로 활용가능한 변수가 33개로 매우 많음


# In[71]:


data.describe()


# In[72]:


data['diagnosis'].unique()


# In[73]:


malignant=data[data["diagnosis"]=="M"]
benign=data[data["diagnosis"]=="B"]


# In[74]:


malignant.describe()


# In[75]:


benign.describe()


# ##EDA 분석(데이터 시각화)

# In[79]:


print(diagnosis_count)


# In[113]:


#diagnosis 변수에서 M과 B의 비율 시각화
import matplotlib.pyplot as plt

diagnosis_count = data['diagnosis'].value_counts()
catagories=["Malignant","Benign"]
# 막대 그래프 생성

plt.bar(catagories,diagnosis_count)
plt.xlabel('diagnosis')
plt.ylabel('count')
plt.title("Number of Malignant and Benign Samples")
# 그래프 출력
plt.show()


# In[78]:


#HEATMAP
import seaborn as sns
import matplotlib
import matplotlib.pylab as plt
plt.figure(figsize =(20,20))
sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
plt.show()


# ##머신러닝 준비
# ### 학습 데이터 및 테스트 데이터 준비

# In[80]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[81]:


#dianosis데이터의 범주를 0과1로 수치화
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
data.head(10)


# In[82]:


#종속변수를 target에 넘파이로 저장
target=data['diagnosis'].to_numpy()


# In[83]:


#이후 특성 중요도를 변수를 찾기위한 데이터프레임 남겨두기 위한 코드
train_data=data.iloc[:,2:-1]

#활용가능한 독립변수만 data에 numpy형태로 저장(id, diagnosis, Unnamed:32 변수 제거)
data=data.iloc[:,2:-1].to_numpy()


# In[84]:


#데이터를 8:2의 비율로 훈련세트와 테스트 세트로 나누기
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 특성 표준화
scaler = StandardScaler()
train_input = scaler.fit_transform(train_input)
test_input = scaler.transform(test_input)


# In[85]:


train_input


# In[86]:


test_input


# ###KNN

# In[176]:


from sklearn.model_selection import GridSearchCV #어떤 파라미터가 최적일지
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

kn=KNeighborsClassifier()

hyper_parmas = {'n_neighbors' : list(range(1,20))}

grid_kn_model = GridSearchCV(kn, param_grid=hyper_parmas, cv=5, refit=True, return_train_score=True)
grid_kn_model.fit(train_input, train_target)

best_kn_model = grid_kn_model.best_estimator_ #최적의 모델
train_accuracy = best_kn_model.score(train_input, train_target)
test_accuracy = best_kn_model.score(test_input, test_target)

# 교차 검증을 통한 정확도 평균 계산
cv_scores = cross_val_score(best_kn_model, train_input, train_target, cv=5)
cv_accuracy_mean = cv_scores.mean()

# 최적의 파라미터
best_params = grid_kn_model.best_params_

# 결과 출력
print(f"학습 세트 정확도: {train_accuracy:.4f}")
print(f"테스트 세트 정확도: {test_accuracy:.4f}")
print(f"교차 검증 정확도 평균: {cv_accuracy_mean:.4f}")
print(f"최적의 파라미터: {best_params}")


# In[177]:


# auc
y_scores = best_kn_model.predict_proba(test_input)[:, 1]
fpr, tpr, thresholds = roc_curve(test_target, y_scores)
roc_auc = roc_auc_score(test_target, y_scores)

# roc 곡선 그리기
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate(Recall)')
plt.title('ROC Curve of KNN')
plt.legend(loc='lower right')
plt.show()


# In[178]:


# 예측값 생성
train_pred = best_kn_model.predict(train_input)
test_pred = best_kn_model.predict(test_input)

# True Positive, True Negative, False Positive, False Negative 계산
def calculate_confusion_matrix(true, pred):
    true_positive = sum((true == 1) & (pred == 1))
    true_negative = sum((true == 0) & (pred == 0))
    false_positive = sum((true == 0) & (pred == 1))
    false_negative = sum((true == 1) & (pred == 0))
    return true_positive, true_negative, false_positive, false_negative

# 훈련 세트에 적용
train_tp, train_tn, train_fp, train_fn = calculate_confusion_matrix(train_target, train_pred)

# 테스트 세트에 적용
test_tp, test_tn, test_fp, test_fn = calculate_confusion_matrix(test_target, test_pred)

# 민감도(recall), 특이도, 정밀도, 정확도, F1 Score 계산
sensitivity_train = train_tp / (train_tp + train_fn)
specificity_train = train_tn / (train_tn + train_fp)
precision_train = train_tp / (train_tp + train_fp)
accuracy_train = (train_tp + train_tn) / (train_tp + train_tn + train_fp + train_fn)
f1_train = (2 * precision_train * sensitivity_train) / (precision_train + sensitivity_train)

sensitivity_test = test_tp / (test_tp + test_fn)
specificity_test = test_tn / (test_tn + test_fp)
precision_test = test_tp / (test_tp + test_fp)
accuracy_test = (test_tp + test_tn) / (test_tp + test_tn + test_fp + test_fn)
f1_test = (2 * precision_test * sensitivity_test) / (precision_test + sensitivity_test)

# 결과 출력
print(f"훈련 세트 - 민감도: {sensitivity_train:.3f}, 특이도: {specificity_train:.3f}, 정밀도: {precision_train:.3f}, 정확도: {accuracy_train:.3f}, F1 Score: {f1_train:.3f}")
print(f"테스트 세트 - 민감도: {sensitivity_test:.3f}, 특이도: {specificity_test:.3f}, 정밀도: {precision_test:.3f}, 정확도: {accuracy_test:.3f}, F1 Score: {f1_test:.3f}")


# ##Random Forest(RF)

# In[179]:


from sklearn.ensemble import RandomForestClassifier

# 랜덤 포레스트 모델 생성
random_forest = RandomForestClassifier(random_state=42)

# 그리드 서치를 사용하여 최적의 파라미터 찾기
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [1,2,3,4,5,6,7,8,9,10],
    'min_samples_split': [2, 5, 10]
}
grid_rf_search = GridSearchCV(random_forest, param_grid, cv=5)
grid_rf_search.fit(train_input, train_target)

# 최적의 모델을 사용하여 예측
best_rf_model = grid_rf_search.best_estimator_
train_accuracy = best_rf_model.score(train_input, train_target)
test_accuracy = best_rf_model.score(test_input, test_target)

# 교차 검증을 통한 정확도 평균 계산
cv_scores = cross_val_score(best_rf_model, train_input, train_target, cv=5)
cv_accuracy_mean = cv_scores.mean()

# 최적의 파라미터
best_params = grid_rf_search.best_params_

# 특성 중요도
feature_importances = best_rf_model.feature_importances_

# 결과 출력
print(f"학습 세트 정확도: {train_accuracy:.4f}")
print(f"테스트 세트 정확도: {test_accuracy:.4f}")
print(f"교차 검증 정확도 평균: {cv_accuracy_mean:.4f}")
print(f"최적의 파라미터: {best_params}")
print(f"특성 중요도: {feature_importances}")


# In[180]:


# AUC
y_scores = best_rf_model.predict_proba(test_input)[:, 1]
fpr, tpr, thresholds = roc_curve(test_target, y_scores)
roc_auc = roc_auc_score(test_target, y_scores)

# ROC 곡선 그리기
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Random Forest')
plt.legend(loc='lower right')
plt.show()


# In[181]:


# 각 특성의 중요도를 기준으로 내림차순 정렬하여 상위 5개의 특성을 선택
top_features_indices = feature_importances.argsort()[::-1][:5]

# 상위 5개 특성의 이름 출력
top_features_names = train_data.columns[top_features_indices]
print("상위 5개의 중요한 특성:")
for feature_name in top_features_names:
    print(feature_name)


# In[182]:


# 예측값 생성
train_pred = best_rf_model.predict(train_input)
test_pred = best_rf_model.predict(test_input)

# True Positive, True Negative, False Positive, False Negative 계산
def calculate_confusion_matrix(true, pred):
    true_positive = sum((true == 1) & (pred == 1))
    true_negative = sum((true == 0) & (pred == 0))
    false_positive = sum((true == 0) & (pred == 1))
    false_negative = sum((true == 1) & (pred == 0))
    return true_positive, true_negative, false_positive, false_negative

# 훈련 세트에 적용
train_tp, train_tn, train_fp, train_fn = calculate_confusion_matrix(train_target, train_pred)

# 테스트 세트에 적용
test_tp, test_tn, test_fp, test_fn = calculate_confusion_matrix(test_target, test_pred)

# 민감도(recall), 특이도, 정밀도, 정확도, F1 Score 계산
sensitivity_train = train_tp / (train_tp + train_fn)
specificity_train = train_tn / (train_tn + train_fp)
precision_train = train_tp / (train_tp + train_fp)
accuracy_train = (train_tp + train_tn) / (train_tp + train_tn + train_fp + train_fn)
f1_train = (2 * precision_train * sensitivity_train) / (precision_train + sensitivity_train)

sensitivity_test = test_tp / (test_tp + test_fn)
specificity_test = test_tn / (test_tn + test_fp)
precision_test = test_tp / (test_tp + test_fp)
accuracy_test = (test_tp + test_tn) / (test_tp + test_tn + test_fp + test_fn)
f1_test = (2 * precision_test * sensitivity_test) / (precision_test + sensitivity_test)

# 결과 출력
print(f"훈련 세트 - 민감도: {sensitivity_train:.3f}, 특이도: {specificity_train:.3f}, 정밀도: {precision_train:.3f}, 정확도: {accuracy_train:.3f}, F1 Score: {f1_train:.3f}")
print(f"테스트 세트 - 민감도: {sensitivity_test:.3f}, 특이도: {specificity_test:.3f}, 정밀도: {precision_test:.3f}, 정확도: {accuracy_test:.3f}, F1 Score: {f1_test:.3f}")


# ##SVM

# In[183]:


from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score

svm_classifier = svm.SVC(kernel='linear', probability=True)

svm_params={'C':[0.001, 0.01, 0.1, 1, 10, 100]}

grid_svm_search = GridSearchCV(svm_classifier, svm_params, cv=5)
grid_svm_search.fit(train_input, train_target)

# 최적의 모델을 사용하여 예측
best_svm_model = grid_svm_search.best_estimator_
train_accuracy = best_svm_model.score(train_input, train_target)
test_accuracy = best_svm_model.score(test_input, test_target)

# 교차 검증을 통한 정확도 평균 계산
cv_scores = cross_val_score(best_svm_model, train_input, train_target, cv=5)
cv_accuracy_mean = cv_scores.mean()

# 최적의 파라미터
best_params = grid_svm_search.best_params_

# 특성 중요도
feature_importances = abs(best_svm_model.coef_)

# 결과 출력
print(f"학습 세트 정확도: {train_accuracy:.4f}")
print(f"테스트 세트 정확도: {test_accuracy:.4f}")
print(f"교차 검증 정확도 평균: {cv_accuracy_mean:.4f}")
print(f"최적의 파라미터: {best_params}")
print(f"특성 중요도: {feature_importances}")


# In[184]:


# ROC 커브 시각화
y_scores = best_svm_model.predict_proba(test_input)[:, 1]
fpr, tpr, thresholds = roc_curve(test_target, y_scores)
roc_auc = roc_auc_score(test_target, y_scores)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of SVM')
plt.legend(loc='lower right')
plt.show()


# In[185]:


# 중요도 기준으로 내림차순 정렬하여 상위 5개 특성 선택
feature_importances=feature_importances[0]
top_features_indices = feature_importances.argsort()[::-1][:5]

# 상위 5개 특성의 이름 출력
top_features_names = train_data.columns[top_features_indices]
print("상위 5개의 중요한 특성:")
for feature_name in top_features_names:
    print(feature_name)


# In[186]:


# 예측값 생성
train_pred = best_svm_model.predict(train_input)
test_pred = best_svm_model.predict(test_input)

# True Positive, True Negative, False Positive, False Negative 계산
def calculate_confusion_matrix(true, pred):
    true_positive = sum((true == 1) & (pred == 1))
    true_negative = sum((true == 0) & (pred == 0))
    false_positive = sum((true == 0) & (pred == 1))
    false_negative = sum((true == 1) & (pred == 0))
    return true_positive, true_negative, false_positive, false_negative

# 훈련 세트에 적용
train_tp, train_tn, train_fp, train_fn = calculate_confusion_matrix(train_target, train_pred)

# 테스트 세트에 적용
test_tp, test_tn, test_fp, test_fn = calculate_confusion_matrix(test_target, test_pred)

# 민감도(recall), 특이도, 정밀도, 정확도, F1 Score 계산
sensitivity_train = train_tp / (train_tp + train_fn)
specificity_train = train_tn / (train_tn + train_fp)
precision_train = train_tp / (train_tp + train_fp)
accuracy_train = (train_tp + train_tn) / (train_tp + train_tn + train_fp + train_fn)
f1_train = (2 * precision_train * sensitivity_train) / (precision_train + sensitivity_train)

sensitivity_test = test_tp / (test_tp + test_fn)
specificity_test = test_tn / (test_tn + test_fp)
precision_test = test_tp / (test_tp + test_fp)
accuracy_test = (test_tp + test_tn) / (test_tp + test_tn + test_fp + test_fn)
f1_test = (2 * precision_test * sensitivity_test) / (precision_test + sensitivity_test)

# 결과 출력
print(f"훈련 세트 - 민감도: {sensitivity_train:.3f}, 특이도: {specificity_train:.3f}, 정밀도: {precision_train:.3f}, 정확도: {accuracy_train:.3f}, F1 Score: {f1_train:.3f}")
print(f"테스트 세트 - 민감도: {sensitivity_test:.3f}, 특이도: {specificity_test:.3f}, 정밀도: {precision_test:.3f}, 정확도: {accuracy_test:.3f}, F1 Score: {f1_test:.3f}")


# ##Logistic Regression

# In[187]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 로지스틱 회귀 모델 생성
logistic_regression = LogisticRegression(max_iter=1000)

# 그리드 서치를 사용하여 최적의 파라미터 찾기
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_lr_search = GridSearchCV(logistic_regression, param_grid, cv=5)
grid_lr_search.fit(train_input, train_target)

# 최적의 모델을 사용하여 예측
best_lr_model = grid_lr_search.best_estimator_
train_accuracy = best_lr_model.score(train_input, train_target)
test_accuracy = best_lr_model.score(test_input, test_target)

# 교차 검증을 통한 정확도 평균 계산
cv_scores = cross_val_score(best_lr_model, train_input, train_target, cv=5)
cv_accuracy_mean = cv_scores.mean()

# 최적의 파라미터
best_params = grid_lr_search.best_params_

# 특성 중요도
feature_importances = best_lr_model.coef_[0]

# 결과 출력
print(f"학습 세트 정확도: {train_accuracy:.4f}")
print(f"테스트 세트 정확도: {test_accuracy:.4f}")
print(f"교차 검증 정확도 평균: {cv_accuracy_mean:.4f}")
print(f"최적의 파라미터: {best_params}")
print(f"특성 중요도: {feature_importances}")


# In[188]:


# ROC 커브 시각화
y_scores = best_lr_model.decision_function(test_input)
fpr, tpr, thresholds = roc_curve(test_target, y_scores)
roc_auc = roc_auc_score(test_target, y_scores)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Logistic Regression')
plt.legend(loc='lower right')
plt.show()


# In[189]:


# 각 특성의 중요도를 기준으로 내림차순 정렬하여 상위 5개의 특성을 선택
top_features_indices = feature_importances.argsort()[::-1][:5]

# 상위 5개 특성의 이름 출력
top_features_names = train_data.columns[top_features_indices]
print("상위 5개의 중요한 특성:")
for feature_name in top_features_names:
    print(feature_name)


# In[190]:


# 예측값 생성
train_pred = best_lr_model.predict(train_input)
test_pred = best_lr_model.predict(test_input)

# True Positive, True Negative, False Positive, False Negative 계산
def calculate_confusion_matrix(true, pred):
    true_positive = sum((true == 1) & (pred == 1))
    true_negative = sum((true == 0) & (pred == 0))
    false_positive = sum((true == 0) & (pred == 1))
    false_negative = sum((true == 1) & (pred == 0))
    return true_positive, true_negative, false_positive, false_negative

# 훈련 세트에 적용
train_tp, train_tn, train_fp, train_fn = calculate_confusion_matrix(train_target, train_pred)

# 테스트 세트에 적용
test_tp, test_tn, test_fp, test_fn = calculate_confusion_matrix(test_target, test_pred)

# 민감도(recall), 특이도, 정밀도, 정확도, F1 Score 계산
sensitivity_train = train_tp / (train_tp + train_fn)
specificity_train = train_tn / (train_tn + train_fp)
precision_train = train_tp / (train_tp + train_fp)
accuracy_train = (train_tp + train_tn) / (train_tp + train_tn + train_fp + train_fn)
f1_train = (2 * precision_train * sensitivity_train) / (precision_train + sensitivity_train)

sensitivity_test = test_tp / (test_tp + test_fn)
specificity_test = test_tn / (test_tn + test_fp)
precision_test = test_tp / (test_tp + test_fp)
accuracy_test = (test_tp + test_tn) / (test_tp + test_tn + test_fp + test_fn)
f1_test = (2 * precision_test * sensitivity_test) / (precision_test + sensitivity_test)

# 결과 출력
print(f"훈련 세트 - 민감도: {sensitivity_train:.3f}, 특이도: {specificity_train:.3f}, 정밀도: {precision_train:.3f}, 정확도: {accuracy_train:.3f}, F1 Score: {f1_train:.3f}")
print(f"테스트 세트 - 민감도: {sensitivity_test:.3f}, 특이도: {specificity_test:.3f}, 정밀도: {precision_test:.3f}, 정확도: {accuracy_test:.3f}, F1 Score: {f1_test:.3f}")


# ##XG 부스트

# In[191]:


from xgboost import XGBClassifier

# XGBoost 모델 생성
xgboost = XGBClassifier(random_state=42)

# 그리드 서치를 사용하여 최적의 파라미터 찾기
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [1,2,3, 4, 5,6,7,8,9,10],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_xgb_search = GridSearchCV(xgboost, param_grid, cv=5)
grid_xgb_search.fit(train_input, train_target)

# 최적의 모델을 사용하여 예측
best_xgb_model = grid_xgb_search.best_estimator_
train_accuracy = best_xgb_model.score(train_input, train_target)
test_accuracy = best_xgb_model.score(test_input, test_target)

# 교차 검증을 통한 정확도 평균 계산
cv_scores = cross_val_score(best_xgb_model, train_input, train_target, cv=5)
cv_accuracy_mean = cv_scores.mean()

# 최적의 파라미터
best_params = grid_xgb_search.best_params_

# 특성 중요도
feature_importances = best_xgb_model.feature_importances_

# 결과 출력
print(f"학습 세트 정확도: {train_accuracy:.4f}")
print(f"테스트 세트 정확도: {test_accuracy:.4f}")
print(f"교차 검증 정확도 평균: {cv_accuracy_mean:.4f}")
print(f"최적의 파라미터: {best_params}")
print(f"특성 중요도: {feature_importances}")


# In[192]:


# ROC 커브 시각화
y_scores = best_xgb_model.predict_proba(test_input)[:, 1]
fpr, tpr, thresholds = roc_curve(test_target, y_scores)
roc_auc = roc_auc_score(test_target, y_scores)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of XGBoost')
plt.legend(loc='lower right')
plt.show()


# In[193]:


# 각 특성의 중요도를 기준으로 내림차순 정렬하여 상위 5개의 특성을 선택
top_features_indices = feature_importances.argsort()[::-1][:5]

# 상위 5개 특성의 이름 출력
top_features_names = train_data.columns[top_features_indices]
print("상위 5개의 중요한 특성:")
for feature_name in top_features_names:
    print(feature_name)


# In[194]:


# 예측값 생성
train_pred = best_xgb_model.predict(train_input)
test_pred = best_xgb_model.predict(test_input)

# True Positive, True Negative, False Positive, False Negative 계산
def calculate_confusion_matrix(true, pred):
    true_positive = sum((true == 1) & (pred == 1))
    true_negative = sum((true == 0) & (pred == 0))
    false_positive = sum((true == 0) & (pred == 1))
    false_negative = sum((true == 1) & (pred == 0))
    return true_positive, true_negative, false_positive, false_negative

# 훈련 세트에 적용
train_tp, train_tn, train_fp, train_fn = calculate_confusion_matrix(train_target, train_pred)

# 테스트 세트에 적용
test_tp, test_tn, test_fp, test_fn = calculate_confusion_matrix(test_target, test_pred)

# 민감도(recall), 특이도, 정밀도, 정확도, F1 Score 계산
sensitivity_train = train_tp / (train_tp + train_fn)
specificity_train = train_tn / (train_tn + train_fp)
precision_train = train_tp / (train_tp + train_fp)
accuracy_train = (train_tp + train_tn) / (train_tp + train_tn + train_fp + train_fn)
f1_train = (2 * precision_train * sensitivity_train) / (precision_train + sensitivity_train)

sensitivity_test = test_tp / (test_tp + test_fn)
specificity_test = test_tn / (test_tn + test_fp)
precision_test = test_tp / (test_tp + test_fp)
accuracy_test = (test_tp + test_tn) / (test_tp + test_tn + test_fp + test_fn)
f1_test = (2 * precision_test * sensitivity_test) / (precision_test + sensitivity_test)

# 결과 출력
print(f"훈련 세트 - 민감도: {sensitivity_train:.3f}, 특이도: {specificity_train:.3f}, 정밀도: {precision_train:.3f}, 정확도: {accuracy_train:.3f}, F1 Score: {f1_train:.3f}")
print(f"테스트 세트 - 민감도: {sensitivity_test:.3f}, 특이도: {specificity_test:.3f}, 정밀도: {precision_test:.3f}, 정확도: {accuracy_test:.3f}, F1 Score: {f1_test:.3f}")


# ### 논문용 roc 그래프(합친거)

# In[196]:


# KNN ROC Curve
y_scores_knn = best_kn_model.predict_proba(test_input)[:, 1]
fpr_knn, tpr_knn, thresholds_knn = roc_curve(test_target, y_scores_knn)
roc_auc_knn = roc_auc_score(test_target, y_scores_knn)

# 랜덤 포레스트 ROC Curve
y_scores_rf = best_rf_model.predict_proba(test_input)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(test_target, y_scores_rf)
roc_auc_rf = roc_auc_score(test_target, y_scores_rf)

# SVM ROC Curve
y_scores_svm = best_svm_model.predict_proba(test_input)[:, 1]
fpr_svm, tpr_svm, thresholds_svm = roc_curve(test_target, y_scores_svm)
roc_auc_svm = roc_auc_score(test_target, y_scores_svm)

# 로지스틱 회귀 ROC Curve
y_scores_lr = best_lr_model.predict_proba(test_input)[:, 1]
fpr_lr, tpr_lr, thresholds_lr = roc_curve(test_target, y_scores_lr)
roc_auc_lr = roc_auc_score(test_target, y_scores_lr)

# XGBoost ROC Curve
y_scores_xgb = best_xgb_model.predict_proba(test_input)[:, 1]
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(test_target, y_scores_xgb)
roc_auc_xgb = roc_auc_score(test_target, y_scores_xgb)

# ROC 커브 시각화
plt.figure(figsize=(10, 10))

plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.4f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.4f})')
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.4f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()

