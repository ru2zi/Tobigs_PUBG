import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import hdbscan

# ======================================
# 파생 변수 생성
# ======================================
# 각 특징에 대해 비율을 계산
df['team_kill_ratio'] = df['team_kills'] / (df['kills'] + 1)
df['road_kill_ratio'] = df['road_kills'] / (df['kills'] + 1)
df['vehicle_destroy_ratio'] = df['vehicle_destroys'] / (df['kills'] + 1)

# 사용할 특징 선택
features = ['team_kill_ratio', 'road_kill_ratio', 'vehicle_destroy_ratio']
X = df[features]

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================
# HDBSCAN 클러스터링 수행
# ======================================
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=10, prediction_data=True)
df['hdbscan_cluster'] = hdbscan_clusterer.fit_predict(X_scaled)

# ======================================
# GMM 클러스터링
# ======================================
gmm = GaussianMixture(n_components=2, random_state=42)
df['gmm_cluster'] = gmm.fit_predict(X_scaled)

# ======================================
# 경계 도출 함수 정의
# ======================================

def calculate_boundary(model, X, y, method_name):
    """
    주어진 모델로 경계를 계산하고 수식을 출력하는 함수
    model: 사용할 분류 모델 (e.g., LogisticRegression, SVM)
    X: 특징 데이터
    y: 레이블 데이터
    method_name: 모델 이름 (e.g., "Logistic Regression", "SVM")
    """
    model.fit(X, y)  # 모델 학습
    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):  # 선형 모델의 경우
        coefficients = model.coef_[0]
        intercept = model.intercept_[0]
        print(f"[{method_name}] Linear Boundary Equation:")
        print(f"{coefficients[0]:.3f} * team_kill_ratio + "
              f"{coefficients[1]:.3f} * road_kill_ratio + "
              f"{coefficients[2]:.3f} * vehicle_destroy_ratio = {-intercept:.3f}\n")
    else:
        print(f"[{method_name}] 선형 경계식 생성 불가 (비선형 모델일 수 있음)\n")

# ======================================
# HDBSCAN 클러스터를 기반으로 경계 도출
# ======================================
# HDBSCAN 결과를 사용하여 Logistic Regression 및 SVM 적용
print("=== HDBSCAN 기반 경계 도출 ===")
dbscan_labels = df['dbscan_cluster']
if len(np.unique(dbscan_labels)) > 1:  # 두 개 이상의 클러스터가 있는 경우에만 경계 도출
    calculate_boundary(LogisticRegression(), X_scaled, hdbscan_labels, "Logistic Regression (DBSCAN)")
    calculate_boundary(SVC(kernel='linear'), X_scaled, hdbscan_labels, "SVM (HDBSCAN)")
else:
    print("DBSCAN에서 유효한 클러스터를 찾을 수 없습니다.\n")

# ======================================
# GMM 클러스터를 기반으로 경계 도출
# ======================================
# GMM 결과를 사용하여 Logistic Regression 및 SVM 적용
print("=== GMM 기반 경계 도출 ===")
gmm_labels = df['gmm_cluster']
if len(np.unique(gmm_labels)) > 1:  # 두 개 이상의 클러스터가 있는 경우에만 경계 도출
    calculate_boundary(LogisticRegression(), X_scaled, gmm_labels, "Logistic Regression (GMM)")
    calculate_boundary(SVC(kernel='linear'), X_scaled, gmm_labels, "SVM (GMM)")
else:
    print("GMM에서 유효한 클러스터를 찾을 수 없습니다.\n")

