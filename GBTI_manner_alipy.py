"""
[ALiPy를 사용한 예시 코드 - GMM 분류 후 확률 99% 이상 샘플만 골라 소규모 세트 구성]

1) 데이터 불러오기 + 파생 변수 생성
2) GMM으로 2개 클러스터 분류 + 확률 계산
3) 클러스터 확률 >= 0.99인 '확실' 샘플만 추출 → (X_cert, y_cert)
4) ALiPy AlExperiment로 풀 기반 샘플링 시나리오 구성
5) 불확실성(least_confident) 기반 쿼리 전략 적용
6) 능동학습 반복 후, 학습 곡선 시각화

주의:
- GMM 라벨(0/1)은 실제 라벨과 다를 수 있음.
- 연구/시연 목적 외에는 실제 도메인 라벨을 사용해야 의미있는 결과를 얻음.
"""

import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from alipy.experiment.al_experiment import AlExperiment
from sklearn.linear_model import LogisticRegression

# =======================================
# 1) 데이터 불러오기 및 파생 변수 생성
# =======================================
PLAYER_DATA_PATH = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\output\player_data_enriched.csv'
df = pd.read_csv(PLAYER_DATA_PATH)

# 간단한 파생 변수: kills 대비 팀킬/도로킬/차량파괴 비율
df['team_kill_ratio'] = df['team_kills'] / (df['kills'] + 1)
df['road_kill_ratio'] = df['road_kills'] / (df['kills'] + 1)
df['vehicle_destroy_ratio'] = df['vehicle_destroys'] / (df['kills'] + 1)

features = ['team_kill_ratio', 'road_kill_ratio', 'vehicle_destroy_ratio']
X_raw = df[features].values

# =======================================
# 2) GMM으로 2개 클러스터 분류 + 확률 계산
# =======================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

gmm = GaussianMixture(n_components=2, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
gmm_probs = gmm.predict_proba(X_scaled)  # shape: (N, 2)

# 샘플별 최대 확률과 argmax(실제 클러스터 라벨)
max_probs = gmm_probs.max(axis=1)        # 각 샘플에서 가장 높은 클러스터 확률
assigned_clusters = gmm_probs.argmax(axis=1)  # 0 또는 1

# =======================================
# 3) 확실한 샘플(>= 0.99)만 추출 -> (X_cert, y_cert)
# =======================================
threshold = 0.99
mask_certain = (max_probs >= threshold)

# X_cert: 확실 샘플들의 피처
X_cert = X_raw[mask_certain]
# y_cert: 확실 샘플들의 클러스터 (0 or 1)
y_cert = assigned_clusters[mask_certain]

print(f"[INFO] 전체 데이터 크기: {len(X_raw)}")
print(f"[INFO] 확실(>= {threshold*100}%) 샘플 수: {len(X_cert)}")

# 만약 확실 샘플이 너무 적다면, threshold를 낮춰서(예: 0.95) 확보량 조절 가능
# (주의) cluster=0이면 '정상', cluster=1이면 '비매너'라는 가정이지만, 실제론 무의미할 수 있음.

# =======================================
# 4) ALiPy AlExperiment 설정
# =======================================
# AlExperiment는 X와 y가 모두 있어야 함.
# 여기서는 "확실 샘플"만으로 구성된 (X_cert, y_cert) 사용.
al = AlExperiment(
    X=X_cert,
    y=y_cert,
    stopping_criteria='num_of_queries',
    stopping_value=10,  # 예: 최대 10번 쿼리
    random_state=42
)

# 데이터 분할
# 기본값(0.3) 비율로 테스트를 만들고, initial_label_rate=0.05 등...
# 필요에 따라 수정 가능
al.split_AL(test_ratio=0.3, initial_label_rate=0.1)

# =======================================
# 5) 불확실성 기반 쿼리 전략 설정
# =======================================
# query_strategy="QueryInstanceUncertainty"
# measure='least_confident', 'margin', 'entropy' 등 가능
al.set_query_strategy(
    strategy="QueryInstanceUncertainty",
    measure='least_confident'
)

# 분류 성능 지표 설정 (정확도)
al.set_performance_metric('accuracy_score')

# =======================================
# 6) 능동학습(Active Learning) 실행
# =======================================
al.start_query(multi_thread=False)

# 결과 시각화(학습 곡선)
al.plot_learning_curve()

# =======================================
# 결과 해석
# =======================================
"""
- 이 코드는 GMM이 '99% 이상 확률'로 클러스터를 잘 구분한 샘플만 추출했으므로,
  클러스터 라벨(0/1)을 실제 y로 사용.

- 실제로는 GMM 라벨이 진짜 비매너/정상을 정확히 반영한다는 보장이 없으므로,
  연구 시뮬레이션이나 ALiPy 사용법 학습 정도의 목적으로만 의미가 있음.

- 확실 샘플이 많지 않다면 능동학습 반복에서 훈련/테스트 구분이 불안정할 수 있음.
  => threshold를 낮추거나, 추가로 불확실 구간을 일부 라벨링하는 전략 고려.

- 만약 실제 비매너 여부 라벨이 존재한다면, y_cert 대신 그 라벨을 사용해야
  모델 평가와 능동학습 결과가 실제 목적에 부합함.
"""


# [INFO] 전체 데이터 크기: 170756
# [INFO] 확실(>= 99.0%) 샘플 수: 170756

# | round | initially labeled data | number of queries | cost | Performance: |
# |   0   |  11953 (10.00% of all) |         10        |  0   | 0.995 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   1   |  11953 (10.00% of all) |         10        |  0   | 0.995 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   2   |  11953 (10.00% of all) |         10        |  0   | 0.995 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   3   |  11953 (10.00% of all) |         10        |  0   | 0.994 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   4   |  11953 (10.00% of all) |         10        |  0   | 0.995 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   5   |  11953 (10.00% of all) |         10        |  0   | 0.994 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   6   |  11953 (10.00% of all) |         10        |  0   | 0.995 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   7   |  11953 (10.00% of all) |         10        |  0   | 0.995 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   8   |  11953 (10.00% of all) |         10        |  0   | 0.995 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   9   |  11953 (10.00% of all) |         10        |  0   | 0.994 ± 0.00 |
# +--------------------------+-------------------+---------------------------+--------------+------------+
# |         Methods          | number_of_queries | number_of_different_split | performance  | batch_size |
# +--------------------------+-------------------+---------------------------+--------------+------------+
# | QueryInstanceUncertainty |         10        |             10            | 0.995 ± 0.00 |     1      |
# +--------------------------+-------------------+---------------------------+--------------+------------+
