import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib 
import random  
import hdbscan

"""
- KMeans를 사용하여 주 클러스터링 수행
- 각 KMeans 클러스터 내에서 HDBSCAN을 적용하여 세부 클러스터 탐지
- avtive learning을 통해 대표 샘플 선택 및 모델 학습
"""

random.seed(42)

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

PLAYER_DATA_PATH = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\output\player_data_enriched.csv'
df = pd.read_csv(PLAYER_DATA_PATH)

df.fillna(0, inplace=True)

df['team_kill_ratio'] = df['team_kills'] / (df['kills'] + 1)
df['road_kill_ratio'] = 장
joblib.dump(al_model, 'logistic_regression_model_active.joblib')
joblib.dump(rand_model, 'logistic_regression_model_random.joblib')
joblib.dump(scaler, 'scaler.joblib')

#########################################################################
# 초기 라벨링된 샘플 수: 155
# [Active Learning Iter 1] Cumulative number of labeling: 160
# [Active Learning Iter 2] Cumulative number of labeling: 165
# [Active Learning Iter 3] Cumulative number of labeling: 170
# [Active Learning Iter 4] Cumulative number of labeling: 175
# [Active Learning Iter 5] Cumulative number of labeling: 180

# === [Active Learning Model] ===
# [Coefficients]
# team_kill_ratio       : -0.583
# road_kill_ratio       : 0.199
# vehicle_destroy_ratio : -1.118
# Intercept             : -3.057

# [Final distribution - Active Learning]
# dopaminer_type_AL
# balance    170577
# toxic         179
# Name: count, dtype: int64
# [Random Sampling Iter 1] Cumulative number of labeling: 160
# [Random Sampling Iter 2] Cumulative number of labeling: 165
# [Random Sampling Iter 3] Cumulative number of labeling: 170
# [Random Sampling Iter 4] Cumulative number of labeling: 175
# [Random Sampling Iter 5] Cumulative number of labeling: 180

# === [Random Sampling Model] ===
# [Coefficients]
# team_kill_ratio       : -0.627
# road_kill_ratio       : 0.189
# vehicle_destroy_ratio : -0.662
# Intercept             : -3.531

# [Final distribution - Random Sampling]
# dopaminer_type_Random
# balance    170619
# toxic         137
# Name: count, dtype: int64

# =============================
#      Comparison Summary
# =============================
# [Active Learning] final labeled set size : 180
# [Random]         final labeled set size : 180

# [Active Learning] distribution :
# dopaminer_type_AL
# balance    170577
# toxic         179
# Name: count, dtype: int64

# [Random Sampling] distribution :
# dopaminer_type_Random
# balance    170619
# toxic         137
# Name: count, dtype: int64

# [Accuracy vs. KMeans labels (just for reference)]
# Active Learning Model Accuracy: 0.9998
# Random Sampling Model Accuracy: 0.9996

