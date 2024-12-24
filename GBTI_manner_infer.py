import numpy as np
import pandas as pd
import joblib
import random

scaler = joblib.load('scaler.joblib')
al_model = joblib.load('logistic_regression_model_active.joblib')

PLAYER_DATA_PATH = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\output\player_data_enriched.csv'
df = pd.read_csv(PLAYER_DATA_PATH)

df.fillna(0, inplace=True)

df['team_kill_ratio'] = df['team_kills'] / (df['kills'] + 1)
df['road_kill_ratio'] = df['road_kills'] / (df['kills'] + 1)
df['vehicle_destroy_ratio'] = df['vehicle_destroys'] / (df['kills'] + 1)

features = ['team_kill_ratio', 'road_kill_ratio', 'vehicle_destroy_ratio']
X = df[features].copy()

# 임의의 행 인덱스 선택
random_idx = random.randint(0, len(df) - 1)
selected_row = X.iloc[random_idx]

print(f"선택된 행의 인덱스: {random_idx}")
print("선택된 행의 데이터:")
print(selected_row)

# 선택된 행을 2차원 배열로 변환
selected_row_scaled = scaler.transform([selected_row.values])

print("스케일링된 선택된 행의 데이터:")
print(selected_row_scaled)

# 능동 학습 모델을 사용한 예측
predicted_label_al = al_model.predict(selected_row_scaled)[0]
predicted_prob_al = al_model.predict_proba(selected_row_scaled)[0]

label_mapping = {0: 'balance', 1: 'toxic'}

print("\n=== [Active Learning Model Prediction] ===")
print(f"예측 라벨: {label_mapping.get(predicted_label_al, 'Unknown')}")
print(f"예측 확률: {predicted_prob_al}")


###################################
# 선택된 행의 인덱스: 148167
# 선택된 행의 데이터:
# team_kill_ratio          0.0
# road_kill_ratio          0.0
# vehicle_destroy_ratio    0.0
# Name: 148167, dtype: float64
# 스케일링된 선택된 행의 데이터:
# [[-0.06883722 -0.03622833 -0.06362621]]

# === [Active Learning Model Prediction] ===
# 예측 라벨: balance
# 예측 확률: [0.95038612 0.04961388]
# c:\Users\inho0\anaconda3\envs\PUBG\lib\site-packages\sklearn\base.py:493: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
#   warnings.warn(
###################################
