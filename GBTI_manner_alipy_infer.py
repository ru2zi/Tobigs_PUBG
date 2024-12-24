import numpy as np
import pandas as pd
import joblib
import random

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = joblib.load('al_scaler.joblib')
al_experiment = joblib.load('model_active.joblib')

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

# 선택된 행을 2차원 배열로 변환 및 스케일링
selected_row_scaled = scaler.transform([selected_row.values])

print("스케일링된 선택된 행의 데이터:")
print(selected_row_scaled)

# AlExperiment 객체에서 기본 분류기 추출
base_classifier = al_experiment._model

# 기본 분류기를 사용한 예측
predicted_label_al = base_classifier.predict(selected_row_scaled)[0]
predicted_prob_al = base_classifier.predict_proba(selected_row_scaled)[0]

label_mapping = {0: 'balance', 1: 'toxic'}

# 능동 학습 모델 결과 출력
print("\n=== [Active Learning Model Prediction] ===")
print(f"예측 라벨: {label_mapping.get(predicted_label_al, 'Unknown')}")
print(f"예측 확률: {predicted_prob_al}")

# 선택된 행의 원본 데이터 및 예측 결과를 데이터프레임에 추가하여 확인
selected_data = df.iloc[random_idx].copy()
selected_data['predicted_label'] = label_mapping.get(predicted_label_al, 'Unknown')
selected_data['predicted_prob_balance'] = predicted_prob_al[0]
selected_data['predicted_prob_toxic'] = predicted_prob_al[1]

print("\n=== [Selected Data with Prediction] ===")
print(selected_data)

##################
# 선택된 행의 인덱스: 66594
# 선택된 행의 데이터:
# team_kill_ratio          0.0
# road_kill_ratio          0.0
# vehicle_destroy_ratio    0.0
# Name: 66594, dtype: float64
# 스케일링된 선택된 행의 데이터:
# [[-0.06883722 -0.03622833 -0.06362621]]

# === [Active Learning Model Prediction] ===
# 예측 라벨: balance
# 예측 확률: [0.99853838 0.00146162]

# === [Selected Data with Prediction] ===
# match_id                                 7a521d29-9655-4ed1-b490-fdae5b940c56
# map_name                                                          Savage_Main
# game_mode                                                               squad
# player_id                                3cc5e1c8-5fd8-4da2-92d1-46d204db5833
# player_name                                                   gksrmfdksehl123
# player_account_id                    account.1323301c1a8c4179a1eaed082333334c
# primary_weapon                                                              0
# secondary_weapon                                                            0
# armor_type                                                                0.0
# use_of_health_items                                                         0
# use_of_boost_items                                                          0
# items_carried               Item_Back_B_01_StartParachutePack_C, Item_Ammo...
# time_spent_looting_sec                                                 67.853
# time_spent_in_combat_sec                                                8.223
# kills                                                                       0
# damage_dealt                                                              0.0
# movement_routes             (345009.53125,228684.453125,615.9566040039062)...
# first_location_x                                                 345009.53125
# first_location_y                                                228684.453125
# first_location_z                                                   615.956604
# final_location_x                                                     231485.5
# final_location_y                                                 78476.921875
# final_location_z                                                   284.079987
# walk_distance                                                        25.50244
# swim_distance                                                             0.0
# ride_distance                                                             0.0
# road_kills                                                                  0
# vehicle_destroys                                                            0
# weapons_acquired                                                            0
# boosts                                                                      0
# heals                                                                       0
# kill_streaks                                                                0
# headshot_kills                                                              0
# assists                                                                     0
# revives                                                                     0
# team_kills                                                                  0
# win_place                                                                  22
# team_id                                                                  28.0
# team_rank                                                                22.0
# team_won                                                                False
# elapsedTime                                                               0.0
# numAlivePlayers                                                          96.0
# user_id                                                               EBBUNYA
# match_start_datetime                                      2024-11-20 05:47:48
# match_date                                                         2024-11-20
# match_duration                                                           1496
# team_kill_ratio                                                           0.0
# road_kill_ratio                                                           0.0
# vehicle_destroy_ratio                                                     0.0
# predicted_label                                                       balance
# predicted_prob_balance                                               0.998538
# predicted_prob_toxic                                                 0.001462
# Name: 66594, dtype: object
