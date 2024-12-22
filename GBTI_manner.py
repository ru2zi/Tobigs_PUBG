import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib 
import random  

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

PLAYER_DATA_PATH = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\output\player_data_enriched.csv'
df = pd.read_csv(PLAYER_DATA_PATH)

df['team_kill_ratio'] = df['team_kills'] / (df['kills'] + 1)
df['road_kill_ratio'] = df['road_kills'] / (df['kills'] + 1)
df['vehicle_destroy_ratio'] = df['vehicle_destroys'] / (df['kills'] + 1)

features = ['team_kill_ratio', 'road_kill_ratio', 'vehicle_destroy_ratio']
X = df[features].copy()

# =======================================
# 데이터 전처리 및 클러스터링 (GMM)
# =======================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(n_components=2, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
df['gmm_cluster'] = gmm_labels  # GMM 클러스터 레이블 저장

# 실루엣 점수 계산 (1에 가까울수록 클러스터가 잘 분리됨)
sil_score = silhouette_score(X_scaled, gmm_labels)
print(f"[Clustering] Silhouette Score: {sil_score:.4f}")

# =======================================
# 클러스터별 대표 샘플 선정 (초기 라벨링)
# =======================================
# GMM에서 각 샘플이 각 클러스터에 속할 확률
probs = gmm.predict_proba(X_scaled)  # shape: (N, 2)
n_clusters = gmm.n_components
rep_per_cluster = 10  # 클러스터당 대표 샘플 개수

initial_labeled_indices = []
for c in range(n_clusters):
    # c번 클러스터 확률이 높은 순서대로 정렬 (내림차순)
    cluster_probs = probs[:, c]
    sorted_idx = np.argsort(cluster_probs)[::-1]
    chosen_indices = sorted_idx[:rep_per_cluster]
    initial_labeled_indices.extend(chosen_indices)

# 중복 제거
initial_labeled_indices = list(set(initial_labeled_indices))

# 전체 인덱스 목록
all_indices = list(range(len(X)))
unlabeled_indices = list(set(all_indices) - set(initial_labeled_indices))

# 초기 라벨 -> GMM 레이블을 임시로 사용
initial_labels = gmm_labels[initial_labeled_indices]

# =======================================
# 불확실성 기반 샘플 선택 함수 (능동학습)
# =======================================
def select_uncertain_samples(model, X_data, unlabeled_idx, batch_size=5):
    """
    모델이 예측에 불확실한(=예측 확률이 0.5 근처인) 샘플 상위 batch_size개를 선택하는 함수.
    """
    # 아직 라벨이 없는(unlabeled_idx) 샘플에 대한 예측 확률
    probs_unlab = model.predict_proba(X_data[unlabeled_idx])[:, 1]
    
    # |p - 0.5|가 작을수록 불확실성이 크다
    uncertainty = np.abs(probs_unlab - 0.5)
    
    # 오름차순 정렬 → 앞쪽이 불확실성 큰 샘플
    sorted_idx = np.argsort(uncertainty)
    
    # 상위 batch_size개 로컬 인덱스
    chosen_local = sorted_idx[:batch_size]
    
    # 실제 전역 인덱스 변환
    chosen_global = [unlabeled_idx[i] for i in chosen_local]
    return chosen_global

# =======================================
# (A) 능동학습(Active Learning) 방식
# =======================================
num_iterations = 5
batch_size = 5

al_labeled_indices = initial_labeled_indices.copy()
al_labels = initial_labels.copy()
al_unlabeled_indices = unlabeled_indices.copy()

al_model = LogisticRegression(solver='lbfgs', random_state=42)
al_model.fit(X_scaled[al_labeled_indices], al_labels)

for it in range(num_iterations):
    chosen_idx = select_uncertain_samples(
        model=al_model,
        X_data=X_scaled,
        unlabeled_idx=al_unlabeled_indices,
        batch_size=batch_size
    )
    
    # (가정) 여기서는 GMM 레이블을 "진짜 라벨"처럼 사용
    new_labels = gmm_labels[chosen_idx]
    al_labeled_indices.extend(chosen_idx)
    al_labels = np.concatenate([al_labels, new_labels])
    
    for idx in chosen_idx:
        al_unlabeled_indices.remove(idx)
    
    al_model.fit(X_scaled[al_labeled_indices], al_labels)
    print(f"[Active Learning Iter {it+1}] Cumulative number of labeling: {len(al_labeled_indices)}")

al_coefs = al_model.coef_[0]
al_intercept = al_model.intercept_[0]

print("\n=== [Active Learning Model] ===")
print("[Coefficients]")
print(f"team_kill_ratio       : {al_coefs[0]:.3f}")
print(f"road_kill_ratio       : {al_coefs[1]:.3f}")
print(f"vehicle_destroy_ratio : {al_coefs[2]:.3f}")
print(f"Intercept             : {al_intercept:.3f}")

# 예측 결과 저장
al_preds = al_model.predict(X_scaled)
df['dopaminer_type_AL'] = al_preds
df['dopaminer_type_AL'] = df['dopaminer_type_AL'].map({0: 'balance', 1: 'toxic'})

print("\n[Final distribution - Active Learning]")
print(df['dopaminer_type_AL'].value_counts())

# =======================================
# (B) 랜덤 샘플링 방식 (No Active Learning)
# =======================================
# 같은 반복 횟수와 배치 크기를 사용하지만 샘플을 무작위로 선택

num_iterations_rand = num_iterations
batch_size_rand = batch_size

rand_labeled_indices = initial_labeled_indices.copy()
rand_labels = initial_labels.copy()
rand_unlabeled_indices = unlabeled_indices.copy()

rand_model = LogisticRegression(solver='lbfgs', random_state=42)
rand_model.fit(X_scaled[rand_labeled_indices], rand_labels)

for it in range(num_iterations_rand):
    # unlabeled_indices가 batch_size보다 작으면 전부 선택
    if len(rand_unlabeled_indices) <= batch_size_rand:
        chosen_idx_rand = rand_unlabeled_indices
    else:
        # 무작위로 batch_size_rand개 선택
        chosen_idx_rand = random.sample(rand_unlabeled_indices, batch_size_rand)
    
    new_labels_rand = gmm_labels[chosen_idx_rand]
    rand_labeled_indices.extend(chosen_idx_rand)
    rand_labels = np.concatenate([rand_labels, new_labels_rand])
    
    for idx in chosen_idx_rand:
        rand_unlabeled_indices.remove(idx)
    
    rand_model.fit(X_scaled[rand_labeled_indices], rand_labels)
    print(f"[Random Sampling Iter {it+1}] Cumulative number of labeling: {len(rand_labeled_indices)}")

# 최종 모델 계수 확인
rand_coefs = rand_model.coef_[0]
rand_intercept = rand_model.intercept_[0]

print("\n=== [Random Sampling Model] ===")
print("[Coefficients]")
print(f"team_kill_ratio       : {rand_coefs[0]:.3f}")
print(f"road_kill_ratio       : {rand_coefs[1]:.3f}")
print(f"vehicle_destroy_ratio : {rand_coefs[2]:.3f}")
print(f"Intercept             : {rand_intercept:.3f}")

# 예측 결과 저장
rand_preds = rand_model.predict(X_scaled)
df['dopaminer_type_Random'] = rand_preds
df['dopaminer_type_Random'] = df['dopaminer_type_Random'].map({0: 'balance', 1: 'toxic'})

print("\n[Final distribution - Random Sampling]")
print(df['dopaminer_type_Random'].value_counts())

# =======================================
# 비교 결과 요약
# =======================================
print("\n=============================")
print("     Comparison Summary")
print("=============================")
print("[Active Learning] final labeled set size :", len(al_labeled_indices))
print("[Random]         final labeled set size :", len(rand_labeled_indices))

print("\n[Active Learning] distribution :")
print(df['dopaminer_type_AL'].value_counts())
print("\n[Random Sampling] distribution :")
print(df['dopaminer_type_Random'].value_counts())

# GMM 레이블을 기준으로 "정확도" 비교 (주의: 실제 정답이 아님)
al_acc = accuracy_score(gmm_labels, al_preds)
rand_acc = accuracy_score(gmm_labels, rand_preds)
print("\n[Accuracy vs. GMM labels (just for reference)]")
print(f"Active Learning Model Accuracy: {al_acc:.4f}")
print(f"Random Sampling Model Accuracy: {rand_acc:.4f}")

# =======================================
# GMM 멤버십 확률 기반, 확실한 샘플만 골라 정확도 비교
# =======================================
print("\n[High-Confidence Sample Evaluation]")
gmm_probs = gmm.predict_proba(X_scaled)
max_probs = gmm_probs.max(axis=1)           # 각 샘플에서 가장 높은 클러스터 확률
assigned_clusters = gmm_probs.argmax(axis=1)

threshold = 0.9  # 예: 0.9 이상이면 '확실'
high_conf_mask = (
    (assigned_clusters == gmm_labels) &  # GMM 할당 클러스터와 argmax가 동일
    (max_probs >= threshold)            # 최대 확률이 threshold 이상
)

high_conf_indices = np.where(high_conf_mask)[0]
print(f"Threshold set at: {threshold}")
print(f"Number of high-confidence samples: {len(high_conf_indices)}")

# 능동학습 vs 랜덤 샘플링 모델 예측 비교
al_preds_highconf = al_model.predict(X_scaled[high_conf_indices])
rand_preds_highconf = rand_model.predict(X_scaled[high_conf_indices])

al_true_highconf = gmm_labels[high_conf_indices]   # 이 샘플들에 한해 GMM 레이블을 '정답'으로
rand_true_highconf = gmm_labels[high_conf_indices]

al_acc_highconf = accuracy_score(al_true_highconf, al_preds_highconf)
rand_acc_highconf = accuracy_score(rand_true_highconf, rand_preds_highconf)

print(f"Active Learning Model Accuracy (High Conf Only): {al_acc_highconf:.4f}")
print(f"Random Sampling Model Accuracy (High Conf Only): {rand_acc_highconf:.4f}")

# =======================================
# 모델과 스케일러 저장 
# =======================================
joblib.dump(al_model, 'logistic_regression_model_active.joblib')
joblib.dump(rand_model, 'logistic_regression_model_random.joblib')
joblib.dump(scaler, 'scaler.joblib')
