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
- Active learning을 통해 대표 샘플 선택 및 모델 학습
"""

random.seed(42)

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

PLAYER_DATA_PATH = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\output\player_data_enriched.csv'
df = pd.read_csv(PLAYER_DATA_PATH)

df.fillna(0, inplace=True)

df['team_kill_ratio'] = df['team_kills'] / (df['kills'] + 1)
df['road_kill_ratio'] = df['road_kills'] / (df['kills'] + 1)
df['vehicle_destroy_ratio'] = df['vehicle_destroys'] / (df['kills'] + 1)

features = ['team_kill_ratio', 'road_kill_ratio', 'vehicle_destroy_ratio']
X = df[features].copy()

# =======================================
# 데이터 전처리 및 클러스터링 (KMeans)
# =======================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans 클러스터링 수행
# n_init=10로 설정하여 초기 중심값을 여러 번 시도
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
df['kmeans_cluster'] = kmeans_labels  

sil_score_kmeans = silhouette_score(X_scaled, kmeans_labels)
print(f"[Clustering] KMeans Silhouette Score: {sil_score_kmeans:.4f}")

# =======================================
# 클러스터별 HDBSCAN을 통한 세부 클러스터링 및 대표 샘플 선정
# =======================================

n_clusters_kmeans = len(np.unique(kmeans_labels))
print(f"KMeans로 형성된 클러스터 수: {n_clusters_kmeans}")
df['hdbscan_subcluster'] = -1

subcluster_counts = {}

for cluster in range(n_clusters_kmeans):
    cluster_data = X_scaled[kmeans_labels == cluster]

    # HDBSCAN 클러스터링 수행
    # min_cluster_size=10, min_samples=5는 파라미터 샘플
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
    hdbscan_labels = hdbscan_model.fit_predict(cluster_data)

    # 클러스터 수 확인 (노이즈 레이블 -1 제외)
    unique_hdbscan_labels = set(hdbscan_labels)
    if -1 in unique_hdbscan_labels:
        n_subclusters = len(unique_hdbscan_labels) - 1
    else:
        n_subclusters = len(unique_hdbscan_labels)

    subcluster_counts[cluster] = n_subclusters
    print(f"KMeans 클러스터 {cluster} 내 HDBSCAN 클러스터 수: {n_subclusters}")
    
    df.loc[kmeans_labels == cluster, 'hdbscan_subcluster'] = hdbscan_labels + (cluster * 1000)

###############################################
# [Clustering] KMeans Silhouette Score: 0.9901
# KMeans로 형성된 클러스터 수: 2
# KMeans 클러스터 0 내 HDBSCAN 클러스터 수: 26
# KMeans 클러스터 1 내 HDBSCAN 클러스터 수: 4
##############################################

# =======================================
# 초기 라벨링을 위한 대표 샘플 선정 
# =======================================
def select_representative_samples_hdbscan(X_scaled, labels, n_samples_per_subcluster=5):
    """
    HDBSCAN 클러스터 레이블에 기반해, 각 세부 클러스터에서 대표 샘플 인덱스를 추출하는 함수.
    - 노이즈 레이블(-1)은 제외
    - 각 세부 클러스터의 중심에서 가까운 순으로 n_samples_per_subcluster개를 선택
    """
    representative_indices = []
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            # 노이즈는 제외
            continue

        subcluster_indices = np.where(labels == label)[0]
        if len(subcluster_indices) == 0:
            continue

        # 중심점 계산
        cluster_center = X_scaled[subcluster_indices].mean(axis=0)
        distances = np.linalg.norm(X_scaled[subcluster_indices] - cluster_center, axis=1)
        sorted_indices = subcluster_indices[np.argsort(distances)]

        num_to_select = min(n_samples_per_subcluster, len(sorted_indices))
        chosen_indices = sorted_indices[:num_to_select]
        representative_indices.extend(chosen_indices)

    return list(set(representative_indices))

# 초기 라벨링을 위한 대표 샘플 선정
initial_labeled_indices = select_representative_samples_hdbscan(
    X_scaled,
    df['hdbscan_subcluster'],
    n_samples_per_subcluster=5
)

# 전체 인덱스 목록과 미라벨링 목록 정의
all_indices = list(range(len(X)))
unlabeled_indices = list(set(all_indices) - set(initial_labeled_indices))

# =======================================
# KMeans와 HDBSCAN이 모두 일치하는 샘플의 라벨 사용
# =======================================
def assign_labels_based_on_clustering(kmeans_labels, hdbscan_subcluster_labels):
    """
    KMeans 클러스터와 HDBSCAN 서브클러스터에 기반하여 라벨을 할당하는 함수.
    KMeans 클러스터 0 내 HDBSCAN 서브클러스터 0은 라벨 0,
    KMeans 클러스터 1 내 HDBSCAN 서브클러스터 1000은 라벨 1 등으로 매핑.
    """
    label_mapping = {}
    unique_subclusters = set(hdbscan_subcluster_labels)

    for subcluster in unique_subclusters:
        if subcluster == -1:
            continue  # 노이즈는 제외
        # KMeans 클러스터 번호 추출 (hdbscan_subcluster_labels = hdbscan_labels + (cluster * 1000))
        kmeans_cluster = subcluster // 1000
        label_mapping[subcluster] = kmeans_cluster

    # 전체 라벨 할당
    labels = np.full_like(kmeans_labels, fill_value=-1)
    for subcluster, label in label_mapping.items():
        labels[hdbscan_subcluster_labels == subcluster] = label

    return labels

# 실제 라벨(y)이 존재하지 않으므로, 클러스터링 기반 임시 라벨을 사용
y = assign_labels_based_on_clustering(kmeans_labels, df['hdbscan_subcluster'])

# 초기 라벨을 KMeans 라벨로 설정 
initial_labels = y[initial_labeled_indices]

print(f"초기 라벨링된 샘플 수: {len(initial_labeled_indices)}")

# =======================================
# 불확실성 기반 샘플 선택 함수 (능동학습)
# =======================================
def select_uncertain_samples(model, X_data, unlabeled_idx, batch_size=5):
    """
    모델이 예측에 불확실한(=예측 확률이 0.5 근처인) 샘플 중
    상위 batch_size개를 골라 전역 인덱스로 반환하는 함수.
    """
    # 아직 라벨이 없는 샘플들(unlabeled_idx)에 대한 예측 확률
    probs_unlab = model.predict_proba(X_data[unlabeled_idx])[:, 1]

    # |p - 0.5|가 작을수록 불확실성이 큼
    uncertainty = np.abs(probs_unlab - 0.5)

    # 오름차순 정렬 → 앞쪽이 불확실성 큰 샘플
    sorted_idx = np.argsort(uncertainty)

    # 상위 batch_size개 로컬 인덱스
    chosen_local = sorted_idx[:batch_size]

    # 실제 전역 인덱스로 변환
    chosen_global = [unlabeled_idx[i] for i in chosen_local]
    return chosen_global

# =======================================
# 능동학습(Active Learning) 방식
# =======================================
num_iterations = 5
batch_size = 5

al_labeled_indices = initial_labeled_indices.copy()
al_labels = initial_labels.copy()
al_unlabeled_indices = unlabeled_indices.copy()

al_model = LogisticRegression(solver='lbfgs', random_state=42, max_iter=1000)
al_model.fit(X_scaled[al_labeled_indices], al_labels)

for it in range(num_iterations):
    # 불확실성 기반 샘플 선택
    chosen_idx = select_uncertain_samples(
        model=al_model,
        X_data=X_scaled,
        unlabeled_idx=al_unlabeled_indices,
        batch_size=batch_size
    )

    # 클러스터링 기반 임시 라벨 사용
    new_labels = y[chosen_idx]

    al_labeled_indices.extend(chosen_idx)
    al_labels = np.concatenate([al_labels, new_labels])

    # 선택된 샘플은 미라벨링 목록에서 제거
    for idx in chosen_idx:
        al_unlabeled_indices.remove(idx)

    # 모델 재학습
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

al_preds = al_model.predict(X_scaled)
df['dopaminer_type_AL'] = al_preds
df['dopaminer_type_AL'] = df['dopaminer_type_AL'].map({0: 'balance', 1: 'toxic'})

print("\n[Final distribution - Active Learning]")
print(df['dopaminer_type_AL'].value_counts())

# =======================================
# 랜덤 샘플링 방식 (No Active Learning)
# =======================================
num_iterations_rand = num_iterations
batch_size_rand = batch_size

rand_labeled_indices = initial_labeled_indices.copy()
rand_labels = initial_labels.copy()
rand_unlabeled_indices = unlabeled_indices.copy()

rand_model = LogisticRegression(solver='lbfgs', random_state=42, max_iter=1000)
rand_model.fit(X_scaled[rand_labeled_indices], rand_labels)

for it in range(num_iterations_rand):
    # unlabeled_indices가 batch_size보다 작으면 전부 선택
    if len(rand_unlabeled_indices) <= batch_size_rand:
        chosen_idx_rand = rand_unlabeled_indices
    else:
        # 무작위로 batch_size_rand개 선택
        chosen_idx_rand = random.sample(rand_unlabeled_indices, batch_size_rand)

    # 클러스터링 기반 임시 라벨 사용
    new_labels_rand = y[chosen_idx_rand]
    rand_labeled_indices.extend(chosen_idx_rand)
    rand_labels = np.concatenate([rand_labels, new_labels_rand])

    for idx in chosen_idx_rand:
        rand_unlabeled_indices.remove(idx)

    rand_model.fit(X_scaled[rand_labeled_indices], rand_labels)
    print(f"[Random Sampling Iter {it+1}] Cumulative number of labeling: {len(rand_labeled_indices)}")

rand_coefs = rand_model.coef_[0]
rand_intercept = rand_model.intercept_[0]

print("\n=== [Random Sampling Model] ===")
print("[Coefficients]")
print(f"team_kill_ratio       : {rand_coefs[0]:.3f}")
print(f"road_kill_ratio       : {rand_coefs[1]:.3f}")
print(f"vehicle_destroy_ratio : {rand_coefs[2]:.3f}")
print(f"Intercept             : {rand_intercept:.3f}")

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

# KMeans 레이블을 기준으로 "정확도" 비교 (주의: 실제 정답이 아님)
if y is not None:
    al_acc = accuracy_score(kmeans_labels, al_preds)
    rand_acc = accuracy_score(kmeans_labels, rand_preds)
    print("\n[Accuracy vs. KMeans labels (just for reference)]")
    print(f"Active Learning Model Accuracy: {al_acc:.4f}")
    print(f"Random Sampling Model Accuracy: {rand_acc:.4f}")

# =======================================
# 모델과 스케일러 저장 
# =======================================
joblib.dump(al_model, 'logistic_regression_model_active.joblib')
joblib.dump(rand_model, 'logistic_regression_model_random.joblib')
joblib.dump(scaler, 'scaler.joblib')

##########################################################
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

