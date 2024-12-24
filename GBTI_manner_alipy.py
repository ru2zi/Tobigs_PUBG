import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from alipy.experiment.al_experiment import AlExperiment
import matplotlib.pyplot as plt
import joblib

PLAYER_DATA_PATH = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\output\player_data_enriched.csv'

df = pd.read_csv(PLAYER_DATA_PATH)
df.fillna(0, inplace=True)

df['team_kill_ratio'] = df['team_kills'] / (df['kills'] + 1)
df['road_kill_ratio'] = df['road_kills'] / (df['kills'] + 1)
df['vehicle_destroy_ratio'] = df['vehicle_destroys'] / (df['kills'] + 1)

features = ['team_kill_ratio', 'road_kill_ratio', 'vehicle_destroy_ratio']
X_raw = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 각 KMeans 클러스터 내에서 HDBSCAN 서브 클러스터링 수행
hdbscan_subcluster_labels = np.full(len(X_scaled), -1)

for cluster_id in np.unique(kmeans_labels):
    cluster_indices = np.where(kmeans_labels == cluster_id)[0]
    cluster_data = X_scaled[cluster_indices]
    hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
    sub_labels = hdb.fit_predict(cluster_data)
    
    # 주요 서브클러스터(가장 많은 샘플을 가진 서브클러스터) 선택
    unique_subclusters, counts = np.unique(sub_labels, return_counts=True)
    if len(unique_subclusters) == 0:
        continue  # 모든 샘플이 노이즈인 경우
    main_subcluster = unique_subclusters[np.argmax(counts)]
    
    # 주요 서브클러스터는 KMeans 클러스터 라벨로 설정, 나머지는 노이즈(-1)로 설정
    hdbscan_subcluster_labels[cluster_indices] = np.where(sub_labels == main_subcluster, cluster_id, -1)

# 클러스터링 기반 라벨 할당 함수
def assign_labels_based_on_clustering(kmeans_labels, hdbscan_subcluster_labels):
    """
    KMeans 클러스터와 HDBSCAN 서브클러스터에 기반하여 라벨을 할당하는 함수.
    HDBSCAN이 주요 서브클러스터로 판단한 샘플은 KMeans 라벨을 그대로 사용,
    그렇지 않은 샘플은 -1(노이즈)로 설정.
    """
    labels = np.full_like(kmeans_labels, fill_value=-1)
    labels[hdbscan_subcluster_labels != -1] = kmeans_labels[hdbscan_subcluster_labels != -1]
    return labels

# 클러스터링 기반 임시 라벨 사용
y = assign_labels_based_on_clustering(kmeans_labels, hdbscan_subcluster_labels)

# 초기 샘플 선택: KMeans와 HDBSCAN이 모두 동일한 클러스터로 판단한 샘플
def select_agreed_samples(kmeans_labels, hdbscan_subcluster_labels):
    """
    KMeans와 HDBSCAN이 모두 동일한 클러스터(0 또는 1)로 판단한 샘플을 선택합니다.
    
    Parameters:
    - kmeans_labels: KMeans 클러스터 라벨 (0 또는 1)
    - hdbscan_subcluster_labels: HDBSCAN 서브클러스터 라벨 (0, 1 또는 -1)
    
    Returns:
    - indices: 초기 샘플로 선택된 데이터의 인덱스
    """
    # HDBSCAN이 노이즈로 판단하지 않은 샘플 중에서 KMeans와 HDBSCAN 라벨이 동일한 샘플 선택
    mask_agreed = (hdbscan_subcluster_labels != -1) & (kmeans_labels == hdbscan_subcluster_labels)
    indices = np.where(mask_agreed)[0]
    return indices

init_labeled = select_agreed_samples(kmeans_labels, hdbscan_subcluster_labels)

X_cert = X_raw[init_labeled]
y_cert = y[init_labeled]

print(f"[INFO] Total data size: {len(X_raw)}")
print(f"[INFO] Number of agreed samples (KMeans and HDBSCAN agree): {len(X_cert)}")

# AlExperiment는 X와 y가 모두 필요
# 여기서는 "확실한 샘플"로 구성된 (X_cert, y_cert)를 사용
al = AlExperiment(
    X=X_cert,
    y=y_cert,
    stopping_criteria='num_of_queries',
    stopping_value=10,  # 최대 10번 쿼리
    random_state=42
)

# 기본 비율(test_ratio=0.3), 초기 라벨 비율(initial_label_rate=0.1) 등 필요에 따라 수정 가능
al.split_AL(test_ratio=0.3, initial_label_rate=0.1)

# 불확실성 기반 쿼리 전략 설정
# query_strategy="QueryInstanceUncertainty"
# measure='least_confident', 'margin', 'entropy' 등 가능
al.set_query_strategy(
    strategy="QueryInstanceUncertainty",
    measure='least_confident'
)

al.set_performance_metric('accuracy_score')

# 능동 학습 구현
al.start_query(multi_thread=False)

# 학습 곡선 시각화
al.plot_learning_curve()

joblib.dump(scaler, 'al_scaler.joblib')
joblib.dump(al, 'model_active.joblib')

# 클러스터링 결과 시각화 
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.5, label='Clustered Data')
plt.scatter(X_scaled[init_labeled, 0], X_scaled[init_labeled, 1], c='red', edgecolor='k', label='Initial Samples')
plt.title('KMeans and HDBSCAN Clustering with Initial Samples')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

################
# [INFO] Total data size: 170756
# [INFO] Number of agreed samples (KMeans and HDBSCAN agree): 168402

# | round | initially labeled data | number of queries | cost | Performance: |
# |   0   |  11788 (10.00% of all) |         10        |  0   | 1.000 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   1   |  11788 (10.00% of all) |         10        |  0   | 1.000 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   2   |  11788 (10.00% of all) |         10        |  0   | 1.000 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   3   |  11788 (10.00% of all) |         10        |  0   | 1.000 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   4   |  11788 (10.00% of all) |         10        |  0   | 1.000 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   5   |  11788 (10.00% of all) |         10        |  0   | 1.000 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   6   |  11788 (10.00% of all) |         10        |  0   | 1.000 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   7   |  11788 (10.00% of all) |         10        |  0   | 1.000 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   8   |  11788 (10.00% of all) |         10        |  0   | 0.999 ± 0.00 |
# | round | initially labeled data | number of queries | cost | Performance: |
# |   9   |  11788 (10.00% of all) |         10        |  0   | 1.000 ± 0.00 |
# +--------------------------+-------------------+---------------------------+--------------+------------+
# |         Methods          | number_of_queries | number_of_different_split | performance  | batch_size |
# +--------------------------+-------------------+---------------------------+--------------+------------+
# | QueryInstanceUncertainty |         10        |             10            | 1.000 ± 0.00 |     1      |
# +--------------------------+-------------------+---------------------------+--------------+------------+
