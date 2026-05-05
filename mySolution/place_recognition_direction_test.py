import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _prepare_descriptors(ref_features: np.ndarray, query_features: np.ndarray):
    scaler = StandardScaler()
    all_features = np.vstack([ref_features, query_features])
    all_scaled = scaler.fit_transform(all_features)

    n_components = min(80, all_scaled.shape[1], max(2, all_scaled.shape[0] - 1))
    pca = PCA(n_components=n_components, whiten=True, random_state=41)
    all_desc = pca.fit_transform(all_scaled).astype(np.float32)
    all_desc /= np.linalg.norm(all_desc, axis=1, keepdims=True) + 1e-6
    return all_desc[: len(ref_features)], all_desc[len(ref_features) :]


def _dtw_match(ref_desc: np.ndarray, query_desc: np.ndarray, diagonal_bias: float = 0.04):
    cost = cdist(query_desc, ref_desc, metric="sqeuclidean").astype(np.float32)
    qn, rn = cost.shape
    dp = np.full((qn, rn), np.inf, dtype=np.float32)
    back = np.zeros((qn, rn), dtype=np.uint8)
    dp[0, 0] = cost[0, 0]

    for i in range(qn):
        for j in range(rn):
            if i == 0 and j == 0:
                continue
            candidates = []
            if i > 0 and j > 0:
                candidates.append((dp[i - 1, j - 1] - diagonal_bias, 1))
            if i > 0:
                candidates.append((dp[i - 1, j], 2))
            if j > 0:
                candidates.append((dp[i, j - 1], 3))
            best, move = min(candidates, key=lambda item: item[0])
            dp[i, j] = cost[i, j] + best
            back[i, j] = move

    matches = [[] for _ in range(qn)]
    i, j = qn - 1, rn - 1
    while True:
        matches[i].append(j)
        if i == 0 and j == 0:
            break
        move = back[i, j]
        if move == 1:
            i -= 1
            j -= 1
        elif move == 2:
            i -= 1
        else:
            j -= 1

    pred_ref_index = np.zeros(qn, dtype=np.float32)
    last = 0.0
    for idx, refs in enumerate(matches):
        if refs:
            last = float(np.mean(refs))
        pred_ref_index[idx] = last
    return pred_ref_index, float(dp[-1, -1] / max(1, qn + rn))


def _sequence_place_predict(ref_features, ref_labels, query_features):
    ref_desc, query_desc = _prepare_descriptors(ref_features, query_features)
    matched_ref_index, mean_cost = _dtw_match(ref_desc, query_desc)
    ref_x = np.arange(len(ref_labels), dtype=np.float32)
    prediction = np.interp(matched_ref_index, ref_x, ref_labels).astype(np.float32)
    return prediction, matched_ref_index, mean_cost
