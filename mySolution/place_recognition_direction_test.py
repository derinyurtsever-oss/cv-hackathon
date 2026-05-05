import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from localization_model import aligned_labels, extract_video_features, project_root, results_dir


def _prepare_descriptors(ref_features: np.ndarray, query_features: np.ndarray):
    scaler = StandardScaler()
    all_features = np.vstack([ref_features, query_features])
    all_scaled = scaler.fit_transform(all_features)

    n_components = min(80, all_scaled.shape[1], max(2, all_scaled.shape[0] - 1))
    pca = PCA(n_components=n_components, whiten=True, random_state=41)
    all_desc = pca.fit_transform(all_scaled).astype(np.float32)

    # Unit-normalized PCA descriptors make cosine/euclidean sequence matching
    # less sensitive to lens dirt and global contrast shifts.
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


def _metrics(name: str, prediction: np.ndarray, labels: np.ndarray, channel_length: float):
    err = np.abs(prediction - labels)
    mae = float(err.mean())
    med = float(np.median(err))
    p90 = float(np.percentile(err, 90))
    print(
        f"{name}: MAE={mae:.3f} m, norm={mae / channel_length:.4f}, "
        f"median={med:.3f} m, p90={p90:.3f} m"
    )
    return mae, mae / channel_length, med, p90


def _triangle(n: int, channel_length: float):
    half = max(1, n // 2)
    return np.concatenate(
        [
            np.linspace(0.0, channel_length, half, dtype=np.float32),
            np.linspace(channel_length, 0.0, n - half, dtype=np.float32),
        ]
    )


def run(video: int):
    channel_lengths = np.load(project_root() / "channel_lengths.npy")
    channel_length = float(channel_lengths[video - 1])
    frame_numbers, features = extract_video_features(video)
    labels = aligned_labels(video, frame_numbers)
    if labels is None:
        raise FileNotFoundError(f"No distance label available for video {video}")

    turn = int(np.argmax(labels))
    inward = np.arange(0, turn + 1, dtype=np.int32)
    ret = np.arange(turn, len(labels), dtype=np.int32)

    # Return side is physically traversed from max distance back to zero.  For
    # same-place matching, reverse it so both sequences run from zero to max.
    ret_rev = ret[::-1]
    pred_return_rev, match_a, cost_a = _sequence_place_predict(
        features[inward], labels[inward], features[ret_rev]
    )
    pred_return = pred_return_rev[::-1]

    pred_inward, match_b, cost_b = _sequence_place_predict(
        features[ret_rev], labels[ret_rev], features[inward]
    )

    baseline = _triangle(len(labels), channel_length)
    print(f"Video {video}: frames={len(labels)}, true turn={turn}")
    print(f"Inward reference -> return query DTW cost: {cost_a:.4f}")
    a_metrics = _metrics("place inward->return", pred_return, labels[ret], channel_length)
    a_base = _metrics("triangle on return side", baseline[ret], labels[ret], channel_length)
    print(f"Return reference -> inward query DTW cost: {cost_b:.4f}")
    b_metrics = _metrics("place return->inward", pred_inward, labels[inward], channel_length)
    b_base = _metrics("triangle on inward side", baseline[inward], labels[inward], channel_length)

    pred_a_full = np.full_like(labels, np.nan, dtype=np.float32)
    pred_b_full = np.full_like(labels, np.nan, dtype=np.float32)
    pred_a_full[ret] = pred_return
    pred_b_full[inward] = pred_inward

    out_dir = results_dir()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(labels, label="label", linewidth=1.5)
    axes[0].plot(pred_a_full, label="place inward -> return", linewidth=1.2)
    axes[0].plot(baseline, label="triangle", linewidth=1.0, alpha=0.7)
    axes[0].axvline(turn, color="black", linewidth=0.8, alpha=0.55)
    axes[0].set_title(f"Video {video}: place recognition, inward to return")
    axes[0].set_ylabel("distance [m]")
    axes[0].legend()

    axes[1].plot(labels, label="label", linewidth=1.5)
    axes[1].plot(pred_b_full, label="place return -> inward", linewidth=1.2)
    axes[1].plot(baseline, label="triangle", linewidth=1.0, alpha=0.7)
    axes[1].axvline(turn, color="black", linewidth=0.8, alpha=0.55)
    axes[1].set_title(f"Video {video}: place recognition, return to inward")
    axes[1].set_xlabel("trimmed frame index")
    axes[1].set_ylabel("distance [m]")
    axes[1].legend()
    fig.tight_layout()

    plot_path = out_dir / f"place_recognition_direction_video_{video}.png"
    csv_path = out_dir / f"place_recognition_direction_video_{video}.csv"
    summary_path = out_dir / f"place_recognition_direction_video_{video}_summary.txt"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("frame_number,label_m,triangle_m,inward_to_return_m,return_to_inward_m\n")
        for frame, label, tri, pred_a, pred_b in zip(
            frame_numbers, labels, baseline, pred_a_full, pred_b_full
        ):
            f.write(f"{frame},{label:.6f},{tri:.6f},{pred_a:.6f},{pred_b:.6f}\n")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"video={video}\nturn={turn}\n")
        f.write(f"inward_to_return_dtw_cost={cost_a:.6f}\n")
        f.write(f"inward_to_return_mae={a_metrics[0]:.6f}\n")
        f.write(f"triangle_return_mae={a_base[0]:.6f}\n")
        f.write(f"return_to_inward_dtw_cost={cost_b:.6f}\n")
        f.write(f"return_to_inward_mae={b_metrics[0]:.6f}\n")
        f.write(f"triangle_inward_mae={b_base[0]:.6f}\n")

    print(f"Saved plot: {plot_path}")
    print(f"Saved csv: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=int, default=11)
    args = parser.parse_args()
    run(args.video)
