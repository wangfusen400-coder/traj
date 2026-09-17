import json
import os

import cv2
import numpy as np


def _as_points(track):
    points = np.asarray(track, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("data error !")
    return points


def _direction_bgr(dx: float, dy: float):
    angle = np.arctan2(dy, dx)
    hue = int(((angle + np.pi) / (2.0 * np.pi) * 180.0) % 180.0)
    hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _resample_by_arclength(track, num_points=40):
    """Resample a trajectory uniformly by arc length without changing direction."""
    points = _as_points(track)
    if len(points) < 2:
        return points

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, segment_lengths > 1e-6]]
    if len(points) < 2:
        return points

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    targets = np.linspace(0.0, cumulative[-1], num_points, dtype=np.float32)
    return np.column_stack([
        np.interp(targets, cumulative, points[:, 0]),
        np.interp(targets, cumulative, points[:, 1]),
    ]).astype(np.float32)


def _same_directed_path(a, b, distance_threshold_px, direction_threshold_deg):
    """Match spatially close paths only when their local travel directions agree."""
    if len(a) < 2 or len(b) < 2 or a.shape != b.shape:
        return False

    point_distances = np.linalg.norm(a - b, axis=1)
    if float(np.mean(point_distances)) > distance_threshold_px:
        return False
    if float(np.percentile(point_distances, 90)) > distance_threshold_px * 1.8:
        return False

    tangent_a = np.diff(a, axis=0)
    tangent_b = np.diff(b, axis=0)
    norm_a = np.linalg.norm(tangent_a, axis=1)
    norm_b = np.linalg.norm(tangent_b, axis=1)
    valid = (norm_a > 1e-6) & (norm_b > 1e-6)
    if not np.any(valid):
        return False

    cosine = np.sum(tangent_a[valid] * tangent_b[valid], axis=1) / (
        norm_a[valid] * norm_b[valid]
    )
    minimum_cosine = np.cos(np.deg2rad(direction_threshold_deg))
    return float(np.mean(cosine >= minimum_cosine)) >= 0.85


def thin_similar_trajectories(
    trajs,
    distance_threshold_px=10.0,
    direction_threshold_deg=30.0,
    resample_points=40,
):
    """
    Thin spatially similar trajectories while retaining different directions.

    Returns:
        representative trajectories,
        their original indices,
        original indices represented by every retained trajectory.
    """
    valid_indices = []
    resampled = []
    for index, track in enumerate(trajs):
        sampled = _resample_by_arclength(track, resample_points)
        if len(sampled) >= 2:
            valid_indices.append(index)
            resampled.append(sampled)

    parent = list(range(len(resampled)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(resampled)):
        for j in range(i + 1, len(resampled)):
            if _same_directed_path(
                resampled[i],
                resampled[j],
                distance_threshold_px,
                direction_threshold_deg,
            ):
                union(i, j)

    groups = {}
    for local_index, original_index in enumerate(valid_indices):
        groups.setdefault(find(local_index), []).append((local_index, original_index))

    representatives = []
    grouped_indices = []
    for members in groups.values():
        local_indices = [item[0] for item in members]
        original_indices = [item[1] for item in members]
        stack = np.stack([resampled[index] for index in local_indices])
        mean_path = np.mean(stack, axis=0)
        costs = np.mean(np.linalg.norm(stack - mean_path, axis=2), axis=1)
        representatives.append(original_indices[int(np.argmin(costs))])
        grouped_indices.append(original_indices)

    order = np.argsort(representatives)
    representatives = [representatives[index] for index in order]
    grouped_indices = [grouped_indices[index] for index in order]
    return [trajs[index] for index in representatives], representatives, grouped_indices


def draw_directional_trajectories(
    image,
    trajs,
    thickness=4,
    arrow_spacing_px=55,
    arrow_tip_length=0.35,
):
    output = image.copy()
    for track in trajs:
        points = _as_points(track)
        if len(points) < 2:
            continue

        distance_since_arrow = 0.0
        arrow_start = tuple(np.rint(points[0]).astype(int))
        for p0f, p1f in zip(points[:-1], points[1:]):
            delta = p1f - p0f
            segment_length = float(np.linalg.norm(delta))
            if segment_length < 1e-6:
                continue

            p0 = tuple(np.rint(p0f).astype(int))
            p1 = tuple(np.rint(p1f).astype(int))
            color = _direction_bgr(float(delta[0]), float(delta[1]))
            cv2.line(output, p0, p1, color, thickness, cv2.LINE_AA)

            distance_since_arrow += segment_length
            if distance_since_arrow >= arrow_spacing_px:
                cv2.arrowedLine(
                    output,
                    arrow_start,
                    p1,
                    color,
                    max(thickness, 2),
                    cv2.LINE_AA,
                    tipLength=arrow_tip_length,
                )
                distance_since_arrow = 0.0
                arrow_start = p1
    return output


def demo(
    traj_path,
    output_dir,
    distance_threshold_px=10.0,
    direction_threshold_deg=30.0,
):
    with open(traj_path, encoding="utf-8") as file:
        traj_data = json.load(file)["trajectories"]
    trajs = [item["points"] for item in traj_data]

    thinned_trajs, kept_indices, groups = thin_similar_trajectories(
        trajs,
        distance_threshold_px=distance_threshold_px,
        direction_threshold_deg=direction_threshold_deg,
        resample_points=40,
    )

    canvas = np.full((512, 512, 3), 28, dtype=np.uint8)
    rendered = draw_directional_trajectories(
        canvas,
        thinned_trajs,
        thickness=4,
        arrow_spacing_px=55,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "directionl_trajectory_arrow.png")
    cv2.imwrite(output_path, rendered)

    manifest = {
        "input_count": len(trajs),
        "retained_count": len(thinned_trajs),
        "distance_threshold_px": distance_threshold_px,
        "direction_threshold_deg": direction_threshold_deg,
        "groups": [
            {
                "representative_index": kept,
                "source_indices": members,
            }
            for kept, members in zip(kept_indices, groups)
        ],
    }
    manifest_path = os.path.join(output_dir, "trajectory_thinning_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"trajectories: {len(trajs)} -> {len(thinned_trajs)}")
    print(f"image: {output_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    traj_path = r"D:\wsl\bev_trajectories\bev_trajectories_100.json"
    output_dir = r"D:\wsl"
    demo(traj_path, output_dir)
