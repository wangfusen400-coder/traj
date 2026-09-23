import os

import cv2
import numpy as np


CANVAS_SIZE = 512
PADDING = 18


def _as_points(track):
    points = np.asarray(track, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Each trajectory must have shape [num_points, 2].")
    return points


def load_npy_trajectories(path):
    data = np.load(path, allow_pickle=True)
    if data.ndim == 3 and data.shape[2] == 2:
        raw_tracks = data
    elif data.ndim == 1:
        raw_tracks = data
    else:
        raise ValueError(
            "Expected a fixed array [N, T, 2] or a ragged object array [N], "
            f"got shape={data.shape}, dtype={data.dtype}."
        )

    tracks = []
    source_indices = []
    dropped_stationary_count = 0
    for source_index, track in enumerate(raw_tracks):
        points = np.asarray(track, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"Trajectory {source_index} must have shape [Li, 2], got {points.shape}."
            )
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 2:
            dropped_stationary_count += 1
            continue
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        points = points[np.r_[True, segment_lengths > 1e-6]]
        if len(points) < 2:
            dropped_stationary_count += 1
            continue
        tracks.append(points)
        source_indices.append(source_index)
    if not tracks:
        raise ValueError("No moving trajectory with at least two distinct points was found.")

    all_points = np.concatenate(tracks, axis=0)
    source_min = np.min(all_points, axis=0)
    source_max = np.max(all_points, axis=0)
    if np.any(source_min < 0):
        raise ValueError(
            f"Negative coordinates cannot be drawn without translation: min={source_min.tolist()}."
        )

    image_width = int(np.ceil(source_max[0])) + 1
    image_height = int(np.ceil(source_max[1])) + 1
    metadata = {
        "source_min": source_min.tolist(),
        "source_max": source_max.tolist(),
        "scale": 1.0,
        "offset": [0.0, 0.0],
        "image_width": image_width,
        "image_height": image_height,
        "source_track_count": int(len(data)),
        "moving_track_count": int(len(tracks)),
        "dropped_stationary_count": int(dropped_stationary_count),
    }
    return tracks, metadata, source_indices


def _direction_bgr(dx: float, dy: float):
    angle = np.arctan2(dy, dx)
    hue = int(((angle + np.pi) / (2.0 * np.pi) * 180.0) % 180.0)
    hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_directional_trajectories(
    image,
    trajs,
    thickness=4,
    arrow_spacing_px=45,
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


def _resample_by_arclength(track, num_points=40):
    points = _as_points(track)
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


def _directed_path_metrics(a, b, direction_threshold_deg):
    point_distances = np.linalg.norm(a - b, axis=1)
    tangent_a = np.diff(a, axis=0)
    tangent_b = np.diff(b, axis=0)
    norm_a = np.linalg.norm(tangent_a, axis=1)
    norm_b = np.linalg.norm(tangent_b, axis=1)
    valid = (norm_a > 1e-6) & (norm_b > 1e-6)

    if np.any(valid):
        cosine = np.sum(tangent_a[valid] * tangent_b[valid], axis=1) / (
            norm_a[valid] * norm_b[valid]
        )
        minimum_cosine = np.cos(np.deg2rad(direction_threshold_deg))
        direction_agreement = float(np.mean(cosine >= minimum_cosine))
        mean_cosine = float(np.mean(cosine))
    else:
        direction_agreement = 0.0
        mean_cosine = -1.0

    return {
        "mean_distance_px": float(np.mean(point_distances)),
        "p90_distance_px": float(np.percentile(point_distances, 90)),
        "direction_agreement": direction_agreement,
        "mean_direction_cosine": mean_cosine,
    }


def _same_directed_path(
    a,
    b,
    distance_threshold_px,
    direction_threshold_deg,
):
    if len(a) < 2 or len(b) < 2 or a.shape != b.shape:
        return False
    metrics = _directed_path_metrics(a, b, direction_threshold_deg)
    return (
        metrics["mean_distance_px"] <= distance_threshold_px
        and metrics["p90_distance_px"] <= distance_threshold_px * 1.8
        and metrics["direction_agreement"] >= 0.85
    )


def thin_similar_trajectories(
    trajs,
    distance_threshold_px=200.0,
    direction_threshold_deg=30.0,
    resample_points=40,
):
    """
    Vectorized equivalent of the original greedy grouping algorithm.

    The representative order and first-match rule are unchanged. Only the
    comparisons against the current representatives are evaluated as a batch.
    """
    sampled = [_resample_by_arclength(track, resample_points) for track in trajs]
    points = np.stack(sampled, axis=0)

    tangents = np.diff(points, axis=1)
    tangent_norms = np.linalg.norm(tangents, axis=2)
    valid_tangents = tangent_norms > 1e-6
    unit_tangents = np.divide(
        tangents,
        tangent_norms[:, :, None],
        out=np.zeros_like(tangents),
        where=valid_tangents[:, :, None],
    )
    minimum_cosine = np.cos(np.deg2rad(direction_threshold_deg))
    maximum_p90_distance = distance_threshold_px * 1.8

    groups = []
    representatives = []

    for index in range(len(points)):
        if not representatives:
            representatives.append(index)
            groups.append([index])
            continue

        representative_array = np.asarray(representatives, dtype=np.intp)

        # Spatial metrics for this candidate against every current representative.
        distances = np.linalg.norm(
            points[representative_array] - points[index],
            axis=2,
        )
        mean_distances = np.mean(distances, axis=1)
        p90_distances = np.percentile(distances, 90, axis=1)
        spatial_match = (
            (mean_distances <= distance_threshold_px)
            & (p90_distances <= maximum_p90_distance)
        )

        # Direction metrics reuse precomputed normalized local tangents.
        pair_valid = (
            valid_tangents[representative_array]
            & valid_tangents[index][None, :]
        )
        cosine = np.einsum(
            "rkd,kd->rk",
            unit_tangents[representative_array],
            unit_tangents[index],
        )
        valid_counts = np.sum(pair_valid, axis=1)
        direction_agreements = np.divide(
            np.sum((cosine >= minimum_cosine) & pair_valid, axis=1),
            valid_counts,
            out=np.zeros(len(representatives), dtype=np.float64),
            where=valid_counts > 0,
        )

        matching_groups = np.flatnonzero(
            spatial_match & (direction_agreements >= 0.85)
        )
        if len(matching_groups):
            # Preserve the original greedy rule: use the first matching group.
            groups[int(matching_groups[0])].append(index)
        else:
            representatives.append(index)
            groups.append([index])

    return [trajs[index] for index in representatives], representatives, groups, sampled


def validate_thinning(
    sampled,
    representative_indices,
    groups,
    distance_threshold_px,
    direction_threshold_deg,
):
    comparisons = []
    violations = []
    for representative_index, members in zip(representative_indices, groups):
        representative = sampled[representative_index]
        for member_index in members:
            metrics = _directed_path_metrics(
                representative,
                sampled[member_index],
                direction_threshold_deg,
            )
            comparisons.append(metrics)
            if not (
                metrics["mean_distance_px"] <= distance_threshold_px
                and metrics["p90_distance_px"] <= distance_threshold_px * 1.8
                and metrics["direction_agreement"] >= 0.85
            ):
                violations.append({
                    "representative_index": representative_index,
                    "member_index": member_index,
                    **metrics,
                })

    reverse_path_test = np.array(
        [[40.0, 100.0], [256.0, 100.0], [472.0, 100.0]],
        dtype=np.float32,
    )
    near_path_test = reverse_path_test + np.array([0.0, 2.0], dtype=np.float32)
    synthetic = [
        _resample_by_arclength(reverse_path_test, 40),
        _resample_by_arclength(near_path_test, 40),
        _resample_by_arclength(reverse_path_test[::-1], 40),
    ]
    same_direction_merged = _same_directed_path(
        synthetic[0],
        synthetic[1],
        distance_threshold_px,
        direction_threshold_deg,
    )
    reverse_direction_merged = _same_directed_path(
        synthetic[0],
        synthetic[2],
        distance_threshold_px,
        direction_threshold_deg,
    )

    if comparisons:
        worst_mean_distance = max(item["mean_distance_px"] for item in comparisons)
        worst_p90_distance = max(item["p90_distance_px"] for item in comparisons)
        minimum_direction_agreement = min(item["direction_agreement"] for item in comparisons)
        minimum_mean_cosine = min(item["mean_direction_cosine"] for item in comparisons)
    else:
        worst_mean_distance = 0.0
        worst_p90_distance = 0.0
        minimum_direction_agreement = 1.0
        minimum_mean_cosine = 1.0

    return {
        "passed": (
            not violations
            and same_direction_merged
            and not reverse_direction_merged
        ),
        "group_member_comparisons": len(comparisons),
        "threshold_violations": violations,
        "worst_mean_distance_px": worst_mean_distance,
        "worst_p90_distance_px": worst_p90_distance,
        "minimum_direction_agreement": minimum_direction_agreement,
        "minimum_mean_direction_cosine": minimum_mean_cosine,
        "synthetic_same_direction_merged": bool(same_direction_merged),
        "synthetic_reverse_direction_merged": bool(reverse_direction_merged),
    }


def demo(
    traj_path,
    output_dir,
    distance_threshold_px=200.0,
    direction_threshold_deg=30.0,
):
    trajs, transform, source_indices = load_npy_trajectories(traj_path)
    thinned, kept_indices, groups, sampled = thin_similar_trajectories(
        trajs,
        distance_threshold_px=distance_threshold_px,
        direction_threshold_deg=direction_threshold_deg,
        resample_points=40,
    )
    validation = validate_thinning(
        sampled,
        kept_indices,
        groups,
        distance_threshold_px,
        direction_threshold_deg,
    )

    canvas = np.full((transform["image_height"], transform["image_width"], 3), 28, dtype=np.uint8)
    rendered = draw_directional_trajectories(
        canvas,
        thinned,
        thickness=4,
        arrow_spacing_px=55,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "directionl_trajectory_arrow_speed.png")
    cv2.imwrite(output_path, rendered)

    manifest = {
        "source": os.path.abspath(traj_path),
        "source_count": transform["source_track_count"],
        "moving_count": len(trajs),
        "retained_count": len(thinned),
        "distance_threshold_px": distance_threshold_px,
        "direction_threshold_deg": direction_threshold_deg,
        "resample_points": 40,
        "coordinate_transform": transform,
        "validation": validation,
        "groups": [
            {
                "representative_index": source_indices[representative],
                "source_indices": [source_indices[index] for index in members],
            }
            for representative, members in zip(kept_indices, groups)
        ],
    }
    manifest_path = os.path.join(output_dir, "trajectory_thinning_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        import json
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"trajectories: {transform['source_track_count']} raw, {len(trajs)} moving -> {len(thinned)} retained")
    print(f"validation passed: {validation['passed']}")
    print(f"threshold violations: {len(validation['threshold_violations'])}")
    print(f"image: {output_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    from time import time
    start_time = time()
    traj_path = r"D:\wsl\traj\merged_trajectories.npy"
    output_dir = r"D:\wsl\pose"
    demo(traj_path, output_dir)
    print(f"consume time: {time() - start_time}")
