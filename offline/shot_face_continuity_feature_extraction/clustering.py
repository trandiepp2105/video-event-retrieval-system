from collections import Counter, defaultdict
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .extractor import l2_normalize


class FaceClusterer:
    def __init__(self, same_face_threshold: float = 0.45):
        self.same_face_threshold = float(same_face_threshold)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def cluster_online(
        self,
        detections: list[dict[str, Any]],
        show_progress: bool = True,
    ) -> tuple[list[dict[str, Any]], list[list[int]], list[np.ndarray]]:
        clusters: list[list[int]] = []
        centroids: list[np.ndarray] = []

        iterator = enumerate(detections)
        if show_progress:
            iterator = tqdm(iterator, total=len(detections), desc="Cluster faces")

        for det_idx, det in iterator:
            emb = det["embedding"]

            if len(centroids) == 0:
                det["face_id"] = 0
                clusters.append([det_idx])
                centroids.append(emb.copy())
                continue

            sims = np.asarray([self.cosine(emb, centroid) for centroid in centroids], dtype=np.float32)
            best_id = int(np.argmax(sims))
            best_sim = float(sims[best_id])

            if best_sim >= self.same_face_threshold:
                det["face_id"] = best_id
                clusters[best_id].append(det_idx)
                cluster_embs = np.stack([detections[i]["embedding"] for i in clusters[best_id]])
                centroids[best_id] = l2_normalize(cluster_embs.mean(axis=0))
            else:
                new_id = len(clusters)
                det["face_id"] = new_id
                clusters.append([det_idx])
                centroids.append(emb.copy())

        return detections, clusters, centroids

    @staticmethod
    def attach_to_shots(shots: list[dict[str, Any]], detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        shot_to_dets = defaultdict(list)
        for det in detections:
            shot_to_dets[int(det["shot_idx"])].append(det)

        for shot_idx, shot in enumerate(shots):
            shot["face_detections"] = shot_to_dets.get(shot_idx, [])
        return shots

    @staticmethod
    def assign_face_ids_to_shots(shots: list[dict[str, Any]], min_count: int = 1) -> list[dict[str, Any]]:
        for shot in shots:
            ids = [int(det["face_id"]) for det in shot.get("face_detections", []) if "face_id" in det]
            counter = Counter(ids)
            stable_ids = sorted([int(face_id) for face_id, count in counter.items() if count >= min_count])
            shot["face_ids"] = stable_ids
            shot["face_id_counts"] = {str(k): int(v) for k, v in sorted(counter.items())}
        return shots
