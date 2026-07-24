from typing import Any, Optional


class FaceBoundaryScorer:
    def __init__(self, window_size: int = 3):
        self.window_size = int(window_size)

    @staticmethod
    def jaccard(a, b) -> Optional[float]:
        a = set(a)
        b = set(b)
        if len(a) == 0 and len(b) == 0:
            return None
        if len(a) == 0 or len(b) == 0:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def collect_face_ids(shots: list[dict[str, Any]], start: int, end: int) -> set:
        ids = set()
        for idx in range(start, end):
            ids.update(shots[idx].get("face_ids", []))
        return ids

    def compute(self, shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        n = len(shots)

        for idx in range(n - 1):
            left_start = max(0, idx - self.window_size + 1)
            left_end = idx + 1
            right_start = idx + 1
            right_end = min(n, idx + 1 + self.window_size)

            left_faces = self.collect_face_ids(shots, left_start, left_end)
            right_faces = self.collect_face_ids(shots, right_start, right_end)
            overlap = self.jaccard(left_faces, right_faces)
            valid = overlap is not None
            face_change = 0.0 if overlap is None else 1.0 - overlap

            results.append(
                {
                    "boundary_index": idx,
                    "left_shot_id": shots[idx]["shot_id"],
                    "right_shot_id": shots[idx + 1]["shot_id"],
                    "left_context_range": [left_start, left_end - 1],
                    "right_context_range": [right_start, right_end - 1],
                    "left_face_ids": sorted(int(x) for x in left_faces),
                    "right_face_ids": sorted(int(x) for x in right_faces),
                    "face_overlap": None if overlap is None else float(overlap),
                    "face_change": float(face_change),
                    "valid": bool(valid),
                }
            )

        return results
