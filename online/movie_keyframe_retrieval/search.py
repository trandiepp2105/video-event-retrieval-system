from __future__ import annotations

import concurrent.futures
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .encoders import OpenClipTextEncoder
from .faiss_index import FaissIndex
from .io_utils import load_pickle, utc_timestamp
from .meili_search import MeiliSearchService
from .metadata import MetadataStore
from .registry import IndexRegistry
from .reranker import Reranker
from .schemas import FrameRangeMetadata, SearchResult, StageQuery

QUERY_CHANNELS = ["text", "ocr", "subtitle", "image"]


@dataclass
class StageCandidate:
    video_id: str
    frame_idx: int
    fused_score: float
    visual_score: float = 0.0
    ocr_score: float = 0.0
    subtitle_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FAISSSearchEngine:
    def __init__(
        self,
        list_faiss_configs: list[dict[str, Any]],
        reranker: Optional[Reranker] = None,
    ) -> None:
        if not isinstance(list_faiss_configs, list) or not list_faiss_configs:
            raise ValueError("list_faiss_configs phải là list không rỗng.")

        self.configs: dict[str, dict[str, Any]] = {}
        self.indexes: dict[str, FaissIndex] = {}
        self.embedders: dict[str, Any] = {}
        self.index_to_meta: dict[str, list[tuple[str, int]]] = {}
        self.meta_to_index: dict[str, dict[tuple[str, int], int]] = {}
        self.total_vectors: dict[str, int] = {}
        self.embedding_dims: dict[str, int] = {}
        self.gpu_resources_map: dict[int, Any] = {}

        for cfg in list_faiss_configs:
            model_name = str(cfg["model_name"])
            self.configs[model_name] = dict(cfg)
            embedder = cfg.get("embedder")
            if embedder is not None:
                self.embedders[model_name] = embedder
            existing_index = cfg.get("index")
            if existing_index is not None:
                self._register_loaded_index(model_name, existing_index)

        self.reranker = reranker if reranker else Reranker()

    def _register_loaded_index(self, model_name: str, index: FaissIndex) -> None:
        self.indexes[model_name] = index
        metas = index.metadata_store.get_many(index.ids.tolist())
        meta_list = [(meta.video_id, int(meta.frame_start)) for meta in metas]
        self.index_to_meta[model_name] = meta_list
        self.meta_to_index[model_name] = {meta: row for row, meta in enumerate(meta_list)}
        self.total_vectors[model_name] = int(index.ids.shape[0])
        self.embedding_dims[model_name] = int(index.vectors.shape[1]) if index.vectors.ndim == 2 else 0

    def _get_gpu_resource(self, gpu_id: int):
        if gpu_id not in self.gpu_resources_map:
            try:
                import faiss  # type: ignore

                self.gpu_resources_map[gpu_id] = faiss.StandardGpuResources()
            except Exception:
                self.gpu_resources_map[gpu_id] = None
        return self.gpu_resources_map[gpu_id]

    @staticmethod
    def _iter_embedding_records(embedding_path: Path):
        if embedding_path.is_file():
            payload = load_pickle(embedding_path)
            if isinstance(payload, list):
                for item in payload:
                    yield item
                return
            if isinstance(payload, dict):
                yield payload
                return
            raise ValueError(f"Embedding file không hợp lệ: {embedding_path}")

        if embedding_path.is_dir():
            for pkl_path in sorted(embedding_path.glob("*.pkl")):
                payload = load_pickle(pkl_path)
                if not isinstance(payload, dict):
                    raise ValueError(f"Mỗi file pkl phải là dict: {pkl_path}")
                if "video_name" not in payload:
                    payload = {**payload, "video_name": pkl_path.stem}
                yield payload
            return

        raise FileNotFoundError(f"Không tìm thấy embedding path: {embedding_path}")

    def _build_single_index(self, model_name: str) -> None:
        if model_name not in self.configs:
            raise KeyError(f"Không tìm thấy cấu hình cho model '{model_name}'.")

        config = self.configs[model_name]
        embedding_path_value = config.get("embedding_path")
        if embedding_path_value is None:
            raise ValueError(f"Cấu hình của '{model_name}' thiếu embedding_path.")
        embedding_path = Path(embedding_path_value)

        vectors: list[np.ndarray] = []
        ids: list[int] = []
        store = MetadataStore()
        index_id = 0

        for video_record in self._iter_embedding_records(embedding_path):
            if not isinstance(video_record, dict):
                raise ValueError("Mỗi phần tử embedding record phải là dict.")

            video_name = str(video_record.get("video_name", ""))
            frame_indices = video_record.get("keyframe_frame_indices")
            embeddings = video_record.get("keyframe_embeddings")

            if not video_name:
                raise ValueError(f"Thiếu video_name trong embedding record của model '{model_name}'.")
            if frame_indices is None or embeddings is None:
                raise ValueError(
                    f"Thiếu keyframe_frame_indices/keyframe_embeddings trong dữ liệu của model '{model_name}'."
                )

            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            if len(frame_indices) != int(embeddings.shape[0]):
                raise ValueError(
                    f"Frame/embedding mismatch cho video '{video_name}' của model '{model_name}': "
                    f"{len(frame_indices)} vs {embeddings.shape[0]}"
                )

            for row, frame_idx in enumerate(frame_indices):
                frame_idx = int(frame_idx)
                vectors.append(np.asarray(embeddings[row], dtype=np.float32).reshape(-1))
                ids.append(index_id)
                store.add(
                    FrameRangeMetadata(
                        index_id=index_id,
                        video_id=video_name,
                        frame_start=frame_idx,
                        frame_end=frame_idx,
                        item_id=frame_idx,
                    )
                )
                index_id += 1

        if not vectors:
            raise ValueError(f"Không tìm thấy embedding nào cho model '{model_name}'.")

        matrix = np.stack(vectors).astype(np.float32)
        index = FaissIndex.build(
            vectors=matrix,
            ids=np.asarray(ids, dtype=np.int64),
            metadata_store=store,
            index_name=model_name,
            config={
                "index_name": model_name,
                "metric": "cosine",
                "index_type": config.get("index_type", "IndexFlatIP"),
                "normalized": True,
                "dim": int(matrix.shape[1]),
                "num_vectors": int(matrix.shape[0]),
                "input_dir": str(embedding_path),
                "created_at": utc_timestamp(),
            },
        )
        self._register_loaded_index(model_name, index)

    def build_all_indexes(self) -> None:
        for model_name in self.configs:
            if model_name not in self.indexes:
                self._build_single_index(model_name)

    def save_all_indexes(self) -> None:
        for model_name, index in self.indexes.items():
            output_path = self.configs[model_name].get("output_index_path")
            if not output_path:
                continue
            index.save(Path(output_path))

    def load_all_indexes(self) -> None:
        for model_name, config in self.configs.items():
            input_index_path = config.get("input_index_path")
            if not input_index_path:
                continue
            loaded_index = FaissIndex.load(Path(input_index_path))
            self._register_loaded_index(model_name, loaded_index)

    def _search_single_model(self, model_name: str, queries: list[str], k: int):
        if model_name not in self.indexes:
            return [[] for _ in queries]
        if model_name not in self.embedders:
            raise ValueError(f"Model '{model_name}' chưa có embedder để search.")
        index = self.indexes[model_name]
        embedder = self.embedders[model_name]
        query_array = embedder.encode_batch(queries)
        batch_results = []
        for query_vector in query_array:
            results = index.search(query_vector, top_k=k)
            batch_results.append([((item.video_id, int(item.frame_start)), float(item.score)) for item in results])
        return batch_results

    def search(self, queries, models_to_search, k=100):
        if not queries:
            return []
        per_model_results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._search_single_model, model_cfg["model_name"], queries, k): model_cfg["model_name"]
                for model_cfg in models_to_search
                if model_cfg["model_name"] in self.indexes
            }
            for future in concurrent.futures.as_completed(futures):
                model_name = futures[future]
                per_model_results[model_name] = future.result()
        batch_results_per_model = list(per_model_results.values())
        return self.reranker(batch_results_per_model=batch_results_per_model, top_k=k)


class SearchEngine:
    def __init__(
        self,
        *,
        vector_engine: FAISSSearchEngine,
        ocr_engine: MeiliSearchService,
    ) -> None:
        self.vector_engine = vector_engine
        self.ocr_engine = ocr_engine
        self.video_keyframes = self._build_video_keyframes_cache()

    def _default_channel_weights(self) -> dict[str, float]:
        return {"text": 0.4, "ocr": 0.3, "subtitle": 0.15, "image": 0.15}

    def _sanitize_stage_query(self, stage_query: Optional[dict[str, Any]]) -> dict[str, Any]:
        if not stage_query:
            return {}
        cleaned = {}
        for channel in QUERY_CHANNELS:
            value = stage_query.get(channel)
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if value:
                    cleaned[channel] = value
            else:
                cleaned[channel] = value
        return cleaned

    def _build_stage_weights(
        self,
        stage_query: dict[str, Any],
        weights: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        base = self._default_channel_weights()
        if weights:
            for channel in QUERY_CHANNELS:
                if channel in weights:
                    base[channel] = max(float(weights[channel]), 0.0)
        active = {channel: base[channel] for channel in QUERY_CHANNELS if stage_query.get(channel)}
        normalized = {channel: 0.0 for channel in QUERY_CHANNELS}
        if not active:
            return normalized
        total = sum(active.values())
        if total > 0:
            for channel, value in active.items():
                normalized[channel] = value / total
        else:
            equal_weight = 1.0 / len(active)
            for channel in active:
                normalized[channel] = equal_weight
        return normalized

    def _build_video_keyframes_cache(self) -> dict[str, np.ndarray]:
        by_video = defaultdict(set)
        for meta_list in getattr(self.vector_engine, "index_to_meta", {}).values():
            for video_name, frame_idx in meta_list:
                by_video[video_name].add(int(frame_idx))
        return {
            video_name: np.array(sorted(frame_indices), dtype=np.int32)
            for video_name, frame_indices in by_video.items()
        }

    def _get_keyframes_in_range(self, video_name: str, frame_start: int, frame_end: int) -> np.ndarray:
        keyframes = self.video_keyframes.get(video_name)
        if keyframes is None or keyframes.size == 0:
            return np.array([], dtype=np.int32)
        left = np.searchsorted(keyframes, frame_start, side="left")
        right = np.searchsorted(keyframes, frame_end, side="right")
        return keyframes[left:right]

    def _get_nearest_keyframe(self, video_name: str, target_frame: float) -> Optional[int]:
        keyframes = self.video_keyframes.get(video_name)
        if keyframes is None or keyframes.size == 0:
            return None
        pos = np.searchsorted(keyframes, target_frame, side="left")
        candidates = []
        if pos < keyframes.size:
            candidates.append(int(keyframes[pos]))
        if pos > 0:
            candidates.append(int(keyframes[pos - 1]))
        if not candidates:
            return None
        return min(candidates, key=lambda frame_idx: abs(frame_idx - target_frame))

    def _subtitle_hit_to_keys(self, subtitle_hit: dict[str, Any]) -> list[tuple[str, int]]:
        video_name = subtitle_hit.get("video_name", "")
        frame_start = subtitle_hit.get("frame_start", -1)
        frame_end = subtitle_hit.get("frame_end", -1)
        if not video_name or frame_start in [None, -1] or frame_end in [None, -1]:
            return []
        frame_start = int(frame_start)
        frame_end = int(frame_end)
        if frame_end < frame_start:
            frame_start, frame_end = frame_end, frame_start
        matched_keyframes = self._get_keyframes_in_range(video_name, frame_start, frame_end)
        if matched_keyframes.size > 0:
            return [(video_name, int(frame_idx)) for frame_idx in matched_keyframes.tolist()]
        center_frame = (frame_start + frame_end) / 2.0
        nearest_keyframe = self._get_nearest_keyframe(video_name, center_frame)
        if nearest_keyframe is None:
            return []
        return [(video_name, nearest_keyframe)]

    def _fuse_and_rerank_candidates(
        self,
        raw_text_results: list[tuple[tuple[str, int], float]],
        raw_image_results: list[tuple[tuple[str, int], float]],
        raw_ocr_results: list[dict[str, Any]],
        raw_subtitle_results: list[dict[str, Any]],
        weights: dict[str, float],
    ) -> list[tuple[tuple[str, int], float]]:
        all_scores = defaultdict(lambda: {channel: 0.0 for channel in QUERY_CHANNELS})
        for key, score in raw_text_results:
            all_scores[key]["text"] = score
        for key, score in raw_image_results:
            all_scores[key]["image"] = score
        for ocr_hit in raw_ocr_results:
            video_name = ocr_hit.get("video_name", "")
            frame_index = ocr_hit.get("frame_index", -1)
            ocr_score = ocr_hit.get("_rankingScore", 0.0)
            if video_name and frame_index != -1:
                key = (video_name, int(frame_index))
                all_scores[key]["ocr"] = max(all_scores[key]["ocr"], ocr_score)
        for subtitle_hit in raw_subtitle_results:
            subtitle_score = subtitle_hit.get("_rankingScore", 0.0)
            for key in self._subtitle_hit_to_keys(subtitle_hit):
                all_scores[key]["subtitle"] = max(all_scores[key]["subtitle"], subtitle_score)
        if not all_scores:
            return []
        max_map = {
            channel: max((scores[channel] for scores in all_scores.values() if scores[channel] > 0), default=0.0)
            for channel in QUERY_CHANNELS
        }
        temp_combined_results = []
        for key, scores in all_scores.items():
            normalized_scores = {}
            for channel in QUERY_CHANNELS:
                max_val = max_map[channel]
                raw_score = scores[channel]
                if raw_score == 0:
                    normalized_scores[channel] = 0.0
                elif max_val > 0:
                    normalized_scores[channel] = raw_score / max_val
                else:
                    normalized_scores[channel] = 1.0
            fusion_score = sum(weights.get(channel, 0.0) * normalized_scores[channel] for channel in QUERY_CHANNELS)
            if fusion_score > 0:
                temp_combined_results.append((key, fusion_score))
        if not temp_combined_results:
            return []
        max_fusion_score = max(score for _, score in temp_combined_results)
        final_results = []
        for key, score in temp_combined_results:
            normalized_fusion_score = score / max_fusion_score if max_fusion_score > 0 else (1.0 if score > 0 else 0.0)
            final_results.append((key, normalized_fusion_score))
        final_results.sort(key=lambda item: item[1], reverse=True)
        return final_results

    def _resolve_vector_models_config(self, vector_models_config):
        if vector_models_config is not None:
            return vector_models_config
        model_names = list(self.vector_engine.configs.keys())
        return [{"model_name": model_name} for model_name in model_names]

    def _prepare_hybrid_vector_queries(
        self,
        text_query: Optional[str] = None,
        image_query: Any = None,
    ) -> tuple[list[Any], list[str]]:
        vector_queries = []
        vector_types = []
        if text_query:
            vector_queries.append(text_query)
            vector_types.append("text")
        if image_query:
            vector_queries.append(image_query)
            vector_types.append("image")
        return vector_queries, vector_types

    def hybrid_search(
        self,
        text_query=None,
        image_query=None,
        ocr_query=None,
        subtitle_query=None,
        k: int = 100,
        weights=None,
        vector_models_config=None,
    ):
        if weights is None:
            weights = self._default_channel_weights()
        vector_models_config = self._resolve_vector_models_config(vector_models_config)
        raw_results = {channel: [] for channel in QUERY_CHANNELS}
        vector_queries, vector_types = self._prepare_hybrid_vector_queries(
            text_query=text_query,
            image_query=image_query,
        )
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_vector = executor.submit(
                self.vector_engine.search,
                vector_queries,
                vector_models_config,
                k,
            ) if vector_queries else None
            future_subtitle = executor.submit(self.ocr_engine.search_subtitle, subtitle_query, k) if subtitle_query else None
            future_ocr = executor.submit(self.ocr_engine.search_ocr, ocr_query, k) if ocr_query else None

            if future_vector is not None:
                batch_vector_results = future_vector.result()
                for channel, results in zip(vector_types, batch_vector_results):
                    raw_results[channel] = results
            if future_ocr is not None:
                raw_results["ocr"] = future_ocr.result()
            if future_subtitle is not None:
                raw_results["subtitle"] = future_subtitle.result()

        combined_results = self._fuse_and_rerank_candidates(
            raw_text_results=raw_results["text"],
            raw_image_results=raw_results["image"],
            raw_ocr_results=raw_results["ocr"],
            raw_subtitle_results=raw_results["subtitle"],
            weights=weights,
        )
        return combined_results[:k]

    def stage_search(
        self,
        stage_query: StageQuery,
        *,
        top_k: int = 100,
        visual_top_k: Optional[int] = None,
        subtitle_top_k: Optional[int] = None,
        ocr_top_k: Optional[int] = None,
        visual_weight: float = 0.45,
        ocr_weight: float = 0.35,
        subtitle_weight: float = 0.20,
    ) -> list[StageCandidate]:
        internal_k = max(
            int(top_k),
            int(visual_top_k) if visual_top_k is not None else int(top_k),
            int(subtitle_top_k) if subtitle_top_k is not None else int(top_k),
            int(ocr_top_k) if ocr_top_k is not None else int(top_k),
        )
        weights = {"text": visual_weight, "ocr": ocr_weight, "subtitle": subtitle_weight, "image": 0.0}
        weights = self._build_stage_weights(
            {"text": stage_query.visual, "ocr": stage_query.ocr, "subtitle": stage_query.subtitle},
            weights,
        )
        fused = self.hybrid_search(
            text_query=stage_query.visual or None,
            ocr_query=stage_query.ocr or None,
            subtitle_query=stage_query.subtitle or None,
            k=internal_k,
            weights=weights,
        )
        return [
            StageCandidate(video_id=video_id, frame_idx=frame_idx, fused_score=float(score))
            for (video_id, frame_idx), score in fused[: int(top_k)]
        ]

    def _group_temporal_candidates(self, temporal_candidates, frame_distance: int):
        temporal_groups = []
        for candidate in temporal_candidates:
            info = candidate[0]
            video, frame = info
            if (
                not temporal_groups
                or video != temporal_groups[-1][-1][0][0]
                or frame - temporal_groups[-1][-1][0][1] > frame_distance
            ):
                temporal_groups.append([])
            if temporal_groups[-1] and frame == temporal_groups[-1][-1][0][1]:
                prev_info, prev_scores, prev_key = temporal_groups[-1][-1]
                merged_scores = tuple(a + b for a, b in zip(prev_scores, candidate[1]))
                temporal_groups[-1][-1] = (prev_info, merged_scores, prev_key)
            else:
                temporal_groups[-1].append(candidate)
        return temporal_groups

    def _build_segment_timeline(self, temporal_group):
        timeline = []
        for info, stage_scores, key in temporal_group:
            video_name, frame_index = info
            timeline.append(
                {
                    "video_name": video_name,
                    "frame_index": frame_index,
                    "stage_scores": stage_scores,
                    "key": key,
                }
            )
        return timeline

    def _aggregate_stage_scores_over_windows(self, timeline, num_stages, window_size_frames=100):
        aggregated = []
        for center_item in timeline:
            center_frame_index = center_item["frame_index"]
            aggregated_scores = []
            for stage_idx in range(num_stages):
                best_score = 0.0
                best_key = None
                best_frame_index = None
                for neighbor in timeline:
                    if abs(neighbor["frame_index"] - center_frame_index) <= window_size_frames:
                        candidate_score = neighbor["stage_scores"][stage_idx]
                        if candidate_score > best_score:
                            best_score = candidate_score
                            best_key = neighbor["key"]
                            best_frame_index = neighbor["frame_index"]
                aggregated_scores.append(
                    {"score": best_score, "key": best_key, "frame_index": best_frame_index}
                )
            aggregated.append(
                {
                    "video_name": center_item["video_name"],
                    "center_frame_index": center_frame_index,
                    "stage_supports": aggregated_scores,
                }
            )
        return aggregated

    def _soft_temporal_dp(self, aggregated_timeline, num_stages, lambda_skip=0.7, min_stage_gap: int = 30):
        num_points = len(aggregated_timeline)
        best_score = [[0.0] * num_stages for _ in range(num_points)]
        decision = [[None] * num_stages for _ in range(num_points)]
        prev_state = [[None] * num_stages for _ in range(num_points)]
        last_key = [[None] * num_stages for _ in range(num_points)]
        last_frame = [[None] * num_stages for _ in range(num_points)]
        for time_idx in range(num_points):
            for stage_idx in range(num_stages):
                support = aggregated_timeline[time_idx]["stage_supports"][stage_idx]
                current_score = support["score"]
                current_key = support["key"]
                current_frame = support["frame_index"]
                if current_score > 0 and current_key is not None and current_frame is not None:
                    best_score[time_idx][stage_idx] = current_score
                    decision[time_idx][stage_idx] = "start"
                    last_key[time_idx][stage_idx] = current_key
                    last_frame[time_idx][stage_idx] = current_frame
                else:
                    decision[time_idx][stage_idx] = "empty"

                if time_idx > 0 and best_score[time_idx - 1][stage_idx] > best_score[time_idx][stage_idx]:
                    best_score[time_idx][stage_idx] = best_score[time_idx - 1][stage_idx]
                    decision[time_idx][stage_idx] = "carry"
                    prev_state[time_idx][stage_idx] = (time_idx - 1, stage_idx)
                    last_key[time_idx][stage_idx] = last_key[time_idx - 1][stage_idx]
                    last_frame[time_idx][stage_idx] = last_frame[time_idx - 1][stage_idx]

                if time_idx > 0 and stage_idx > 0 and current_score > 0 and current_key is not None and current_frame is not None:
                    prev_t = time_idx - 1
                    prev_s = stage_idx - 1
                    previous_score = best_score[prev_t][prev_s]
                    previous_frame = last_frame[prev_t][prev_s]
                    if previous_score > 0 and previous_frame is not None and current_frame - previous_frame >= min_stage_gap:
                        transition_score = previous_score + current_score
                        if transition_score > best_score[time_idx][stage_idx]:
                            best_score[time_idx][stage_idx] = transition_score
                            decision[time_idx][stage_idx] = "transition"
                            prev_state[time_idx][stage_idx] = (prev_t, prev_s)
                            last_key[time_idx][stage_idx] = current_key
                            last_frame[time_idx][stage_idx] = current_frame

                if stage_idx > 0:
                    best_previous_stage_score = 0.0
                    best_previous_stage_state = None
                    for previous_time_idx in range(time_idx + 1):
                        if best_score[previous_time_idx][stage_idx - 1] > best_previous_stage_score:
                            best_previous_stage_score = best_score[previous_time_idx][stage_idx - 1]
                            best_previous_stage_state = (previous_time_idx, stage_idx - 1)
                    skip_score = lambda_skip * best_previous_stage_score
                    if skip_score > best_score[time_idx][stage_idx] and best_previous_stage_state is not None:
                        prev_t, prev_s = best_previous_stage_state
                        best_score[time_idx][stage_idx] = skip_score
                        decision[time_idx][stage_idx] = "skip"
                        prev_state[time_idx][stage_idx] = best_previous_stage_state
                        last_key[time_idx][stage_idx] = last_key[prev_t][prev_s]
                        last_frame[time_idx][stage_idx] = last_frame[prev_t][prev_s]
        return best_score, decision, prev_state

    def _is_valid_temporal_chain(self, chain, min_stage_gap: int = 30):
        if len(chain) <= 1:
            return True
        seen_keys = set()
        previous_video_name = None
        previous_frame_index = None
        for key, _stage_scores, _matched_stage_idx in chain:
            if key in seen_keys:
                return False
            seen_keys.add(key)
            video_name, frame_index = key
            if previous_video_name is not None and video_name != previous_video_name:
                return False
            if previous_frame_index is not None and frame_index - previous_frame_index < min_stage_gap:
                return False
            previous_video_name = video_name
            previous_frame_index = frame_index
        return True

    def temporal_search(
        self,
        queries: Optional[list[dict[str, Any]]] = None,
        k: int = 10,
        frame_distance: int = 1500,
        initial_search_k: int = 2048,
        weights: Optional[dict[str, float]] = None,
        vector_models_config: Optional[list[dict[str, Any]]] = None,
        window_size_frames: int = 100,
        lambda_skip: float = 0.7,
        min_stage_gap: int = 30,
    ):
        cleaned_queries = []
        for stage_query in queries or []:
            cleaned = self._sanitize_stage_query(stage_query)
            if cleaned:
                cleaned_queries.append(cleaned)
        if not cleaned_queries:
            return []
        vector_models_config = self._resolve_vector_models_config(vector_models_config)
        num_stages = len(cleaned_queries)
        if num_stages <= 1:
            stage_query = cleaned_queries[0]
            stage_weights = self._build_stage_weights(stage_query, weights)
            results = self.hybrid_search(
                text_query=stage_query.get("text"),
                image_query=stage_query.get("image"),
                ocr_query=stage_query.get("ocr"),
                subtitle_query=stage_query.get("subtitle"),
                k=initial_search_k,
                weights=stage_weights,
                vector_models_config=vector_models_config,
            )
            return results[:k] if results else []

        vector_requests = []
        ocr_queries_to_process = []
        subtitle_queries_to_process = []
        for stage_idx, stage_data in enumerate(cleaned_queries):
            if stage_data.get("text"):
                vector_requests.append({"stage_idx": stage_idx, "channel": "text", "query": stage_data["text"]})
            if stage_data.get("image"):
                vector_requests.append({"stage_idx": stage_idx, "channel": "image", "query": stage_data["image"]})
            if stage_data.get("ocr"):
                ocr_queries_to_process.append((stage_idx, stage_data["ocr"]))
            if stage_data.get("subtitle"):
                subtitle_queries_to_process.append((stage_idx, stage_data["subtitle"]))

        vector_queries_to_process = [request["query"] for request in vector_requests]
        batch_vector_results = []
        ocr_results_by_stage = defaultdict(list)
        subtitle_results_by_stage = defaultdict(list)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_vector = executor.submit(
                self.vector_engine.search,
                vector_queries_to_process,
                vector_models_config,
                initial_search_k,
            ) if vector_queries_to_process else None
            future_ocr = {
                executor.submit(self.ocr_engine.search_ocr, ocr_text, 1024): stage_idx
                for stage_idx, ocr_text in ocr_queries_to_process
            }
            future_subtitle = {
                executor.submit(self.ocr_engine.search_subtitle, subtitle_text, 1024): stage_idx
                for stage_idx, subtitle_text in subtitle_queries_to_process
            }
            if future_vector is not None:
                batch_vector_results = future_vector.result()
            for future in concurrent.futures.as_completed(future_ocr):
                stage_idx = future_ocr[future]
                ocr_results_by_stage[stage_idx] = future.result()
            for future in concurrent.futures.as_completed(future_subtitle):
                stage_idx = future_subtitle[future]
                subtitle_results_by_stage[stage_idx] = future.result()

        raw_results_by_stage = defaultdict(lambda: {channel: [] for channel in QUERY_CHANNELS})
        for request, results in zip(vector_requests, batch_vector_results):
            raw_results_by_stage[request["stage_idx"]][request["channel"]] = results
        for stage_idx, results in ocr_results_by_stage.items():
            raw_results_by_stage[stage_idx]["ocr"] = results
        for stage_idx, results in subtitle_results_by_stage.items():
            raw_results_by_stage[stage_idx]["subtitle"] = results

        candidate_map = defaultdict(lambda: {"scores": [0.0] * num_stages, "info": None, "key": None})
        for stage_idx in range(num_stages):
            stage_data = raw_results_by_stage[stage_idx]
            stage_weights = self._build_stage_weights(cleaned_queries[stage_idx], weights)
            reranked_results = self._fuse_and_rerank_candidates(
                stage_data["text"],
                stage_data["image"],
                stage_data["ocr"],
                stage_data["subtitle"],
                stage_weights,
            )
            for key, score in reranked_results:
                candidate_map[key]["scores"][stage_idx] = score
                if candidate_map[key]["info"] is None:
                    video_name, frame_id = key
                    candidate_map[key]["info"] = (video_name, frame_id)
                    candidate_map[key]["key"] = key

        temporal_candidates = [
            (item["info"], tuple(item["scores"]), item["key"])
            for item in candidate_map.values()
            if item["info"] is not None
        ]
        if not temporal_candidates:
            return []
        temporal_candidates.sort(key=lambda item: (item[0][0], item[0][1]))
        temporal_groups = self._group_temporal_candidates(temporal_candidates, frame_distance)
        final_chains = []
        for temporal_group in temporal_groups:
            if not temporal_group:
                continue
            timeline = self._build_segment_timeline(temporal_group)
            aggregated_timeline = self._aggregate_stage_scores_over_windows(
                timeline=timeline,
                num_stages=num_stages,
                window_size_frames=window_size_frames,
            )
            best_score, decision, prev_state = self._soft_temporal_dp(
                aggregated_timeline=aggregated_timeline,
                num_stages=num_stages,
                lambda_skip=lambda_skip,
                min_stage_gap=min_stage_gap,
            )
            best_final_score = 0.0
            best_final_stage = -1
            best_final_time_idx = -1
            for time_idx in range(len(aggregated_timeline)):
                for stage_idx in range(num_stages):
                    if best_score[time_idx][stage_idx] > best_final_score:
                        best_final_score = best_score[time_idx][stage_idx]
                        best_final_stage = stage_idx
                        best_final_time_idx = time_idx
            if best_final_stage == -1 or best_final_score <= 0:
                continue
            chain = []
            skipped_stages = 0
            state = (best_final_time_idx, best_final_stage)
            while state is not None:
                time_idx, stage_idx = state
                action = decision[time_idx][stage_idx]
                if action == "carry":
                    state = prev_state[time_idx][stage_idx]
                    continue
                if action == "skip":
                    skipped_stages += 1
                    state = prev_state[time_idx][stage_idx]
                    continue
                if action in ("start", "transition"):
                    support = aggregated_timeline[time_idx]["stage_supports"][stage_idx]
                    if support["key"] is not None and support["score"] > 0:
                        stage_score_vector = tuple(
                            stage_support["score"]
                            for stage_support in aggregated_timeline[time_idx]["stage_supports"]
                        )
                        chain.append((support["key"], stage_score_vector, stage_idx))
                    state = prev_state[time_idx][stage_idx]
                    continue
                break
            chain.reverse()
            if chain and self._is_valid_temporal_chain(chain, min_stage_gap=min_stage_gap):
                final_chains.append(
                    {
                        "chain": chain,
                        "score": best_final_score,
                        "num_stages_matched": len(chain),
                        "num_stages_skipped": skipped_stages,
                    }
                )
        final_chains.sort(key=lambda item: (item["num_stages_matched"], item["score"]), reverse=True)
        output_results = []
        for item in final_chains[:k]:
            formatted_chain = []
            for key, stage_scores, matched_stage_idx in item["chain"]:
                formatted_chain.append((key, (stage_scores, item["score"], matched_stage_idx)))
            output_results.append(formatted_chain)
        return output_results
