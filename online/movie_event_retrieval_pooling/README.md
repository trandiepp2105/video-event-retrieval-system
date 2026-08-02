# Movie Event Retrieval Pooling

Module nay chay cung logic voi `movie_event_retrieval_temporal`, nhung visual query cho:

- `event-level search`
- `shot-level search`

duoc encode truc tiep bang CLIP text encoder thay vi checkpoint temporal.

Input visual embeddings:
- `event embeddings` pooling
- `shot embeddings` pooling

Input text embeddings:
- `caption embeddings`
- `subtitle Meilisearch` mac dinh
- `subtitle embeddings` giu lai de co the bat lai backend `embedding`
- `OCR` qua Meilisearch

CLI:

```bash
python event_retrieval_pooling_main.py build-all ...
python event_retrieval_pooling_main.py search ...
```

## Python service

`PoolingRetrievalService.load()` nap metadata, FAISS indexes, CLIP, text
encoder, query analyzer va kiem tra Meilisearch mot lan. Cac lan goi
`search()` sau do tai su dung cung tai nguyen.

```python
from pathlib import Path

from online.movie_event_retrieval_pooling.config import SearchConfig
from online.movie_event_retrieval_pooling.service import PoolingRetrievalService

service = PoolingRetrievalService(
    SearchConfig(
        store_dir=Path("/path/to/pooling_retrieval_store"),
        clip_model_path=Path("/path/to/open_clip_weights.pt"),
        caption_model_path="/path/to/multilingual-e5-large-instruct",
        enable_shot_temporal=True,
        temporal_query_model_path="/path/to/qwen-model",
        meilisearch_url="http://127.0.0.1:7700",
        meilisearch_index_name="movie_event_pooling_ocr",
        subtitle_meilisearch_index_name="movie_event_pooling_subtitle",
        meilisearch_api_key="meilisearch-api-key",
    )
)
service.load()
segments = service.search("Mot nguoi dan ong roi khoi can phong", top_k=10)
```

## FastAPI

Dam bao Meilisearch dang chay va them `NGROK_AUTHTOKEN` vao Kaggle
Secrets. Khoi dong API bang args, khong can file config JSON:

```bash
python event_retrieval_pooling_api_main.py \
  --store_dir /path/to/pooling_retrieval_store \
  --clip_model_path /path/to/open_clip_weights.pt \
  --clip_model_name ViT-H-14-quickgelu \
  --caption_model_path /path/to/multilingual-e5-large-instruct \
  --subtitle_backend meilisearch \
  --visual_device cuda:0 \
  --caption_device cuda:0 \
  --enable_shot_temporal \
  --temporal_query_model_path /path/to/qwen-model \
  --temporal_query_device_map auto \
  --meilisearch_url http://127.0.0.1:7700 \
  --meilisearch_index_name movie_event_pooling_ocr \
  --subtitle_meilisearch_index_name movie_event_pooling_subtitle \
  --meilisearch_api_key meilisearch-api-key \
  --host 0.0.0.0 \
  --port 8000
```

Khi chay thanh cong, terminal in:

```text
Public URL: https://example.ngrok-free.app
Search endpoint: https://example.ngrok-free.app/search
Health endpoint: https://example.ngrok-free.app/health
```

Co the truyen token truc tiep bang `--ngrok_authtoken`, hoac tat ngrok
bang `--no_ngrok` khi chi can truy cap local.

Kiem tra trang thai:

```bash
curl http://127.0.0.1:8000/health
```

Search:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Mot nguoi dan ong roi khoi can phong", "top_k":10}'
```

Response la mot JSON array:

```json
[
  {
    "video_id": "10",
    "start_time_sec": 120.4,
    "end_time_sec": 138.8
  }
]
```
