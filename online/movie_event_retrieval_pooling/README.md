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
- `subtitle embeddings`
- `OCR` qua Meilisearch

CLI:

```bash
python event_retrieval_pooling_main.py build-all ...
python event_retrieval_pooling_main.py search ...
```
