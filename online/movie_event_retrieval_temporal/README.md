# Movie Event Retrieval Temporal

Module nay trien khai retrieval theo 2 tang:

1. Event-level coarse retrieval
   - event embedding FAISS
   - event caption embedding FAISS
   - subtitle embedding FAISS
   - OCR Meilisearch
2. Shot-level refinement
   - subset search tren shot FAISS bang `faiss.IDSelectorArray`
   - reuse subtitle/OCR evidence
   - cong them prior tu event-level

## Build store

```bash
python event_retrieval_temporal_main.py build-all \
  --event_dir ... \
  --event_embedding_dir ... \
  --caption_embedding_dir ... \
  --shot_embedding_dir ... \
  --subtitle_embedding_dir ... \
  --ocr_dir ... \
  --output_dir ... \
  --meilisearch_url http://127.0.0.1:7700 \
  --meilisearch_index_name movie_event_ocr
```

## Search

```bash
python event_retrieval_temporal_main.py search \
  --store_dir ... \
  --query_json query.json \
  --temporal_checkpoint_path ... \
  --caption_model_path ... \
  --subtitle_model_path ... \
  --meilisearch_url http://127.0.0.1:7700 \
  --meilisearch_index_name movie_event_ocr
```

`query.json` co the chua:

```json
{
  "raw_query": "...",
  "translated_query": "...",
  "subtitle_query": "...",
  "ocr_query": "..."
}
```
