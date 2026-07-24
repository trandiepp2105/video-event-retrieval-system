# Movie Event Retrieval System

Thu muc nay duoc chia theo 2 phase lon:

1. `offline/`
Chua cac task tien xu ly, embedding, indexing va cac buoc chuan bi du lieu.

2. `online/`
Chua cac module retrieval/search phuc vu pha truy van.

Ben trong `online/` hien tai co:

1. `movie_keyframe_retrieval`
Da duoc trien khai va co the chay duoc hien tai.

2. `movie_event_retrieval_pooling`
Khung cho event retrieval theo huong pooling, chua trien khai logic.

3. `movie_event_retrieval_temporal`
Khung cho event retrieval theo huong temporal, chua trien khai logic.

Entry scripts o muc goc van duoc giu lai de tranh vo cach chay cu:

```bash
python keyframe_retrieval_main.py ...
python event_retrieval_pooling_main.py
python event_retrieval_temporal_main.py
python offline_text_embedding_main.py ...
```

Khong con giu cac thu muc retrieval cung cap voi `online/` nua.
Cau truc hien tai chi de lai `online/` va `offline/` cho ro phase.
