# Movie Event Retrieval System

Thu muc nay duoc chia theo 2 phase lon:

1. `offline/`
Chua cac task tien xu ly, embedding, indexing va cac buoc chuan bi du lieu.

Ben trong `offline/` hien tai da co cac nhom task:

1. `shot_detection`
Detect shot boundary va sinh file shots JSON cho tung video.

2. `shot_visual_keyframe_embedding`
Trich xuat keyframe embedding ben trong moi shot.

3. `shot_action_feature_extraction`
Trich xuat dac trung action cho moi shot bang SlowFast.

4. `subtitle_ocr_extraction`
Trich xuat subtitle/OCR tu video.

5. `shot_face_continuity_feature_extraction`
Trich xuat face continuity features va boundary-related face signals theo shot.

6. `text_embedding`
Embed subtitle va event caption bang Vietnamese embedding model.

7. `event_boundary_detection`
Dung cac modality o muc shot de group shot thanh event va sinh event-level outputs.

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
python offline_shot_detection_main.py ...
python offline_shot_visual_keyframe_embedding_main.py ...
python offline_shot_action_feature_extraction_main.py ...
python offline_subtitle_ocr_extraction_main.py ...
python offline_shot_face_continuity_feature_extraction_main.py ...
python offline_text_embedding_main.py ...
python offline_event_boundary_detection_main.py ...
```

Khong con giu cac thu muc retrieval cung cap voi `online/` nua.
Cau truc hien tai chi de lai `online/` va `offline/` cho ro phase.
