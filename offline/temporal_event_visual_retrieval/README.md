# Temporal Event Visual Retrieval

Module nay huan luyen va encode visual-temporal retrieval theo huong phan cap:

- `keyframe embeddings -> temporal shot encoder -> shot embeddings`
- `shot embeddings -> temporal event encoder -> event embeddings`

Input chinh:

- `event_output/<video_id>/events.json`
- `event_captions/<video_id>.json`
- `shot_keyframe_embedding_module` output: `shot_keyframe_dir/<video_id>.pkl`

Text supervision:

- caption event duoc dua qua CLIP text encoder
- caption nen duoc dich sang tieng Anh o buoc rieng `prepare_translations`
- train va encode doc `translated_caption` tu file da chuan bi san

Shot supervision:

- dung `caption-to-shot similarity` lam tin hieu chinh
- dung `subtitle overlap` lam tin hieu phu
- tao `soft positive weights` cho `shot loss`

Kien truc:

- `keyframe + keyframe temporal metadata -> keyframe transformer`
- `attention pooling -> shot base embedding`
- `shot base embedding + shot metadata -> shot transformer`
- `attention pooling -> event embedding`
- text side giu CLIP text encoder goc va 2 projection heads:
  - `text_to_event_head`
  - `text_to_shot_head`

Lenh train:

```bash
python movie_event_retrieval_system/offline_temporal_event_visual_retrieval_main.py train \
  --event_dir ./event_output \
  --shot_keyframe_dir ./shot_keyframe_features \
  --event_captions_dir ./event_captions \
  --translated_captions_dir ./translated_event_captions \
  --clip_model_path /path/to/open_clip_pretrained.bin \
  --output_dir ./temporal_event_visual_outputs/run_01
```

Lenh dich caption truoc:

```bash
python movie_event_retrieval_system/offline_temporal_event_visual_retrieval_main.py prepare_translations \
  --event_captions_dir ./event_captions \
  --translation_model_path /path/to/Qwen3-4B-Instruct-2507 \
  --output_dir ./translated_event_captions \
  --device cuda
```

Lenh encode:

```bash
python movie_event_retrieval_system/offline_temporal_event_visual_retrieval_main.py encode \
  --event_dir ./event_output \
  --shot_keyframe_dir ./shot_keyframe_features \
  --clip_model_path /path/to/open_clip_pretrained.bin \
  --checkpoint_path ./temporal_event_visual_outputs/run_01/best.pt \
  --output_dir ./temporal_event_visual_outputs/encoded
```

Encode chi dung visual branch, nen khong can caption. Module se doc:

- `event_dir/<video_id>/events.json`
- `shot_keyframe_dir/<video_id>.pkl`
- optional `raw_subtitle_dir/<video_id>.json`
