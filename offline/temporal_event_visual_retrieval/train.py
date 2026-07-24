from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .dataset import HierarchicalTemporalEventDatasetBuilder, hierarchical_temporal_event_collate_fn
from .io_utils import ensure_dir, save_json, set_seed
from .model import HierarchicalTemporalRetrievalModel


def symmetric_contrastive_loss(query_embeddings: torch.Tensor, item_embeddings: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (query_embeddings @ item_embeddings.t()) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_q2i = F.cross_entropy(logits, labels)
    loss_i2q = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_q2i + loss_i2q)


def multi_positive_shot_loss(
    query_embeddings: torch.Tensor,
    shot_embeddings: torch.Tensor,
    shot_mask: torch.Tensor,
    positive_weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    batch_size, max_shots, dim = shot_embeddings.shape
    flat_shots = shot_embeddings.reshape(batch_size * max_shots, dim)
    flat_mask = shot_mask.reshape(batch_size * max_shots)
    logits = (query_embeddings @ flat_shots.t()) / temperature
    logits = logits.masked_fill(~flat_mask.unsqueeze(0), -1e9)

    flat_positive_weights = torch.zeros(batch_size, batch_size * max_shots, device=logits.device, dtype=torch.float32)
    for batch_index in range(batch_size):
        start = batch_index * max_shots
        end = start + max_shots
        batch_weights = positive_weights[batch_index] * shot_mask[batch_index].float()
        if float(batch_weights.sum().item()) <= 0:
            batch_weights = shot_mask[batch_index].float()
        batch_weights = batch_weights / batch_weights.sum().clamp_min(1e-12)
        flat_positive_weights[batch_index, start:end] = batch_weights

    log_probs = torch.log_softmax(logits, dim=-1)
    weighted_positive_log_prob = (flat_positive_weights * log_probs).sum(dim=-1)
    return (-weighted_positive_log_prob).mean()


def event_consistency_loss(event_embeddings: torch.Tensor, weighted_shot_summary: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(event_embeddings, weighted_shot_summary, dim=-1)).mean()


def shot_consistency_loss(
    shot_embeddings: torch.Tensor,
    shot_pre_event_embeddings: torch.Tensor,
    shot_mask: torch.Tensor,
) -> torch.Tensor:
    cosine = F.cosine_similarity(shot_embeddings, shot_pre_event_embeddings, dim=-1)
    masked = cosine * shot_mask.float()
    return 1.0 - (masked.sum() / shot_mask.float().sum().clamp_min(1.0))


def build_soft_positive_weights(
    *,
    text_embedding_base: torch.Tensor,
    shot_clip_embeddings: torch.Tensor,
    shot_mask: torch.Tensor,
    subtitle_overlap_scores: torch.Tensor,
    caption_shot_similarity_weight: float,
    subtitle_overlap_weight: float,
    soft_positive_temperature: float,
) -> torch.Tensor:
    caption_shot_similarity = torch.einsum("bd,bsd->bs", text_embedding_base, shot_clip_embeddings)
    combined_scores = (
        float(caption_shot_similarity_weight) * caption_shot_similarity
        + float(subtitle_overlap_weight) * subtitle_overlap_scores
    )
    combined_scores = combined_scores.masked_fill(~shot_mask, -1e9)
    combined_scores = combined_scores / max(float(soft_positive_temperature), 1e-6)
    weights = torch.softmax(combined_scores, dim=-1)
    return weights * shot_mask.float()


def build_model_from_batch(config: TrainConfig, batch: dict[str, object]) -> HierarchicalTemporalRetrievalModel:
    keyframe_dim = int(batch["keyframe_vectors"].shape[-1])
    keyframe_metadata_dim = int(batch["keyframe_metadata"].shape[-1])
    shot_metadata_dim = int(batch["shot_metadata"].shape[-1])
    return HierarchicalTemporalRetrievalModel(
        keyframe_dim=keyframe_dim,
        keyframe_metadata_dim=keyframe_metadata_dim,
        shot_metadata_dim=shot_metadata_dim,
        clip_model_path=str(config.clip_model_path),
        clip_model_name=config.clip_model_name,
        device=config.device,
        keyframe_hidden_dim=config.keyframe_hidden_dim,
        shot_hidden_dim=config.shot_hidden_dim,
        projection_dim=config.projection_dim,
        keyframe_transformer_layers=config.keyframe_transformer_layers,
        shot_transformer_layers=config.shot_transformer_layers,
        keyframe_transformer_heads=config.keyframe_transformer_heads,
        shot_transformer_heads=config.shot_transformer_heads,
        dropout=config.dropout,
    )


def train(config: TrainConfig) -> Path:
    set_seed(config.seed)
    ensure_dir(config.output_dir)

    dataset = HierarchicalTemporalEventDatasetBuilder(config).build_dataset()
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=hierarchical_temporal_event_collate_fn,
    )
    first_batch = next(iter(loader))
    model = build_model_from_batch(config, first_batch).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_log = []
    best_loss = None
    best_path = config.output_dir / "best.pt"

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(loader, desc=f"Train epoch {epoch + 1}/{config.num_epochs}")
        for batch in progress:
            keyframe_vectors = batch["keyframe_vectors"].to(config.device)
            keyframe_metadata = batch["keyframe_metadata"].to(config.device)
            keyframe_mask = batch["keyframe_mask"].to(config.device)
            shot_metadata = batch["shot_metadata"].to(config.device)
            shot_mask = batch["shot_mask"].to(config.device)
            subtitle_overlap_scores = batch["subtitle_overlap_scores"].to(config.device)

            outputs = model(
                keyframe_vectors=keyframe_vectors,
                keyframe_metadata=keyframe_metadata,
                keyframe_mask=keyframe_mask,
                shot_metadata=shot_metadata,
                shot_mask=shot_mask,
                texts=batch["translated_captions"],
            )
            soft_positive_weights = build_soft_positive_weights(
                text_embedding_base=outputs["text_embedding_base"],
                shot_clip_embeddings=outputs["shot_clip_embeddings"],
                shot_mask=shot_mask,
                subtitle_overlap_scores=subtitle_overlap_scores,
                caption_shot_similarity_weight=config.caption_shot_similarity_weight,
                subtitle_overlap_weight=config.subtitle_overlap_weight,
                soft_positive_temperature=config.soft_positive_temperature,
            )
            event_loss = symmetric_contrastive_loss(
                outputs["event_text_embeddings"],
                outputs["event_embeddings"],
                temperature=config.temperature,
            )
            shot_loss = multi_positive_shot_loss(
                outputs["shot_text_embeddings"],
                outputs["shot_embeddings"],
                shot_mask=shot_mask,
                positive_weights=soft_positive_weights,
                temperature=config.temperature,
            )
            event_cons = event_consistency_loss(
                outputs["event_embeddings"],
                outputs["weighted_shot_summary"],
            )
            shot_cons = shot_consistency_loss(
                outputs["shot_embeddings"],
                outputs["shot_pre_event_embeddings"],
                shot_mask=shot_mask,
            )
            loss = (
                config.event_loss_weight * event_loss
                + config.shot_loss_weight * shot_loss
                + config.event_consistency_loss_weight * event_cons
                + config.shot_consistency_loss_weight * shot_cons
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                event=f"{event_loss.item():.4f}",
                shot=f"{shot_loss.item():.4f}",
                event_cons=f"{event_cons.item():.4f}",
                shot_cons=f"{shot_cons.item():.4f}",
            )

        epoch_loss /= max(len(loader), 1)
        train_log.append({"epoch": epoch + 1, "loss": epoch_loss})
        if best_loss is None or epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "config": asdict(config),
                    "model_state_dict": model.state_dict(),
                    "best_loss": best_loss,
                },
                best_path,
            )

    save_json(
        {
            "config": asdict(config),
            "num_samples": len(dataset),
            "train_log": train_log,
            "best_loss": best_loss,
            "best_checkpoint": str(best_path),
        },
        config.output_dir / "train_summary.json",
    )
    return best_path
