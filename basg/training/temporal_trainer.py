from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from basg.evaluation.dual_expert_evaluator import evaluate_popdyn_expert
from basg.features.temporal import build_prefix_temporal_features
from basg.utils.runtime import (
    RunClock,
    close_progress,
    format_seconds,
    loader_total,
    log_event,
    parse_bool,
    progress_iter,
    resolve_progress_enabled,
    set_progress_postfix,
)


def _atomic_torch_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sample_train_negatives(
    histories: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_items: int,
    count: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Source-only rejection sampler for BPR negatives."""
    history_rows = histories.detach().cpu().tolist()
    target_rows = targets.detach().cpu().tolist()
    output: list[list[int]] = []
    for history, target in zip(history_rows, target_rows):
        blocked = {int(x) for x in history if int(x) > 0}
        blocked.add(int(target))
        row: list[int] = []
        attempts = 0
        limit = max(100, count * 100)
        while len(row) < count and attempts < limit:
            item = int(rng.integers(1, num_items + 1))
            if item not in blocked:
                row.append(item)
            attempts += 1
        while len(row) < count:
            row.append(int(rng.integers(1, num_items + 1)))
        output.append(row)
    return torch.as_tensor(output, dtype=torch.long)


def train_popdyn_expert(
    *,
    model: torch.nn.Module,
    semantic_item_features: torch.Tensor,
    train_loader: Iterable,
    selection_loader: Iterable,
    device: torch.device,
    num_items: int,
    checkpoint_path: str | Path,
    training_cfg: dict,
    evaluation_cfg: dict,
    temporal_cfg: dict,
    seed: int,
    source_domain: str,
    run_name: str,
) -> dict:
    """
    Train the temporal expert only on source samples.

    `semantic_item_features` is a fixed source item table. Target semantic
    tables are never loaded by this function.
    """
    epochs = int(training_cfg.get("epochs", 80))
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))
    train_negatives = int(training_cfg.get("train_negatives", 5))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    patience_limit = int(training_cfg.get("early_stop_patience", 10))
    max_batches = int(training_cfg.get("max_batches_per_epoch", 0))
    primary_metric = str(evaluation_cfg.get("primary_metric", "NDCG@10"))
    eval_seed = int(seed) + int(evaluation_cfg.get("fixed_negative_seed_offset", 10000))
    time_scale_ms = float(temporal_cfg.get("time_scale_ms", 86_400_000.0))
    runtime_cfg = dict(training_cfg.get("runtime", {}) or {})
    show_progress = resolve_progress_enabled(runtime_cfg.get("show_progress", True))
    progress_refresh = float(runtime_cfg.get("progress_refresh_seconds", 0.5))
    leave_progress = parse_bool(runtime_cfg.get("leave_progress_bar", True), True)

    # ── Ablation config (retrained input-channel ablations) ──────────
    ablation_cfg = dict(training_cfg.get("ablation", {}) or {})
    use_temporal_features = bool(ablation_cfg.get("use_temporal_features", True))
    use_interaction_property = bool(ablation_cfg.get("use_interaction_property", True))

    # ── Load popularity lookup table (PrepRec-style source popularity) ──
    pop_lookup: torch.Tensor | None = training_cfg.get("pop_lookup")
    if pop_lookup is not None:
        log_event("TRAIN", "POP_LOOKUP", enabled=True,
                  shape=tuple(pop_lookup.shape),
                  min=float(pop_lookup.min().item()),
                  max=float(pop_lookup.max().item()))

    # ── Load behavior lookup table (AlphaFree-style behavior-augmented) ──
    behavior_lookup: torch.Tensor | None = training_cfg.get("behavior_lookup")
    if behavior_lookup is not None:
        log_event("TRAIN", "BEHAVIOR_LOOKUP", enabled=True,
                  shape=tuple(behavior_lookup.shape))

    if semantic_item_features.ndim != 2 or semantic_item_features.shape[0] != num_items + 1:
        raise ValueError("semantic_item_features must be [num_items + 1, dim].")
    item_table_cpu = semantic_item_features.detach().float().cpu()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    model.to(device)
    best_metric = -float("inf")
    best_epoch = 0
    patience = 0
    history: list[dict] = []
    clock = RunClock()

    log_event(
        "TRAIN",
        "START",
        run=run_name,
        epochs=epochs,
        train_batches=loader_total(train_loader, max_batches),
        val_batches=loader_total(selection_loader, int(evaluation_cfg.get("max_batches", 0))),
        device=device,
    )

    # Move behavior_lookup to device once (v8 AlphaFree)
    beh_lookup_dev = behavior_lookup.to(device) if behavior_lookup is not None else None

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        pairs = 0
        latest_grad_norm = 0.0
        rng = np.random.default_rng(int(seed) + epoch * 104729)

        train_iterator = progress_iter(
            train_loader,
            total=loader_total(train_loader, max_batches),
            run_name=run_name,
            phase="TRAIN",
            detail=f"epoch={epoch:03d}/{epochs:03d}",
            enabled=show_progress,
            refresh_seconds=progress_refresh,
            leave=leave_progress,
        )
        try:
            for batch_index, batch in enumerate(train_iterator):
                if max_batches > 0 and batch_index >= max_batches:
                    break

                history_ids_cpu = batch["history"]
                history_times_cpu = batch["history_times"]
                targets_cpu = batch["target"]

                negatives_cpu = _sample_train_negatives(
                    history_ids_cpu,
                    targets_cpu,
                    num_items=int(num_items),
                    count=train_negatives,
                    rng=rng,
                )

                history_ids = history_ids_cpu.to(device, non_blocking=True)
                history_features = build_prefix_temporal_features(
                    history_ids_cpu,
                    history_times_cpu,
                    time_scale_ms=time_scale_ms,
                    pop_lookup=pop_lookup,
                    use_temporal_features=use_temporal_features,
                    use_interaction_property=use_interaction_property,
                ).to(device, non_blocking=True)
                if not torch.isfinite(history_features).all():
                    raise FloatingPointError(
                        f"Non-finite history_features at epoch={epoch}, batch={batch_index}"
                    )

                positive_semantics = item_table_cpu[targets_cpu].unsqueeze(1).to(
                    device, non_blocking=True
                )
                positive_semantics = torch.nan_to_num(
                    positive_semantics, nan=0.0, posinf=0.0, neginf=0.0,
                )
                negative_semantics = item_table_cpu[negatives_cpu].to(
                    device, non_blocking=True
                )
                negative_semantics = torch.nan_to_num(
                    negative_semantics, nan=0.0, posinf=0.0, neginf=0.0,
                )

                state = model.encode_sequence(history_ids, history_features, beh_lookup_dev,
                                               use_interaction_property=use_interaction_property)

                # v8 AlphaFree: pass item IDs + behavior_lookup — model does the concat
                pos_ids = targets_cpu.unsqueeze(1).to(device, non_blocking=True)  # [B, 1]
                neg_ids = negatives_cpu.to(device, non_blocking=True)              # [B, N]

                positive_state = model.encode_candidates(positive_semantics, pos_ids, beh_lookup_dev,
                                                         use_interaction_property=use_interaction_property)
                negative_state = model.encode_candidates(negative_semantics, neg_ids, beh_lookup_dev,
                                                         use_interaction_property=use_interaction_property)

                # Use L2-normalised scoring (same as eval / score_candidates).
                positive_scores = model._score_encoded(state, positive_state)   # [B, 1]
                negative_scores = model._score_encoded(state, negative_state)   # [B, N]
                loss = -torch.nn.functional.logsigmoid(positive_scores - negative_scores).mean()

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite temporal-expert loss at epoch={epoch}, batch={batch_index}"
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                    error_if_nonfinite=True,
                )
                latest_grad_norm = float(grad_norm.detach().cpu())
                optimizer.step()
                max_param_abs = 0.0
                max_param_name = None
                for name, param in model.named_parameters():
                    if not torch.isfinite(param).all():
                        raise FloatingPointError(
                            f"Non-finite parameter after optimizer.step(): {name}"
                        )
                    cur = float(param.detach().abs().max().cpu())
                    if cur > max_param_abs:
                        max_param_abs = cur
                        max_param_name = name
                if max_param_abs > 1000.0:
                    raise FloatingPointError(
                        f"Parameter magnitude exploded after optimizer.step(): "
                        f"name={max_param_name}, absmax={max_param_abs}"
                    )

                batch_size = int(targets_cpu.shape[0])
                loss_sum += float(loss.detach()) * batch_size
                pairs += batch_size
                set_progress_postfix(
                    train_iterator,
                    loss=loss_sum / max(pairs, 1),
                    grad=latest_grad_norm,
                    examples=pairs,
                    elapsed=format_seconds(clock.elapsed()),
                )
        finally:
            close_progress(train_iterator)

        if pairs == 0:
            raise RuntimeError("No temporal-expert training batches were processed.")

        train_seconds = float(time.perf_counter() - train_started)

        # ── Post-train smoke: one val batch before full evaluation ──
        smoke_iter = iter(selection_loader)
        try:
            smoke_batch = next(smoke_iter)
        except StopIteration:
            smoke_batch = None
        if smoke_batch is not None:
            model.eval()
            with torch.no_grad():
                s_ids = smoke_batch["history"].to(device)
                s_mask = s_ids.ne(0)
                s_empty = int((~s_mask.any(dim=1)).sum().item())
                s_feats = build_prefix_temporal_features(
                    smoke_batch["history"],
                    smoke_batch["history_times"],
                    time_scale_ms=time_scale_ms,
                    pop_lookup=pop_lookup,
                    use_temporal_features=use_temporal_features,
                    use_interaction_property=use_interaction_property,
                ).to(device)
                s_sem = item_table_cpu[
                    smoke_batch["target"]
                ].unsqueeze(1).to(device)  # [B, 1, D]
                s_tgt_ids = smoke_batch["target"].unsqueeze(1).to(device)  # [B, 1]
                s_scores = model.score_candidates(s_ids, s_feats, s_sem, s_tgt_ids, beh_lookup_dev,
                                                  use_interaction_property=use_interaction_property)
                log_event(
                    "SMOKE",
                    "POST_TRAIN",
                    empty_rows=s_empty,
                    feat_absmax=float(s_feats.abs().max().item()),
                    score_min=float(s_scores.min().item()),
                    score_max=float(s_scores.max().item()),
                    score_finite=bool(torch.isfinite(s_scores).all().item()),
                )
                if not torch.isfinite(s_scores).all():
                    raise FloatingPointError(
                        "Post-train smoke produced NaN/Inf scores. "
                        f"empty_rows={s_empty}, "
                        f"feat_absmax={float(s_feats.abs().max().item())}"
                    )
        validation_started = time.perf_counter()
        metrics = evaluate_popdyn_expert(
            model=model,
            semantic_item_features=item_table_cpu,
            loader=selection_loader,
            device=device,
            num_items=int(num_items),
            ranking=str(evaluation_cfg.get("ranking", "sampled")),
            eval_negatives=int(evaluation_cfg.get("eval_negatives", 100)),
            ks=tuple(int(x) for x in evaluation_cfg.get("ks", [10, 20])),
            tie_policy=str(evaluation_cfg.get("tie_policy", "worst")),
            seed=eval_seed,
            time_scale_ms=time_scale_ms,
            max_batches=int(evaluation_cfg.get("max_batches", 0)),
            show_progress=show_progress,
            progress_desc=run_name,
            progress_refresh=progress_refresh,
            leave_progress=leave_progress,
            pop_lookup=pop_lookup,
            behavior_lookup=behavior_lookup,
            use_temporal_features=use_temporal_features,
            use_interaction_property=use_interaction_property,
        )
        validation_seconds = float(time.perf_counter() - validation_started)
        current = float(metrics[primary_metric])
        row = {
            "epoch": int(epoch),
            "loss": float(loss_sum / pairs),
            "grad_norm": float(latest_grad_norm),
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "epoch_seconds": float(time.perf_counter() - epoch_started),
            "elapsed_seconds": float(clock.elapsed()),
            **{f"val_{key}": float(value) for key, value in metrics.items()},
        }
        row["eta_seconds"] = float(clock.eta(epoch, epochs) or 0.0)
        history.append(row)

        improved = current > best_metric + 1e-12
        if improved:
            best_metric = current
            best_epoch = epoch
            patience = 0
            _atomic_torch_save(
                {
                    "format_version": 2,
                    "run_name": run_name,
                    "source_domain": source_domain,
                    "seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "best_metric_name": primary_metric,
                    "best_metric_value": float(best_metric),
                    "model_hparams": model.export_hparams(),
                    "model_state": model.state_dict(),
                    "training_cfg": dict(training_cfg),
                    "evaluation_cfg": dict(evaluation_cfg),
                    "temporal_cfg": dict(temporal_cfg),
                    "protocol": {
                        "training_domain": source_domain,
                        "target_interactions_loaded_during_training": False,
                        "target_catalog_statistics_used": False,
                        "target_model_selection_used": False,
                        "prefix_temporal_features": "per_user_prefix_only",
                    },
                    "history": list(history),
                },
                checkpoint_path,
            )
        else:
            patience += 1

        beh_diag = model.behavior_diagnostics() if hasattr(model, "behavior_diagnostics") else {}
        log_event(
            "TRAIN",
            "EPOCH",
            run=run_name,
            epoch=f"{epoch:03d}/{epochs:03d}",
            loss=row["loss"],
            validation_metric=f"{primary_metric}:{current:.6f}",
            best=f"{best_metric:.6f}@{best_epoch}",
            status="checkpoint_saved" if improved else f"patience_{patience}/{patience_limit}",
            elapsed=format_seconds(clock.elapsed()),
            eta=format_seconds(clock.eta(epoch, epochs)),
            **{f"beh_{k}": v for k, v in beh_diag.items()},
        )
        if patience >= patience_limit:
            log_event(
                "TRAIN",
                "EARLY_STOP",
                run=run_name,
                epoch=epoch,
                elapsed=format_seconds(clock.elapsed()),
            )
            break

    if best_epoch == 0:
        raise RuntimeError("Temporal-expert checkpoint was not saved.")
    return {
        "run_name": run_name,
        "source_domain": source_domain,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_metric_name": primary_metric,
        "best_metric_value": float(best_metric),
        "checkpoint_path": str(checkpoint_path),
        "elapsed_seconds": float(clock.elapsed()),
        "history": history,
    }
