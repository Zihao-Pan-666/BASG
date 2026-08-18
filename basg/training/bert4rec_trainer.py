from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from basg.evaluation.bert4rec_family_evaluator import evaluate_bert4rec_family
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
from basg.training.bert4rec_family_losses import (
    compute_alignment_loss,
    dynamic_role_distillation_loss,
    stable_bpr_loss,
)



def _assert_finite(name: str, value: torch.Tensor, context: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(
            f"{name} contains NaN/Inf at {context}. "
            "Training stopped before optimizer.step/checkpoint save."
        )


def _sample_training_negatives(
    histories: torch.Tensor,
    positives: torch.Tensor,
    num_items: int,
    num_negatives: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Sample negatives while excluding every non-PAD history item and the target.
    """
    history_cpu = histories.detach().cpu().numpy()
    positive_cpu = positives.detach().cpu().numpy()
    rows: list[list[int]] = []

    for history, positive in zip(history_cpu, positive_cpu):
        blocked = {int(x) for x in history.tolist() if int(x) > 0}
        blocked.add(int(positive))

        available = num_items - len([x for x in blocked if 1 <= x <= num_items])
        if available <= 0:
            raise ValueError("No valid training negatives remain")

        values: list[int] = []
        while len(values) < num_negatives:
            draw = torch.randint(
                low=1,
                high=num_items + 1,
                size=(max(32, num_negatives * 4),),
                generator=generator,
            ).tolist()
            for item in draw:
                item = int(item)
                if item not in blocked:
                    values.append(item)
                    if len(values) == num_negatives:
                        break
        rows.append(values)

    return torch.as_tensor(rows, dtype=torch.long, device=histories.device)


class AuxiliaryPoolSampler:
    def __init__(
        self,
        pools: dict[str, torch.Tensor],
        samples_per_domain: int,
        seed: int,
    ) -> None:
        self.pools = pools
        self.samples_per_domain = int(samples_per_domain)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def sample(self, device: torch.device) -> list[tuple[str, torch.Tensor]]:
        sampled: list[tuple[str, torch.Tensor]] = []
        for domain, pool in self.pools.items():
            if pool.ndim != 2 or pool.shape[0] == 0:
                raise ValueError(f"Invalid auxiliary pool for {domain}")
            size = min(self.samples_per_domain, int(pool.shape[0]))
            indices = torch.randperm(
                int(pool.shape[0]), generator=self.generator
            )[:size]
            sampled.append((domain, pool[indices].to(device, non_blocking=True)))
        return sampled


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _build_alignment_batch(
    model: torch.nn.Module,
    positives: torch.Tensor,
    auxiliary_sampler: AuxiliaryPoolSampler,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_raw = model.raw_item_features(positives).float()
    raw_parts = [source_raw]
    labels = [
        torch.zeros(source_raw.shape[0], dtype=torch.long, device=device)
    ]

    for domain_index, (_, auxiliary_raw) in enumerate(
        auxiliary_sampler.sample(device), start=1
    ):
        if auxiliary_raw.shape[1] != source_raw.shape[1]:
            raise ValueError(
                "Source and auxiliary semantic embedding dimensions differ: "
                f"{source_raw.shape[1]} vs {auxiliary_raw.shape[1]}"
            )
        raw_parts.append(auxiliary_raw.float())
        labels.append(
            torch.full(
                (auxiliary_raw.shape[0],),
                domain_index,
                dtype=torch.long,
                device=device,
            )
        )

    raw = torch.cat(raw_parts, dim=0)
    domain_labels = torch.cat(labels, dim=0)
    projected = model.project_raw_for_alignment(raw)
    return projected, raw, domain_labels


def _as_bool(value: object, default: bool = False) -> bool:
    return parse_bool(value, default=default)


def _infer_sequence_hidden_dim(model: torch.nn.Module) -> int:
    if hasattr(model, "hidden_dim"):
        return int(getattr(model, "hidden_dim"))
    if hasattr(model, "export_hparams"):
        hparams = model.export_hparams()
        if "hidden_dim" in hparams:
            return int(hparams["hidden_dim"])
    raise AttributeError(
        "Cannot infer sequence hidden dimension. Expected model.hidden_dim "
        "or model.export_hparams()['hidden_dim']."
    )


def _extract_role_table(payload: object) -> torch.Tensor:
    if torch.is_tensor(payload):
        return payload.detach().cpu()
    if isinstance(payload, dict):
        for key in (
            "source_role_table",
            "role_table",
            "item_role_table",
            "roles",
            "table",
        ):
            value = payload.get(key)
            if torch.is_tensor(value):
                return value.detach().cpu()
        tensor_values = [value for value in payload.values() if torch.is_tensor(value)]
        if len(tensor_values) == 1:
            return tensor_values[0].detach().cpu()
        raise KeyError(
            "Could not find a unique role table tensor in the loaded payload. "
            f"Available keys: {list(payload.keys())}"
        )
    raise TypeError(
        "Dynamic role table must be a torch.Tensor or a dict containing a tensor."
    )


def _load_dynamic_role_table(
    dynamic_role_cfg: dict,
    num_items: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    path_value = (
        dynamic_role_cfg.get("role_table")
        or dynamic_role_cfg.get("role_table_path")
        or dynamic_role_cfg.get("source_role_table")
    )
    if not path_value:
        raise ValueError(
            "training.dynamic_role.enabled=true requires role_table or role_table_path"
        )

    path = Path(str(path_value))
    if not path.exists():
        raise FileNotFoundError(path)

    payload = torch.load(path, map_location="cpu")
    table = _extract_role_table(payload)

    if table.ndim == 1:
        num_roles = int(dynamic_role_cfg.get("num_roles", int(table.max().item()) + 1))
        hard = table.long().clamp_min(0)
        table = torch.nn.functional.one_hot(hard, num_classes=num_roles).float()
        if table.shape[0] > 0:
            table[0].zero_()
    elif table.ndim != 2:
        raise ValueError(
            f"Role table must be [num_items+1, K] or hard labels [num_items+1], "
            f"got {tuple(table.shape)}"
        )

    expected = int(num_items) + 1
    if int(table.shape[0]) < expected:
        raise ValueError(
            f"Role table has too few rows: {table.shape[0]} < {expected}. "
            "It must be aligned to the source item map with PAD row 0."
        )
    if int(table.shape[0]) > expected:
        table = table[:expected]

    table = table.float()
    if not torch.isfinite(table).all():
        raise FloatingPointError(f"Role table contains NaN/Inf: {path}")

    table.clamp_(min=0.0)
    if table.shape[0] > 0:
        table[0].zero_()

    row_mass = table.sum(dim=-1)
    valid_rows = row_mass > float(dynamic_role_cfg.get("min_role_mass", 1e-8))
    if int(valid_rows[1:].sum().item()) == 0:
        raise ValueError(f"No non-empty item role rows found in {path}")

    metadata = {
        "role_table_path": str(path),
        "num_roles": int(table.shape[1]),
        "role_table_rows": int(table.shape[0]),
        "role_valid_item_rows": int(valid_rows[1:].sum().item()),
        "role_valid_item_ratio": float(valid_rows[1:].float().mean().item()),
        "role_table_min_mass": float(row_mass[1:].min().item()) if table.shape[0] > 1 else 0.0,
        "role_table_max_mass": float(row_mass[1:].max().item()) if table.shape[0] > 1 else 0.0,
    }
    return table.to(device=device, non_blocking=True), metadata


def _build_role_head(
    model: torch.nn.Module,
    num_roles: int,
    device: torch.device,
) -> nn.Linear:
    role_head = nn.Linear(_infer_sequence_hidden_dim(model), int(num_roles))
    nn.init.xavier_uniform_(role_head.weight)
    nn.init.zeros_(role_head.bias)
    return role_head.to(device)



def train_bert4rec_family(
    model: torch.nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    device: torch.device,
    num_items: int,
    mode: str,
    auxiliary_pools: dict[str, torch.Tensor],
    checkpoint_path: str | Path,
    training_cfg: dict,
    alignment_cfg: dict,
    evaluation_cfg: dict,
    seed: int,
    source_domain: str,
    run_name: str,
) -> dict[str, object]:
    mode = str(mode).lower()
    if mode not in {"sem", "arch0", "recg", "sage"}:
        raise ValueError(f"Unsupported mode: {mode}")

    dynamic_role_cfg = dict(training_cfg.get("dynamic_role", {}) or {})
    dynamic_role_enabled = _as_bool(dynamic_role_cfg.get("enabled", False), False)
    role_table: torch.Tensor | None = None
    role_head: nn.Linear | None = None
    role_metadata: dict[str, object] = {"enabled": False}

    optimizer_parameters = list(model.parameters())
    if dynamic_role_enabled:
        role_table, role_table_metadata = _load_dynamic_role_table(
            dynamic_role_cfg=dynamic_role_cfg,
            num_items=num_items,
            device=device,
        )
        role_head = _build_role_head(
            model=model,
            num_roles=int(role_table.shape[1]),
            device=device,
        )
        optimizer_parameters += list(role_head.parameters())
        role_metadata = {
            "enabled": True,
            **role_table_metadata,
            "lambda_role": float(dynamic_role_cfg.get("lambda_role", 0.1)),
            "loss": str(dynamic_role_cfg.get("loss", "kl")),
            "detach_sequence_state": _as_bool(
                dynamic_role_cfg.get("detach_sequence_state", False), False
            ),
            "normalize_targets": _as_bool(
                dynamic_role_cfg.get("normalize_targets", True), True
            ),
            "min_role_mass": float(dynamic_role_cfg.get("min_role_mass", 1e-8)),
            "save_role_head": _as_bool(
                dynamic_role_cfg.get("save_role_head", True), True
            ),
        }
        log_event(
            "TRAIN",
            "DYN_ROLE",
            run=run_name,
            roles=role_metadata["num_roles"],
            valid_items=role_metadata["role_valid_item_rows"],
            lambda_role=role_metadata["lambda_role"],
            detach_state=role_metadata["detach_sequence_state"],
        )

    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(int(seed) + 17)

    requires_auxiliary = mode in {"recg", "sage"}
    if requires_auxiliary and not auxiliary_pools:
        raise ValueError(f"mode={mode} requires at least one auxiliary pool")
    auxiliary_sampler = AuxiliaryPoolSampler(
        pools=auxiliary_pools,
        samples_per_domain=int(
            alignment_cfg.get("samples_per_domain_per_step", 128)
        ),
        seed=int(seed) + 29,
    )

    epochs = int(training_cfg.get("epochs", 80))
    train_negatives = int(training_cfg.get("train_negatives", 5))
    patience_limit = int(training_cfg.get("early_stop_patience", 10))
    eval_every = int(training_cfg.get("eval_every", 1))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    max_train_batches = int(training_cfg.get("max_batches_per_epoch", 0))
    max_val_batches = int(evaluation_cfg.get("max_batches", 0))
    ranking = str(evaluation_cfg.get("ranking", "sampled"))
    eval_negatives = int(evaluation_cfg.get("eval_negatives", 100))
    tie_policy = str(evaluation_cfg.get("tie_policy", "worst"))
    ks = tuple(int(k) for k in evaluation_cfg.get("ks", [10, 20]))
    primary_metric = str(evaluation_cfg.get("primary_metric", "NDCG@10"))
    eval_seed = int(seed) + int(
        evaluation_cfg.get("fixed_negative_seed_offset", 10000)
    )
    runtime_cfg = dict(training_cfg.get("runtime", {}) or {})
    show_progress = resolve_progress_enabled(runtime_cfg.get("show_progress", True))
    progress_refresh = float(runtime_cfg.get("progress_refresh_seconds", 0.5))
    leave_progress = _as_bool(runtime_cfg.get("leave_progress_bar", True), True)

    checkpoint_path = Path(checkpoint_path)
    best_metric = -math.inf
    best_epoch = 0
    patience = 0
    history: list[dict[str, float | int]] = []
    clock = RunClock()
    started = time.time()
    log_event(
        "TRAIN",
        "START",
        run=run_name,
        epochs=epochs,
        train_batches=loader_total(train_loader, max_train_batches),
        val_batches=loader_total(val_loader, max_val_batches),
        device=device,
    )

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_started = time.perf_counter()
        model.train()
        if role_head is not None:
            role_head.train()
        running = {
            "loss": 0.0,
            "bpr": 0.0,
            "align": 0.0,
            "role": 0.0,
            "role_weighted": 0.0,
            "role_kl": 0.0,
            "role_target_entropy": 0.0,
            "role_pred_entropy": 0.0,
            "role_pred_confidence": 0.0,
            "role_valid_ratio": 0.0,
            "intra": 0.0,
            "inter": 0.0,
            "sic": 0.0,
            "id": 0.0,
            "omega": 0.0,
            "delta": 0.0,
            "grad_norm": 0.0,
        }
        batches = 0

        train_iterator = progress_iter(
            train_loader,
            total=loader_total(train_loader, max_train_batches),
            run_name=run_name,
            phase="TRAIN",
            detail=f"epoch={epoch:03d}/{epochs:03d}",
            enabled=show_progress,
            refresh_seconds=progress_refresh,
            leave=leave_progress,
        )

        try:
            for step, batch in enumerate(train_iterator, start=1):
                if max_train_batches > 0 and step > max_train_batches:
                    break

                histories = batch["history"].to(device, non_blocking=True)
                positives = batch["target"].to(device, non_blocking=True)
                negatives = _sample_training_negatives(
                    histories=histories,
                    positives=positives,
                    num_items=num_items,
                    num_negatives=train_negatives,
                    generator=train_generator,
                )
                candidates = torch.cat([positives[:, None], negatives], dim=1)

                optimizer.zero_grad(set_to_none=True)
                sequence_state = None
                if role_head is not None:
                    # Reuse the same prefix state for recommendation and role KL so
                    # the auxiliary signal directly shapes the transferable sequence
                    # representation. This avoids a second Transformer forward pass.
                    sequence_state = model.encode_sequence(histories)
                    candidate_state = model.project_items_for_recommendation(candidates)
                    scores = torch.einsum("bd,bcd->bc", sequence_state, candidate_state)
                else:
                    scores = model.score_candidates(histories, candidates)
                _assert_finite("scores", scores, f"epoch={epoch}, step={step}")
                bpr = stable_bpr_loss(scores)
                _assert_finite("bpr", bpr, f"epoch={epoch}, step={step}")

                if requires_auxiliary:
                    projected, raw, domain_labels = _build_alignment_batch(
                        model=model,
                        positives=positives,
                        auxiliary_sampler=auxiliary_sampler,
                        device=device,
                    )
                    _assert_finite(
                        "alignment projected embeddings",
                        projected,
                        f"epoch={epoch}, step={step}",
                    )
                    align_output = compute_alignment_loss(
                        mode=mode,
                        projected=projected.float(),
                        raw=raw.float(),
                        domain_labels=domain_labels,
                        config=alignment_cfg,
                    )
                else:
                    # Differentiable zero; Arch0 still trains its dual architecture
                    # through the recommendation loss.
                    projected = model.recommendation_projection.weight
                    align_output = compute_alignment_loss(
                        mode=mode,
                        projected=projected,
                        raw=projected,
                        domain_labels=torch.zeros(
                            projected.shape[0], dtype=torch.long, device=device
                        ),
                        config=alignment_cfg,
                    )

                _assert_finite(
                    "alignment loss",
                    align_output.total,
                    f"epoch={epoch}, step={step}",
                )

                role_loss = scores.sum() * 0.0
                role_weighted = scores.sum() * 0.0
                role_metrics = {
                    "role": 0.0,
                    "role_kl": 0.0,
                    "role_target_entropy": 0.0,
                    "role_pred_entropy": 0.0,
                    "role_pred_confidence": 0.0,
                    "role_valid_ratio": 0.0,
                }
                if role_head is not None and role_table is not None:
                    if sequence_state is None:
                        sequence_state = model.encode_sequence(histories)
                    role_input = sequence_state.detach() if role_metadata[
                        "detach_sequence_state"
                    ] else sequence_state
                    role_logits = role_head(role_input)
                    role_targets = role_table[positives]
                    role_output = dynamic_role_distillation_loss(
                        logits=role_logits,
                        target_distribution=role_targets,
                        eps=float(role_metadata["min_role_mass"]),
                        normalize_targets=bool(role_metadata["normalize_targets"]),
                    )
                    _assert_finite(
                        "dynamic role distillation loss",
                        role_output.total,
                        f"epoch={epoch}, step={step}",
                    )
                    role_loss = role_output.total
                    role_weighted = float(role_metadata["lambda_role"]) * role_loss
                    role_metrics = role_output.detached_metrics()

                loss = bpr + align_output.total + role_weighted
                _assert_finite("total loss", loss, f"epoch={epoch}, step={step}")
                loss.backward()

                grad_norm = clip_grad_norm_(
                    optimizer_parameters,
                    max_norm=grad_clip,
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch={epoch}, step={step}. "
                        "optimizer.step() was skipped."
                    )

                optimizer.step()

                # Catch a parameter corruption immediately, not one epoch later.
                for parameter_name, parameter in model.named_parameters():
                    if parameter.requires_grad and not torch.isfinite(parameter).all():
                        raise FloatingPointError(
                            f"Parameter {parameter_name} became non-finite at "
                            f"epoch={epoch}, step={step}"
                        )
                if role_head is not None:
                    for parameter_name, parameter in role_head.named_parameters():
                        if parameter.requires_grad and not torch.isfinite(parameter).all():
                            raise FloatingPointError(
                                f"Parameter dynamic_role_head.{parameter_name} became "
                                f"non-finite at epoch={epoch}, step={step}"
                            )

                metrics = align_output.detached_metrics()
                running["loss"] += float(loss.detach().cpu())
                running["bpr"] += float(bpr.detach().cpu())
                running["role_weighted"] += float(role_weighted.detach().cpu())
                for key in (
                    "role",
                    "role_kl",
                    "role_target_entropy",
                    "role_pred_entropy",
                    "role_pred_confidence",
                    "role_valid_ratio",
                ):
                    running[key] += role_metrics[key]
                for key in ("align", "intra", "inter", "sic", "id", "omega", "delta"):
                    running[key] += metrics[key]
                running["grad_norm"] += float(torch.as_tensor(grad_norm).cpu())
                batches += 1
                set_progress_postfix(
                    train_iterator,
                    loss=running["loss"] / batches,
                    bpr=running["bpr"] / batches,
                    align=running["align"] / batches,
                    role=running["role"] / batches,
                    grad=running["grad_norm"] / batches,
                    elapsed=format_seconds(clock.elapsed()),
                )

        finally:
            close_progress(train_iterator)

        if batches == 0:
            raise RuntimeError("Training loader produced no batches")

        train_seconds = float(time.perf_counter() - train_started)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_seconds": train_seconds,
            **{key: value / batches for key, value in running.items()},
        }

        if epoch % eval_every == 0:
            validation_started = time.perf_counter()
            validation = evaluate_bert4rec_family(
                model=model,
                loader=val_loader,
                device=device,
                num_items=num_items,
                ranking=ranking,
                eval_negatives=eval_negatives,
                ks=ks,
                tie_policy=tie_policy,
                seed=eval_seed,
                max_batches=max_val_batches,
                show_progress=show_progress,
                progress_desc=run_name,
                progress_refresh=progress_refresh,
                leave_progress=leave_progress,
            )
            row["validation_seconds"] = float(time.perf_counter() - validation_started)
            row.update({f"val_{key}": value for key, value in validation.items()})
            current = float(validation[primary_metric])
            if not np.isfinite(current):
                raise FloatingPointError(
                    f"Primary validation metric {primary_metric} is non-finite"
                )

            improved = current > best_metric + 1e-12
            if improved:
                best_metric = current
                best_epoch = epoch
                patience = 0
                payload = {
                    "format_version": 1,
                    "run_name": run_name,
                    "mode": mode,
                    "source_domain": source_domain,
                    "seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "best_metric_name": primary_metric,
                    "best_metric_value": float(best_metric),
                    "model_hparams": model.export_hparams(),
                    "model_state": model.state_dict(),
                    "training_cfg": dict(training_cfg),
                    "alignment_cfg": dict(alignment_cfg),
                    "evaluation_cfg": dict(evaluation_cfg),
                    "dynamic_role": dict(role_metadata),
                    "role_head_state": (
                        role_head.state_dict()
                        if role_head is not None and bool(role_metadata.get("save_role_head", True))
                        else None
                    ),
                    "history": history + [row],
                }
                _atomic_torch_save(payload, checkpoint_path)
            else:
                patience += 1

            row["epoch_seconds"] = float(time.perf_counter() - epoch_started)
            row["elapsed_seconds"] = float(clock.elapsed())
            eta_seconds = clock.eta(epoch, epochs)
            row["eta_seconds"] = float(eta_seconds or 0.0)
            log_event(
                "TRAIN",
                "EPOCH",
                run=run_name,
                epoch=f"{epoch:03d}/{epochs:03d}",
                loss=float(row["loss"]),
                bpr=float(row["bpr"]),
                align=float(row["align"]),
                role=float(row["role"]) if dynamic_role_enabled else None,
                validation_metric=f"{primary_metric}:{current:.6f}",
                best=f"{best_metric:.6f}@{best_epoch}",
                status="checkpoint_saved" if improved else f"patience_{patience}/{patience_limit}",
                elapsed=format_seconds(clock.elapsed()),
                eta=format_seconds(eta_seconds),
            )

            history.append(row)
            if patience >= patience_limit:
                log_event("TRAIN", "EARLY_STOP", run=run_name, epoch=epoch, elapsed=format_seconds(clock.elapsed()))
                break
        else:
            row["epoch_seconds"] = float(time.perf_counter() - epoch_started)
            row["elapsed_seconds"] = float(clock.elapsed())
            eta_seconds = clock.eta(epoch, epochs)
            row["eta_seconds"] = float(eta_seconds or 0.0)
            history.append(row)
            log_event(
                "TRAIN",
                "EPOCH",
                run=run_name,
                epoch=f"{epoch:03d}/{epochs:03d}",
                loss=float(row["loss"]),
                status="train_only",
                elapsed=format_seconds(clock.elapsed()),
                eta=format_seconds(eta_seconds),
            )

    if best_epoch == 0 or not checkpoint_path.exists():
        raise RuntimeError("No valid checkpoint was saved")

    return {
        "run_name": run_name,
        "mode": mode,
        "source_domain": source_domain,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_metric_name": primary_metric,
        "best_metric_value": float(best_metric),
        "checkpoint_path": str(checkpoint_path),
        "elapsed_seconds": float(clock.elapsed()),
        "dynamic_role": dict(role_metadata),
        "history": history,
    }
