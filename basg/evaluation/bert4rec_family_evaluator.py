from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from basg.evaluation.base import _rank_with_ties, _sample_negatives
from basg.utils.runtime import (
    RunClock,
    close_progress,
    format_seconds,
    loader_total,
    progress_iter,
    resolve_progress_enabled,
    set_progress_postfix,
)


@torch.no_grad()
def evaluate_bert4rec_family(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    num_items: int,
    ranking: str = "sampled",
    eval_negatives: int = 100,
    ks: tuple[int, ...] = (10, 20),
    tie_policy: str = "worst",
    seed: int = 12026,
    max_batches: int = 0,
    show_progress: bool | None = None,
    progress_desc: str = "Evaluate",
    progress_refresh: float = 0.5,
    leave_progress: bool = False,
) -> dict[str, float]:
    """
    Deterministic source/target evaluator.

    The same seed can be reused at every epoch, so validation candidates do not
    change while early stopping compares checkpoints.
    """
    model.eval()
    rng = np.random.default_rng(int(seed))
    ranks: list[int] = []
    all_equal_count = 0
    tie_case_count = 0
    tie_items_total = 0

    clock = RunClock()
    progress_enabled = resolve_progress_enabled(show_progress)
    iterator = progress_iter(
        loader,
        total=loader_total(loader, max_batches),
        run_name=progress_desc,
        phase="EVAL",
        enabled=progress_enabled,
        refresh_seconds=progress_refresh,
        leave=leave_progress,
    )

    try:
        for batch_index, batch in enumerate(iterator):
            if max_batches > 0 and batch_index >= max_batches:
                break

            histories = batch["history"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            history_lists = histories.detach().cpu().tolist()
            target_list = targets.detach().cpu().tolist()

            if ranking == "full":
                score_tensor = model.score_all_items(histories)
                scores = score_tensor.detach().float().cpu().numpy()
                if not np.isfinite(scores).all():
                    raise FloatingPointError(
                        f"Non-finite validation scores in batch {batch_index}"
                    )

                for row, target in enumerate(target_list):
                    row_scores = scores[row].copy()
                    blocked = {int(x) for x in history_lists[row] if int(x) > 0}
                    for item in blocked:
                        if 1 <= item <= num_items and item != int(target):
                            row_scores[item - 1] = -np.inf
                    target_index = int(target) - 1
                    finite = np.isfinite(row_scores)
                    if not finite[target_index]:
                        raise FloatingPointError("Target score is non-finite")
                    valid_scores = row_scores[finite]
                    target_score = row_scores[target_index]
                    greater = int(np.sum(valid_scores > target_score))
                    equal_others = int(np.sum(valid_scores == target_score)) - 1
                    if equal_others > 0:
                        tie_case_count += 1
                        tie_items_total += equal_others
                    if valid_scores.size > 0 and np.all(valid_scores == valid_scores[0]):
                        all_equal_count += 1
                    if tie_policy == "worst":
                        rank = greater + max(equal_others, 0)
                    elif tie_policy == "best":
                        rank = greater
                    else:
                        rank = int(round(greater + max(equal_others, 0) / 2.0))
                    ranks.append(rank)
                continue

            if ranking != "sampled":
                raise ValueError("ranking must be 'sampled' or 'full'")

            candidates: list[list[int]] = []
            target_indices: list[int] = []
            for history, target in zip(history_lists, target_list):
                blocked = {int(x) for x in history if int(x) > 0}
                blocked.add(int(target))
                negatives = _sample_negatives(
                    num_items=num_items,
                    blocked=blocked,
                    count=int(eval_negatives),
                    rng=rng,
                )
                row = negatives + [int(target)]
                rng.shuffle(row)
                candidates.append(row)
                target_indices.append(row.index(int(target)))

            candidate_tensor = torch.as_tensor(
                candidates, dtype=torch.long, device=device
            )
            score_tensor = model.score_candidates(histories, candidate_tensor)
            scores = score_tensor.detach().float().cpu().numpy()
            if not np.isfinite(scores).all():
                raise FloatingPointError(
                    f"Non-finite validation scores in batch {batch_index}; "
                    "the checkpoint is invalid and must not be saved"
                )

            for row_scores, target_index in zip(scores, target_indices):
                equal_others = int(np.sum(row_scores == row_scores[target_index])) - 1
                if equal_others > 0:
                    tie_case_count += 1
                    tie_items_total += equal_others
                if np.all(row_scores == row_scores[0]):
                    all_equal_count += 1
                ranks.append(_rank_with_ties(row_scores, target_index, tie_policy))
            set_progress_postfix(
                iterator,
                examples=len(ranks),
                ties=tie_case_count,
                elapsed=format_seconds(clock.elapsed()),
            )

    finally:
        close_progress(iterator)

    if not ranks:
        raise RuntimeError("Evaluation produced no examples")

    rank_array = np.asarray(ranks, dtype=np.int64)
    metrics: dict[str, float] = {"num_examples": float(len(ranks))}
    for k in sorted(set(int(x) for x in ks)):
        hit = rank_array < k
        metrics[f"Recall@{k}"] = float(hit.mean())
        metrics[f"NDCG@{k}"] = float(
            np.where(hit, 1.0 / np.log2(rank_array + 2.0), 0.0).mean()
        )
        metrics[f"MRR@{k}"] = float(
            np.where(hit, 1.0 / (rank_array + 1.0), 0.0).mean()
        )

    metrics["all_equal_ratio"] = float(all_equal_count / len(ranks))
    metrics["tie_case_ratio"] = float(tie_case_count / len(ranks))
    metrics["avg_tie_items"] = float(tie_items_total / len(ranks))
    metrics["mean_rank_1based"] = float((rank_array + 1).mean())
    metrics["elapsed_seconds"] = float(clock.elapsed())
    metrics["num_batches"] = float(batch_index + 1 if "batch_index" in locals() else 0)
    return metrics
