from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from mdp_utils import load_config, read_ear_records, setup_logger


PRIMARY_BIOMARKERS = ["CochEH", "VestEH", "PTA_recomputed"]
PRIMARY_LABELS = ["Cochlear hydrops", "Vestibular hydrops", "PTA"]
EXTENDED_BIOMARKERS = ["CochEH", "VestEH", "0.5kHZ", "1kHZ", "2kHZ", "3kHZ"]
EXTENDED_LABELS = ["Cochlear hydrops", "Vestibular hydrops", "0.5 kHz", "1 kHz", "2 kHz", "3 kHz"]


def load_source_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def official_gmm_functions(repo: Path):
    gaussian_module = load_source_module(
        "official_pebm_gaussian",
        repo / "pebm" / "distributions" / "gaussian.py",
    )
    gmm_module = load_source_module(
        "official_pebm_gmm",
        repo / "pebm" / "mixture_model" / "gmm.py",
    )
    Gaussian = gaussian_module.Gaussian
    ParametricMM = gmm_module.ParametricMM

    def fit_all_gmm_models(x: np.ndarray, y: np.ndarray):
        mixtures = []
        for column in range(x.shape[1]):
            present = np.isfinite(x[:, column])
            model = ParametricMM(Gaussian(), Gaussian())
            model.fit(x[present, column], y[present])
            mixtures.append(model)
        return mixtures

    def get_prob_mat(x: np.ndarray, mixtures: list) -> np.ndarray:
        probability = np.ones((x.shape[0], x.shape[1], 2), dtype=float)
        for column, model in enumerate(mixtures):
            present = np.isfinite(x[:, column])
            normal, abnormal = model.pdf(model.theta, x[present, column].reshape(-1, 1))
            normal = np.asarray(normal).reshape(-1)
            abnormal = np.asarray(abnormal).reshape(-1)
            scale = normal + abnormal
            probability[present, column, 0] = np.divide(
                normal,
                scale,
                out=np.full_like(normal, 0.5),
                where=scale > 0,
            )
            probability[present, column, 1] = np.divide(
                abnormal,
                scale,
                out=np.full_like(abnormal, 0.5),
                where=scale > 0,
            )
            clipped = np.clip(probability[present, column, :], 1e-12, 1 - 1e-12)
            probability[present, column, :] = clipped / clipped.sum(axis=1, keepdims=True)
        return probability

    return fit_all_gmm_models, get_prob_mat


def as_float(value: object) -> float:
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def build_paired_cohort(records: list[dict], site: str, biomarkers: list[str]) -> list[dict]:
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["source_site"] == site and not record["is_followup"]:
            by_patient[record["patient_id"]].append(record)
    rows: list[dict] = []
    for patient_id, patient_rows in sorted(by_patient.items()):
        stage_rows = [row for row in patient_rows if row.get("stage（AAO-HNS）") not in (None, "")]
        sides = {row["ear_side"] for row in patient_rows}
        if len(patient_rows) != 2 or sides != {"L", "R"} or len(stage_rows) != 1:
            continue
        complete = True
        patient_output: list[dict] = []
        for row in patient_rows:
            values = [as_float(row.get(name)) for name in biomarkers]
            if not np.isfinite(values).all():
                complete = False
                break
            patient_output.append(
                {
                    "patient_id": patient_id,
                    "ear_id": row["ear_id"],
                    "ear_side": row["ear_side"],
                    "source_subject_id": row["source_subject_id"],
                    "affected_proxy": int(row.get("stage（AAO-HNS）") not in (None, "")),
                    "aao_hns_stage": as_float(row.get("stage（AAO-HNS）")),
                    "pta": as_float(row.get("PTA_recomputed")),
                    "values": values,
                }
            )
        if complete:
            rows.extend(patient_output)
    return rows


def ordered_partitions(items: tuple[int, ...]) -> list[tuple[tuple[int, ...], ...]]:
    partitions: list[tuple[tuple[int, ...], ...]] = []

    def recurse(index: int, blocks: list[list[int]]) -> None:
        if index == len(items):
            partitions.append(tuple(tuple(sorted(block)) for block in blocks))
            return
        value = items[index]
        for block_index in range(len(blocks)):
            next_blocks = [list(block) for block in blocks]
            next_blocks[block_index].append(value)
            recurse(index + 1, next_blocks)
        recurse(index + 1, blocks + [[value]])

    recurse(0, [])
    output: set[tuple[tuple[int, ...], ...]] = set()
    for partition in partitions:
        output.update(itertools.permutations(partition))
    return sorted(output, key=lambda item: (len(item), item))


def fit_exact_order(EventOrder, probability: np.ndarray):
    best = None
    for candidate in ordered_partitions(tuple(range(probability.shape[1]))):
        order = EventOrder(ordering=[list(block) for block in candidate])
        order.score_ordering(probability)
        if best is None or order.score > best.score:
            best = order
    return best


def fit_mcmc(EventOrder, probability: np.ndarray, start, seed: int, iterations: int, burn: int, thin: int):
    np.random.seed(seed)
    random.seed(seed)
    current = EventOrder(ordering=[list(block) for block in start.ordering])
    current.score_ordering(probability)
    best = current
    accepted = 0
    samples: list[tuple[tuple[int, ...], ...]] = []
    for iteration in range(iterations):
        proposal = current.swap_events()
        proposal.score_ordering(probability)
        delta = proposal.score - current.score
        if math.log(max(np.random.random(), 1e-300)) < min(0.0, delta):
            current = proposal
            accepted += 1
        if current.score > best.score:
            best = current
        if iteration >= burn and (iteration - burn) % thin == 0:
            samples.append(tuple(tuple(sorted(block)) for block in current.ordering))
    return best, accepted / iterations, samples


def stage_scores(order, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, likelihood = order.stage_data(probability)
    denominator = likelihood.sum(axis=1, keepdims=True)
    posterior = np.divide(likelihood, denominator, out=np.zeros_like(likelihood), where=denominator > 0)
    stages = np.arange(posterior.shape[1], dtype=float)
    expected = posterior @ stages
    maximum = np.argmax(posterior, axis=1)
    return expected, maximum


def best_youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    finite = np.isfinite(thresholds)
    index = np.argmax((tpr - fpr)[finite])
    return float(thresholds[finite][index])


def metric_bundle(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, score)),
        "accuracy": float(accuracy_score(y, predicted)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else math.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else math.nan,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def clustered_metric_ci(
    rows: list[dict],
    score: np.ndarray,
    threshold: float,
    seed: int,
    replicates: int,
) -> dict[str, tuple[float, float]]:
    patient_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        patient_indices[row["patient_id"]].append(index)
    patient_ids = sorted(patient_indices)
    rng = np.random.default_rng(seed)
    bootstrap_values: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled_patients = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        indices = [index for patient in sampled_patients for index in patient_indices[patient]]
        y_boot = np.asarray([rows[index]["affected_proxy"] for index in indices], dtype=int)
        score_boot = score[indices]
        if len(np.unique(y_boot)) < 2:
            continue
        metrics = metric_bundle(y_boot, score_boot, threshold)
        for name in ("auc", "accuracy", "sensitivity", "specificity"):
            bootstrap_values[name].append(metrics[name])
    return {
        name: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        for name, values in bootstrap_values.items()
    }


def spearman_with_ci(rows: list[dict], score: np.ndarray, seed: int, replicates: int):
    affected = [(index, row) for index, row in enumerate(rows) if row["affected_proxy"] and np.isfinite(row["aao_hns_stage"])]
    x = np.asarray([score[index] for index, _ in affected], dtype=float)
    y = np.asarray([row["aao_hns_stage"] for _, row in affected], dtype=float)
    estimate = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        indices = rng.integers(0, len(x), size=len(x))
        if len(np.unique(x[indices])) < 2 or len(np.unique(y[indices])) < 2:
            continue
        value = spearmanr(x[indices], y[indices]).statistic
        if np.isfinite(value):
            values.append(float(value))
    return estimate, (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))), len(x)


def sequence_probabilities(samples: list[tuple[tuple[int, ...], ...]], labels: list[str]) -> list[list[object]]:
    rows: list[list[object]] = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            before = same = after = 0
            for sample in samples:
                positions = {event: position for position, block in enumerate(sample) for event in block}
                if positions[left] < positions[right]:
                    before += 1
                elif positions[left] == positions[right]:
                    same += 1
                else:
                    after += 1
            total = max(len(samples), 1)
            rows.append([labels[left], labels[right], before / total, same / total, after / total])
    return rows


def patient_cluster_sequence_bootstrap(
    EventOrder,
    fit_all_gmm_models,
    get_prob_mat,
    rows: list[dict],
    labels: list[str],
    seed: int,
    replicates: int,
) -> list[tuple[tuple[int, ...], ...]]:
    patient_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        patient_rows[row["patient_id"]].append(row)
    patient_ids = sorted(patient_rows)
    rng = np.random.default_rng(seed)
    samples: list[tuple[tuple[int, ...], ...]] = []
    for _ in range(replicates):
        selected = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        boot_rows = [row for patient in selected for row in patient_rows[patient]]
        x = np.asarray([row["values"] for row in boot_rows], dtype=float)
        y = np.asarray([row["affected_proxy"] for row in boot_rows], dtype=int)
        mixtures = fit_all_gmm_models(x, y)
        probability = get_prob_mat(x, mixtures)
        best = fit_exact_order(EventOrder, probability)
        samples.append(tuple(tuple(sorted(block)) for block in best.ordering))
    return samples


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def render_order_figure(output: Path, results: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(11, 4.8), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    for axis, result in zip(axes, results):
        blocks = result["best_order_labels"]
        y_positions = np.arange(len(blocks), 0, -1)
        axis.scatter(np.zeros_like(y_positions), y_positions, s=250, color="#1f77b4")
        for y, block in zip(y_positions, blocks):
            axis.text(0.08, y, " + ".join(block), va="center", fontsize=10)
        axis.set_ylim(0.4, len(blocks) + 0.6)
        axis.set_xlim(-0.15, 1.5)
        axis.set_xticks([])
        axis.set_yticks(y_positions, [f"Event stage {index + 1}" for index in range(len(blocks))])
        axis.set_title(result["model_name"])
        axis.spines[["top", "right", "bottom"]].set_visible(False)
    fig.suptitle("LS-trained P-EBM event order (pilot analysis)", fontsize=13)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def render_roc_figure(output: Path, results: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    for result in results:
        y = result["external_y"]
        score = result["external_score"]
        fpr, tpr, _ = roc_curve(y, score)
        axis.plot(fpr, tpr, lw=2, label=f"{result['model_name']} (AUC {result['external_metrics']['auc']:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", lw=1)
    axis.set(xlabel="1 − Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.02))
    axis.legend(loc="lower right", frameon=False)
    axis.set_title("External validation in Z2 paired-ear cohort")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LS-trained clinical P-EBM and frozen Z2 external validation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mcmc", type=int, default=80000)
    parser.add_argument("--burn", type=int, default=20000)
    parser.add_argument("--thin", type=int, default=20)
    parser.add_argument("--metric-bootstrap", type=int, default=2000)
    parser.add_argument("--sequence-bootstrap-primary", type=int, default=300)
    parser.add_argument("--sequence-bootstrap-extended", type=int, default=80)
    args = parser.parse_args()

    config, paths = load_config(args.config)
    logger = setup_logger("clinical_pebm_validation", paths.logs / "23_run_clinical_pebm_validation.log")
    repo = paths.intermediate / "vendor" / "pebm"
    sys.path.insert(0, str(repo))
    from pebm.event_order_pebm.event_order_pebm import EventOrder_pebm
    fit_all_gmm_models, get_prob_mat = official_gmm_functions(repo)

    seeds = [int(value) for value in config.get("RANDOM_SEEDS", [20260713, 20260714, 20260715])]
    records = read_ear_records(paths.clinical_table)
    output = paths.final / "clinical_pebm_external_validation_20260731"
    output.mkdir(parents=True, exist_ok=True)

    specifications = [
        ("Primary: hydrops + PTA", PRIMARY_BIOMARKERS, PRIMARY_LABELS, args.sequence_bootstrap_primary),
        ("Sensitivity: hydrops + four frequencies", EXTENDED_BIOMARKERS, EXTENDED_LABELS, args.sequence_bootstrap_extended),
    ]
    all_results: list[dict] = []
    cohort_summary_rows: list[list[object]] = []
    metric_rows: list[list[object]] = []
    sequence_rows: list[list[object]] = []
    pairwise_rows: list[list[object]] = []
    staging_rows: list[list[object]] = []

    for model_index, (model_name, biomarkers, labels, sequence_bootstrap_n) in enumerate(specifications):
        train_rows = build_paired_cohort(records, "LS", biomarkers)
        external_rows = build_paired_cohort(records, "Z2", biomarkers)
        x_train = np.asarray([row["values"] for row in train_rows], dtype=float)
        y_train = np.asarray([row["affected_proxy"] for row in train_rows], dtype=int)
        x_external = np.asarray([row["values"] for row in external_rows], dtype=float)
        y_external = np.asarray([row["affected_proxy"] for row in external_rows], dtype=int)
        mixtures = fit_all_gmm_models(x_train, y_train)
        train_probability = get_prob_mat(x_train, mixtures)
        external_probability = get_prob_mat(x_external, mixtures)
        exact_best = fit_exact_order(EventOrder_pebm, train_probability)

        mcmc_samples: list[tuple[tuple[int, ...], ...]] = []
        acceptance = []
        mcmc_best = exact_best
        for seed in seeds:
            chain_best, chain_acceptance, chain_samples = fit_mcmc(
                EventOrder_pebm,
                train_probability,
                exact_best,
                seed + model_index * 100,
                args.mcmc,
                args.burn,
                args.thin,
            )
            mcmc_samples.extend(chain_samples)
            acceptance.append(chain_acceptance)
            if chain_best.score > mcmc_best.score:
                mcmc_best = chain_best

        train_score, train_stage = stage_scores(exact_best, train_probability)
        external_score, external_stage = stage_scores(exact_best, external_probability)
        threshold = best_youden_threshold(y_train, train_score)
        internal_metrics = metric_bundle(y_train, train_score, threshold)
        external_metrics = metric_bundle(y_external, external_score, threshold)
        external_ci = clustered_metric_ci(
            external_rows,
            external_score,
            threshold,
            seeds[0] + model_index,
            args.metric_bootstrap,
        )
        rho, rho_ci, rho_n = spearman_with_ci(
            external_rows,
            external_score,
            seeds[1] + model_index,
            args.metric_bootstrap,
        )
        bootstrap_sequences = patient_cluster_sequence_bootstrap(
            EventOrder_pebm,
            fit_all_gmm_models,
            get_prob_mat,
            train_rows,
            labels,
            seeds[2] + model_index,
            sequence_bootstrap_n,
        )

        best_order_labels = [[labels[event] for event in block] for block in exact_best.ordering]
        all_results.append(
            {
                "model_name": model_name,
                "biomarkers": biomarkers,
                "labels": labels,
                "best_order": exact_best.ordering,
                "best_order_labels": best_order_labels,
                "log_likelihood": float(exact_best.score),
                "mcmc_acceptance_mean": float(np.mean(acceptance)),
                "mcmc_saved_samples": len(mcmc_samples),
                "train_rows": len(train_rows),
                "train_patients": len({row["patient_id"] for row in train_rows}),
                "external_rows": len(external_rows),
                "external_patients": len({row["patient_id"] for row in external_rows}),
                "threshold": threshold,
                "internal_metrics": internal_metrics,
                "external_metrics": external_metrics,
                "external_ci": external_ci,
                "external_stage_spearman": rho,
                "external_stage_spearman_ci": rho_ci,
                "external_stage_spearman_n": rho_n,
                "external_y": y_external,
                "external_score": external_score,
            }
        )

        cohort_summary_rows.extend(
            [
                [model_name, "LS_internal_training", len({row["patient_id"] for row in train_rows}), len(train_rows), int(y_train.sum()), int((1 - y_train).sum())],
                [model_name, "Z2_external_validation", len({row["patient_id"] for row in external_rows}), len(external_rows), int(y_external.sum()), int((1 - y_external).sum())],
            ]
        )
        sequence_rows.extend(
            [
                [model_name, index + 1, " + ".join(block), float(exact_best.score)]
                for index, block in enumerate(best_order_labels)
            ]
        )
        for source, samples in (("MCMC_conditional", mcmc_samples), ("patient_cluster_bootstrap", bootstrap_sequences)):
            for row in sequence_probabilities(samples, labels):
                pairwise_rows.append([model_name, source, *row])
        for cohort, metrics, ci in (
            ("LS_internal_apparent", internal_metrics, {}),
            ("Z2_external", external_metrics, external_ci),
        ):
            for metric in ("auc", "accuracy", "sensitivity", "specificity"):
                lower, upper = ci.get(metric, (math.nan, math.nan))
                metric_rows.append([model_name, cohort, metric, metrics[metric], lower, upper, threshold])
        metric_rows.append([model_name, "Z2_external_affected_ears", "spearman_vs_AAO_HNS", rho, rho_ci[0], rho_ci[1], ""])

        for row, score, stage in zip(external_rows, external_score, external_stage):
            staging_rows.append(
                [
                    model_name,
                    "Z2",
                    row["patient_id"],
                    row["ear_side"],
                    row["affected_proxy"],
                    row["aao_hns_stage"],
                    row["pta"],
                    float(score),
                    int(stage),
                ]
            )

        logger.info(
            "%s train=%d ears/%d patients external=%d ears/%d patients order=%s external_auc=%.3f",
            model_name,
            len(train_rows),
            len({row["patient_id"] for row in train_rows}),
            len(external_rows),
            len({row["patient_id"] for row in external_rows}),
            best_order_labels,
            external_metrics["auc"],
        )

    write_csv(
        output / "cohort_summary.csv",
        ["model", "cohort", "patients", "ears", "affected_proxy_ears", "paired_reference_ears"],
        cohort_summary_rows,
    )
    write_csv(
        output / "event_sequence.csv",
        ["model", "event_stage", "events", "training_log_likelihood"],
        sequence_rows,
    )
    write_csv(
        output / "sequence_uncertainty.csv",
        ["model", "uncertainty_source", "event_a", "event_b", "prob_a_before_b", "prob_same_stage", "prob_a_after_b"],
        pairwise_rows,
    )
    write_csv(
        output / "validation_metrics.csv",
        ["model", "cohort", "metric", "estimate", "ci_lower", "ci_upper", "frozen_internal_threshold"],
        metric_rows,
    )
    write_csv(
        output / "external_ear_staging.csv",
        ["model", "site", "patient_id", "ear_side", "affected_proxy", "aao_hns_stage", "pta_db_hl", "expected_pebm_stage", "maximum_likelihood_stage"],
        staging_rows,
    )
    render_order_figure(output / "event_order.png", all_results)
    render_roc_figure(output / "external_roc.png", all_results)

    serializable = []
    for result in all_results:
        clean = {key: value for key, value in result.items() if key not in {"external_y", "external_score"}}
        serializable.append(clean)
    summary = {
        "analysis_status": "completed_pilot_with_explicit_proxy_assumptions",
        "split": "LS model development; Z2 frozen external validation",
        "unit": "ear, with paired ears retained within patient clusters",
        "assumptions": [
            "AAO-HNS stage presence identifies the affected/index-ear proxy in patients with exactly one staged ear.",
            "The paired unstaged ear is a reference proxy, not asserted to be a healthy ear.",
            "CochEH and VestEH are ordinal with 0 normal and higher values more abnormal; hearing thresholds increase with impairment.",
            "Only complete paired cases were used; no imputation, mixture refitting, threshold tuning, or event-order refitting was performed in Z2.",
        ],
        "models": serializable,
        "software": {
            "official_pebm_commit": "ffbe8a969b2947769098f1f4e6099edb32f36b97",
            "random_seeds": seeds,
            "mcmc_iterations_per_seed": args.mcmc,
            "mcmc_burn_in": args.burn,
            "mcmc_thin": args.thin,
            "metric_cluster_bootstrap_replicates": args.metric_bootstrap,
            "primary_sequence_cluster_bootstrap_replicates": args.sequence_bootstrap_primary,
            "extended_sequence_cluster_bootstrap_replicates": args.sequence_bootstrap_extended,
        },
        "limitations": [
            "This is a cross-sectional pseudo-temporal model and does not estimate longitudinal transition probabilities.",
            "The affected/reference definition is a workbook-derived proxy and should be confirmed clinically.",
            "AAO-HNS stage and PTA are related constructs, so their association is concordance rather than fully independent validation.",
            "The 400-ear LS imaging archive is larger than the complete paired clinical P-EBM cohort and is not the P-EBM denominator.",
        ],
    }
    (output / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# 丽水训练—浙二外部验证：临床 P-EBM 试点结果",
        "",
        "本分析按用户确认的中心边界执行：丽水用于模型开发，浙二仅作冻结外部验证。模型单位为耳，但重抽样始终以患者为簇，双耳不按独立患者处理。",
        "",
    ]
    for result in serializable:
        ci = result["external_ci"]
        report_lines.extend(
            [
                f"## {result['model_name']}",
                "",
                f"- 可用队列：丽水 {result['train_patients']} 人/{result['train_rows']} 耳；浙二 {result['external_patients']} 人/{result['external_rows']} 耳。",
                f"- 丽水学习到的事件阶段：{' → '.join(' + '.join(block) for block in result['best_order_labels'])}。",
                f"- 浙二外部区分受累耳代理的 AUC：{result['external_metrics']['auc']:.3f}（患者簇 bootstrap 95% CI {ci['auc'][0]:.3f}–{ci['auc'][1]:.3f}）。",
                f"- 冻结丽水阈值后，浙二敏感度 {result['external_metrics']['sensitivity']:.3f}（95% CI {ci['sensitivity'][0]:.3f}–{ci['sensitivity'][1]:.3f}），特异度 {result['external_metrics']['specificity']:.3f}（95% CI {ci['specificity'][0]:.3f}–{ci['specificity'][1]:.3f}）。",
                f"- 浙二受累耳中，P-EBM期望分期与 AAO-HNS 分期的 Spearman ρ={result['external_stage_spearman']:.3f}（95% CI {result['external_stage_spearman_ci'][0]:.3f}–{result['external_stage_spearman_ci'][1]:.3f}；n={result['external_stage_spearman_n']}）。",
                "",
            ]
        )
    report_lines.extend(
        [
            "## 解释边界",
            "",
            "- 这是横断面 pseudo-temporal P-EBM，不代表真实自然病程、因果顺序或纵向状态转移概率。",
            "- “受累耳”由唯一存在 AAO-HNS 分期的耳侧定义；配对另一耳仅是参考代理，不能直接称为健康耳。",
            "- 400耳是丽水影像/分割训练库的分母；临床 P-EBM 仅能使用具有完整成对临床指标的子集。",
            "- 浙二数据未参与混合分布、事件顺序或阈值拟合。",
            "",
        ]
    )
    (output / "RESULTS_SUMMARY.md").write_text("\n".join(report_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
