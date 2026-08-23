from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EVENT_SPECS = (
    ("CochEH_worse_ear", "Cochlear hydrops", 0.0),
    ("VestEH_worse_ear", "Vestibular hydrops", 0.0),
    ("PTA_worse_ear_db", "PTA >25 dB", 25.0),
    ("DHI_T", "DHI-T >0", 0.0),
    ("THI", "THI >0", 0.0),
    ("Ear_fullness", "Ear fullness >0", 0.0),
    ("VADL", "VADL >28", 28.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run patient-level P-EBM in confirmed Z2 MD baseline patients.")
    parser.add_argument(
        "--sheet3-json",
        type=Path,
        default=Path("results_md_progression/final/study_design_corrected_20260801/audit/sheet3_deidentified.json"),
    )
    parser.add_argument(
        "--pebm-repository",
        type=Path,
        default=Path("results_md_progression/intermediate/vendor/pebm"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_md_progression/final/patient_level_md_pebm_20260801"),
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--mcmc-iterations", type=int, default=60000)
    parser.add_argument("--mcmc-burn", type=int, default=10000)
    parser.add_argument("--mcmc-thin", type=int, default=10)
    return parser.parse_args()


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def as_float(value: object) -> float:
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def first_present(rows: list[dict[str, object]], key: str) -> float:
    for row in rows:
        value = as_float(row.get(key))
        if np.isfinite(value):
            return value
    return math.nan


def max_present(rows: list[dict[str, object]], key: str) -> float:
    values = [as_float(row.get(key)) for row in rows]
    values = [value for value in values if np.isfinite(value)]
    return max(values) if values else math.nan


def build_patient_cohort(payload: dict[str, object]) -> list[dict[str, object]]:
    headers = payload["headers"]
    ear_rows = [dict(zip(headers, row, strict=True)) for row in payload["rows"]]
    by_visit: dict[str, list[dict[str, object]]] = {}
    for row in ear_rows:
        visit_id = str(row["ID"])
        by_visit.setdefault(visit_id, []).append(row)
    patients: list[dict[str, object]] = []
    for visit_id, rows in sorted(by_visit.items()):
        lowered = visit_id.lower()
        is_followup = lowered.endswith("_6m") or lowered.endswith("-6m") or lowered.endswith("_3m") or lowered.endswith("-3m")
        if is_followup or "-NC" in visit_id.upper() or "疑似" in visit_id:
            continue
        if len(rows) != 2 or {str(row["side"]) for row in rows} != {"L", "R"}:
            continue
        patients.append(
            {
                "patient_id": visit_id,
                "age_years": first_present(rows, "age"),
                "sex_code": first_present(rows, "sex"),
                "CochEH_worse_ear": max_present(rows, "CochEH"),
                "VestEH_worse_ear": max_present(rows, "VestEH"),
                "PTA_worse_ear_db": max_present(rows, "PTA"),
                "DHI_T": first_present(rows, "DHI-T"),
                "THI": first_present(rows, "THI"),
                "Ear_fullness": first_present(rows, "耳闷"),
                "VADL": first_present(rows, "VADL"),
            }
        )
    return patients


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
    return sorted(output, key=lambda value: (len(value), value))


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
    stage_values = np.arange(posterior.shape[1], dtype=float)
    return posterior @ stage_values, np.argmax(posterior, axis=1)


def fit_mixtures(ParametricMM, Gaussian, values: np.ndarray, thresholds: list[float]):
    models = []
    mixture_rows: list[dict[str, object]] = []
    for column, (_, label, threshold) in enumerate(EVENT_SPECS):
        present = np.isfinite(values[:, column])
        x = values[present, column]
        y = (x > thresholds[column]).astype(int)
        if len(np.unique(y)) != 2:
            raise RuntimeError(f"{label} lacks both normal and abnormal initialization groups")
        unique_values = np.unique(x)
        positive_steps = np.diff(unique_values)
        positive_steps = positive_steps[positive_steps > 0]
        scale_floor = float(0.5 * positive_steps.min()) if positive_steps.size else 0.5

        class ScaleFlooredGaussian(Gaussian):
            """Official Gaussian component with a deterministic measurement-resolution scale floor."""

            def get_bounds(self, X_mix, X_comp, event_sign):
                bounds = list(super().get_bounds(X_mix, X_comp, event_sign))
                sigma_lower, sigma_upper = bounds[1]
                bounds[1] = (max(float(sigma_lower), scale_floor), max(float(sigma_upper), scale_floor))
                return bounds

            def estimate_params(self, X_comp):
                mean, sigma = super().estimate_params(X_comp)
                return [mean, max(float(sigma), scale_floor)]

        model = ParametricMM(ScaleFlooredGaussian(), ScaleFlooredGaussian())
        theta = model.fit(x, y)
        if not np.isfinite(theta).all() or theta[1] <= 0 or theta[3] <= 0:
            raise RuntimeError(f"{label} produced an invalid Gaussian mixture: {theta}")
        models.append(model)
        mixture_rows.append(
            {
                "event_index": column + 1,
                "event": label,
                "initialization_threshold": threshold,
                "available_patients": int(present.sum()),
                "initial_normal_n": int((y == 0).sum()),
                "initial_abnormal_n": int((y == 1).sum()),
                "measurement_resolution_scale_floor": scale_floor,
                "normal_mean": float(theta[0]),
                "normal_sd": float(theta[1]),
                "abnormal_mean": float(theta[2]),
                "abnormal_sd": float(theta[3]),
                "fitted_normal_mixture_fraction": float(theta[4]),
            }
        )
    return models, mixture_rows


def probability_matrix(values: np.ndarray, models: list[object]) -> np.ndarray:
    probability = np.ones((values.shape[0], values.shape[1], 2), dtype=float)
    for column, model in enumerate(models):
        present = np.isfinite(values[:, column])
        with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
            normal, abnormal = model.pdf(model.theta, values[present, column].reshape(-1, 1))
        normal = np.asarray(normal).reshape(-1)
        abnormal = np.asarray(abnormal).reshape(-1)
        total = normal + abnormal
        probability[present, column, 0] = np.divide(normal, total, out=np.full_like(normal, 0.5), where=total > 0)
        probability[present, column, 1] = np.divide(abnormal, total, out=np.full_like(abnormal, 0.5), where=total > 0)
        clipped = np.clip(probability[present, column], 1e-12, 1 - 1e-12)
        probability[present, column] = clipped / clipped.sum(axis=1, keepdims=True)
    if not np.isfinite(probability).all():
        raise RuntimeError("Non-finite event probabilities remain after mixture fitting")
    return probability


def event_stage_map(ordering: tuple[tuple[int, ...], ...] | list[list[int]]) -> dict[int, int]:
    return {event: stage_index for stage_index, block in enumerate(ordering, start=1) for event in block}


def uncertainty_rows(samples: list[tuple[tuple[int, ...], ...]], labels: list[str]) -> list[dict[str, object]]:
    positions = {index: [] for index in range(len(labels))}
    for sample in samples:
        mapping = event_stage_map(sample)
        for event, stage in mapping.items():
            positions[event].append(stage)
    rows: list[dict[str, object]] = []
    for first, second in itertools.combinations(range(len(labels)), 2):
        before = same = after = 0
        for sample in samples:
            mapping = event_stage_map(sample)
            if mapping[first] < mapping[second]:
                before += 1
            elif mapping[first] == mapping[second]:
                same += 1
            else:
                after += 1
        denominator = max(len(samples), 1)
        rows.append(
            {
                "event_a": labels[first],
                "event_b": labels[second],
                "prob_a_before_b": before / denominator,
                "prob_same_stage": same / denominator,
                "prob_a_after_b": after / denominator,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(output_dir: Path, best_order, labels: list[str], samples):
    blocks = [[labels[event] for event in block] for block in best_order.ordering]
    figure, axis = plt.subplots(figsize=(9, max(3.5, 0.8 * len(blocks))), dpi=180)
    y = np.arange(len(blocks))
    axis.barh(y, np.ones(len(blocks)), color="#31688E")
    axis.set_yticks(y, [" + ".join(block) for block in blocks])
    axis.set_xticks([])
    axis.invert_yaxis()
    axis.set_title("Patient-level MD P-EBM event sequence\nAAO-HNS stage excluded")
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    figure.savefig(output_dir / "event_sequence.png", bbox_inches="tight")
    plt.close(figure)

    max_stage = len(labels)
    matrix = np.zeros((len(labels), max_stage), dtype=float)
    for sample in samples:
        mapping = event_stage_map(sample)
        for event, stage in mapping.items():
            matrix[event, stage - 1] += 1
    if samples:
        matrix /= len(samples)
    figure, axis = plt.subplots(figsize=(9, 5), dpi=180)
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=max(matrix.max(), 1e-6))
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xticks(np.arange(max_stage), np.arange(1, max_stage + 1))
    axis.set_xlabel("Event stage")
    axis.set_title("MCMC positional uncertainty")
    figure.colorbar(image, ax=axis, label="Probability")
    figure.tight_layout()
    figure.savefig(output_dir / "event_positional_uncertainty.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.sheet3_json.read_text(encoding="utf-8"))
    patients = build_patient_cohort(payload)
    if len(patients) != 96:
        raise RuntimeError(f"Expected 96 confirmed baseline MD patients, observed {len(patients)}")

    gaussian_module = load_module("patient_level_pebm_gaussian", args.pebm_repository / "pebm/distributions/gaussian.py")
    gmm_module = load_module("patient_level_pebm_gmm", args.pebm_repository / "pebm/mixture_model/gmm.py")
    sys.path.insert(0, str(args.pebm_repository.resolve()))
    from pebm.event_order_pebm.event_order_pebm import EventOrder_pebm

    feature_names = [spec[0] for spec in EVENT_SPECS]
    labels = [spec[1] for spec in EVENT_SPECS]
    thresholds = [spec[2] for spec in EVENT_SPECS]
    values = np.asarray([[as_float(patient[name]) for name in feature_names] for patient in patients], dtype=float)
    models, mixture_rows = fit_mixtures(gmm_module.ParametricMM, gaussian_module.Gaussian, values, thresholds)
    probability = probability_matrix(values, models)
    exact_best = fit_exact_order(EventOrder_pebm, probability)
    mcmc_best, acceptance, samples = fit_mcmc(
        EventOrder_pebm,
        probability,
        exact_best,
        seed=args.seed,
        iterations=args.mcmc_iterations,
        burn=args.mcmc_burn,
        thin=args.mcmc_thin,
    )
    best_order = mcmc_best if mcmc_best.score > exact_best.score else exact_best
    expected_stage, maximum_stage = stage_scores(best_order, probability)
    order_rows = [
        {
            "event_stage": stage_index,
            "events": " + ".join(labels[event] for event in block),
            "event_indices": ",".join(str(event + 1) for event in block),
            "log_likelihood": float(best_order.score),
        }
        for stage_index, block in enumerate(best_order.ordering, start=1)
    ]
    patient_rows = []
    for patient, expected, maximum in zip(patients, expected_stage, maximum_stage, strict=True):
        patient_rows.append(
            {
                **patient,
                "pebm_expected_stage": float(expected),
                "pebm_maximum_likelihood_stage": int(maximum),
                "AAO_HNS_stage_used": False,
            }
        )
    pairwise_rows = uncertainty_rows(samples, labels)
    write_csv(args.output_dir / "patient_level_cohort_and_pebm_stage.csv", patient_rows)
    write_csv(args.output_dir / "event_order.csv", order_rows)
    write_csv(args.output_dir / "mixture_parameters.csv", mixture_rows)
    write_csv(args.output_dir / "pairwise_event_uncertainty.csv", pairwise_rows)
    make_figures(args.output_dir, best_order, labels, samples)

    summary = {
        "analysis_unit": "one baseline visit per confirmed MD patient",
        "patients": len(patients),
        "excluded": {"followup_visits": 3, "explicit_NC_controls": 3, "suspected_cases": 1},
        "AAO_HNS_stage": "excluded from all P-EBM event inputs, mixture initialization, affected-ear definition, and model evaluation",
        "bilateral_aggregation": "maximum/worse-ear for CochEH, VestEH, and PTA; patient-level first nonmissing value for DHI-T, THI, ear fullness, and VADL",
        "event_initialization": "biomarker-specific clinical thresholds initialize official two-Gaussian mixtures; event ordering is then likelihood-estimated",
        "numerical_stabilization": (
            "Gaussian standard deviations are floored at half the smallest observed measurement increment for each biomarker; "
            "this prevents zero-variance threshold-defined normal groups without altering source observations."
        ),
        "event_order": [[labels[event] for event in block] for block in best_order.ordering],
        "log_likelihood": float(best_order.score),
        "mcmc": {
            "iterations": args.mcmc_iterations,
            "burn": args.mcmc_burn,
            "thin": args.mcmc_thin,
            "retained_samples": len(samples),
            "acceptance_rate": acceptance,
            "seed": args.seed,
        },
        "missing_counts": {feature: int(np.isnan(values[:, index]).sum()) for index, feature in enumerate(feature_names)},
        "interpretation_boundary": (
            "Cross-sectional pseudo-temporal ordering only. It does not estimate longitudinal transition rates or prove causality. "
            "Imaging morphometry is not included until external masks pass manual review."
        ),
    }
    (args.output_dir / "pebm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = f"""# 浙二MD患者级P-EBM（2026-08-01）

- 主要队列：96名确认MD基线患者；排除3次随访、3个`-NC`和1个“疑似”。
- 分析单位：患者，不把左右耳视为独立样本。
- AAO-HNS stage：完全不参与P-EBM事件、混合分布初始化、患耳定义或模型评价。
- 事件：Cochlear hydrops、Vestibular hydrops、PTA>25 dB、DHI-T>0、THI>0、耳闷>0、VADL>28。
- 数值稳定：高斯标准差下限设为各指标最小观测步长的一半；不改动原始观测值。
- 事件顺序：{' -> '.join(' + '.join(block) for block in summary['event_order'])}。
- 解释限制：这是横断面伪时间顺序，不是纵向疾病转移率；外部自动mask未人工复核前不加入形态事件。
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print("PATIENT_LEVEL_PEBM_COMPLETE", json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
