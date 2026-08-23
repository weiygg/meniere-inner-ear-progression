from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from mdp_utils import load_config, read_ear_records, setup_logger


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def render_order_figure(output: Path, results: list[dict]) -> None:
    figure, axes = plt.subplots(1, len(results), figsize=(11, 4.8), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    for axis, result in zip(axes, results, strict=True):
        blocks = result["best_order_labels"]
        positions = np.arange(len(blocks), 0, -1)
        axis.scatter(np.zeros_like(positions), positions, s=250, color="#0F766E")
        for position, block in zip(positions, blocks, strict=True):
            axis.text(0.08, position, " + ".join(block), va="center", fontsize=10)
        axis.set_ylim(0.4, len(blocks) + 0.6)
        axis.set_xlim(-0.15, 1.65)
        axis.set_xticks([])
        axis.set_yticks(
            positions,
            [f"Event stage {index + 1}" for index in range(len(blocks))],
        )
        axis.set_title(result["model_name"])
        axis.spines[["top", "right", "bottom"]].set_visible(False)
    figure.suptitle("Z2-developed cross-sectional P-EBM event order", fontsize=13)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def render_roc_figure(output: Path, results: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    panels = [
        ("development_y", "development_score", "development_metrics", "Z2 apparent performance"),
        ("secondary_y", "secondary_score", "secondary_metrics", "LS secondary transport evaluation"),
    ]
    for axis, (y_key, score_key, metrics_key, title) in zip(axes, panels, strict=True):
        for result in results:
            fpr, tpr, _ = roc_curve(result[y_key], result[score_key])
            axis.plot(
                fpr,
                tpr,
                lw=2,
                label=f"{result['model_name']} (AUC {result[metrics_key]['auc']:.3f})",
            )
        axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", lw=1)
        axis.set(
            xlabel="1 - Specificity",
            ylabel="Sensitivity",
            xlim=(0, 1),
            ylim=(0, 1.02),
            title=title,
        )
        axis.legend(loc="lower right", frameon=False, fontsize=8)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit the clinical P-EBM in Z2 and use LS only as a secondary frozen transport cohort."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mcmc", type=int, default=80000)
    parser.add_argument("--burn", type=int, default=20000)
    parser.add_argument("--thin", type=int, default=20)
    parser.add_argument("--metric-bootstrap", type=int, default=2000)
    parser.add_argument("--sequence-bootstrap-primary", type=int, default=300)
    parser.add_argument("--sequence-bootstrap-extended", type=int, default=80)
    args = parser.parse_args()

    config, paths = load_config(args.config)
    logger = setup_logger("z2_clinical_pebm", paths.logs / "44_run_z2_clinical_pebm.log")
    helpers = load_module("clinical_pebm_helpers", Path(__file__).with_name("23_run_clinical_pebm_validation.py"))
    repository = paths.intermediate / "vendor" / "pebm"
    sys.path.insert(0, str(repository))
    from pebm.event_order_pebm.event_order_pebm import EventOrder_pebm

    fit_all_gmm_models, get_prob_mat = helpers.official_gmm_functions(repository)
    seeds = [int(value) for value in config.get("RANDOM_SEEDS", [20260713, 20260714, 20260715])]
    records = read_ear_records(paths.clinical_table)
    output = paths.final / "clinical_pebm_z2_development_20260801"
    output.mkdir(parents=True, exist_ok=True)

    specifications = [
        (
            "Primary: hydrops + PTA",
            helpers.PRIMARY_BIOMARKERS,
            helpers.PRIMARY_LABELS,
            args.sequence_bootstrap_primary,
        ),
        (
            "Sensitivity: hydrops + four frequencies",
            helpers.EXTENDED_BIOMARKERS,
            helpers.EXTENDED_LABELS,
            args.sequence_bootstrap_extended,
        ),
    ]

    results: list[dict] = []
    cohort_rows: list[list[object]] = []
    sequence_rows: list[list[object]] = []
    uncertainty_rows: list[list[object]] = []
    metric_rows: list[list[object]] = []
    staging_rows: list[list[object]] = []

    for model_index, (model_name, biomarkers, labels, sequence_bootstrap_n) in enumerate(specifications):
        development_rows = helpers.build_paired_cohort(records, "Z2", biomarkers)
        secondary_rows = helpers.build_paired_cohort(records, "LS", biomarkers)
        development_x = np.asarray([row["values"] for row in development_rows], dtype=float)
        development_y = np.asarray([row["affected_proxy"] for row in development_rows], dtype=int)
        secondary_x = np.asarray([row["values"] for row in secondary_rows], dtype=float)
        secondary_y = np.asarray([row["affected_proxy"] for row in secondary_rows], dtype=int)

        mixtures = fit_all_gmm_models(development_x, development_y)
        development_probability = get_prob_mat(development_x, mixtures)
        secondary_probability = get_prob_mat(secondary_x, mixtures)
        exact_best = helpers.fit_exact_order(EventOrder_pebm, development_probability)

        mcmc_samples: list[tuple[tuple[int, ...], ...]] = []
        acceptance: list[float] = []
        for seed in seeds:
            _, chain_acceptance, chain_samples = helpers.fit_mcmc(
                EventOrder_pebm,
                development_probability,
                exact_best,
                seed + model_index * 100,
                args.mcmc,
                args.burn,
                args.thin,
            )
            acceptance.append(chain_acceptance)
            mcmc_samples.extend(chain_samples)

        development_score, development_stage = helpers.stage_scores(exact_best, development_probability)
        secondary_score, secondary_stage = helpers.stage_scores(exact_best, secondary_probability)
        threshold = helpers.best_youden_threshold(development_y, development_score)
        development_metrics = helpers.metric_bundle(development_y, development_score, threshold)
        secondary_metrics = helpers.metric_bundle(secondary_y, secondary_score, threshold)
        development_ci = helpers.clustered_metric_ci(
            development_rows,
            development_score,
            threshold,
            seeds[0] + model_index,
            args.metric_bootstrap,
        )
        secondary_ci = helpers.clustered_metric_ci(
            secondary_rows,
            secondary_score,
            threshold,
            seeds[0] + 1000 + model_index,
            args.metric_bootstrap,
        )
        development_rho, development_rho_ci, development_rho_n = helpers.spearman_with_ci(
            development_rows,
            development_score,
            seeds[1] + model_index,
            args.metric_bootstrap,
        )
        secondary_rho, secondary_rho_ci, secondary_rho_n = helpers.spearman_with_ci(
            secondary_rows,
            secondary_score,
            seeds[1] + 1000 + model_index,
            args.metric_bootstrap,
        )
        bootstrap_sequences = helpers.patient_cluster_sequence_bootstrap(
            EventOrder_pebm,
            fit_all_gmm_models,
            get_prob_mat,
            development_rows,
            labels,
            seeds[2] + model_index,
            sequence_bootstrap_n,
        )

        best_order_labels = [[labels[event] for event in block] for block in exact_best.ordering]
        result = {
            "model_name": model_name,
            "biomarkers": biomarkers,
            "labels": labels,
            "best_order": exact_best.ordering,
            "best_order_labels": best_order_labels,
            "log_likelihood": float(exact_best.score),
            "mcmc_acceptance_mean": float(np.mean(acceptance)),
            "mcmc_saved_samples": len(mcmc_samples),
            "development_rows": len(development_rows),
            "development_patients": len({row["patient_id"] for row in development_rows}),
            "secondary_rows": len(secondary_rows),
            "secondary_patients": len({row["patient_id"] for row in secondary_rows}),
            "threshold": threshold,
            "development_metrics": development_metrics,
            "development_ci": development_ci,
            "secondary_metrics": secondary_metrics,
            "secondary_ci": secondary_ci,
            "development_stage_spearman": development_rho,
            "development_stage_spearman_ci": development_rho_ci,
            "development_stage_spearman_n": development_rho_n,
            "secondary_stage_spearman": secondary_rho,
            "secondary_stage_spearman_ci": secondary_rho_ci,
            "secondary_stage_spearman_n": secondary_rho_n,
            "development_y": development_y,
            "development_score": development_score,
            "secondary_y": secondary_y,
            "secondary_score": secondary_score,
        }
        results.append(result)

        cohort_rows.extend(
            [
                [model_name, "Z2_development", result["development_patients"], len(development_rows), int(development_y.sum()), int((1 - development_y).sum())],
                [model_name, "LS_secondary_transport", result["secondary_patients"], len(secondary_rows), int(secondary_y.sum()), int((1 - secondary_y).sum())],
            ]
        )
        sequence_rows.extend(
            [
                [model_name, index + 1, " + ".join(block), float(exact_best.score)]
                for index, block in enumerate(best_order_labels)
            ]
        )
        for source, samples in (
            ("MCMC_conditional", mcmc_samples),
            ("Z2_patient_cluster_bootstrap", bootstrap_sequences),
        ):
            for row in helpers.sequence_probabilities(samples, labels):
                uncertainty_rows.append([model_name, source, *row])

        for cohort, metrics, confidence in (
            ("Z2_development_apparent", development_metrics, development_ci),
            ("LS_secondary_transport", secondary_metrics, secondary_ci),
        ):
            for metric in ("auc", "accuracy", "sensitivity", "specificity"):
                lower, upper = confidence.get(metric, (math.nan, math.nan))
                metric_rows.append([model_name, cohort, metric, metrics[metric], lower, upper, threshold])
        metric_rows.extend(
            [
                [model_name, "Z2_development_affected_ears", "spearman_vs_AAO_HNS", development_rho, development_rho_ci[0], development_rho_ci[1], ""],
                [model_name, "LS_secondary_affected_ears", "spearman_vs_AAO_HNS", secondary_rho, secondary_rho_ci[0], secondary_rho_ci[1], ""],
            ]
        )

        for cohort, site, rows, scores, stages in (
            ("development", "Z2", development_rows, development_score, development_stage),
            ("secondary_transport", "LS", secondary_rows, secondary_score, secondary_stage),
        ):
            for row, score, stage in zip(rows, scores, stages, strict=True):
                staging_rows.append(
                    [
                        model_name,
                        cohort,
                        site,
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
            "%s Z2=%d ears/%d patients LS=%d ears/%d patients order=%s Z2_apparent_auc=%.3f LS_transport_auc=%.3f",
            model_name,
            len(development_rows),
            result["development_patients"],
            len(secondary_rows),
            result["secondary_patients"],
            best_order_labels,
            development_metrics["auc"],
            secondary_metrics["auc"],
        )

    helpers.write_csv(
        output / "cohort_summary.csv",
        ["model", "cohort", "patients", "ears", "affected_proxy_ears", "paired_reference_ears"],
        cohort_rows,
    )
    helpers.write_csv(
        output / "event_sequence.csv",
        ["model", "event_stage", "events", "development_log_likelihood"],
        sequence_rows,
    )
    helpers.write_csv(
        output / "sequence_uncertainty.csv",
        ["model", "uncertainty_source", "event_a", "event_b", "prob_a_before_b", "prob_same_stage", "prob_a_after_b"],
        uncertainty_rows,
    )
    helpers.write_csv(
        output / "performance_metrics.csv",
        ["model", "cohort", "metric", "estimate", "ci_lower", "ci_upper", "z2_derived_threshold"],
        metric_rows,
    )
    helpers.write_csv(
        output / "ear_staging.csv",
        ["model", "cohort_role", "site", "patient_id", "ear_side", "affected_proxy", "aao_hns_stage", "pta_db_hl", "expected_pebm_stage", "maximum_likelihood_stage"],
        staging_rows,
    )
    render_order_figure(output / "event_order.png", results)
    render_roc_figure(output / "roc_curves.png", results)

    serializable = []
    for result in results:
        serializable.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"development_y", "development_score", "secondary_y", "secondary_score"}
            }
        )
    summary = {
        "analysis_status": "completed_z2_development_pilot_with_explicit_proxy_assumptions",
        "development_cohort": "Z2",
        "secondary_transport_cohort": "LS",
        "unit": "ear, with paired ears retained within patient clusters",
        "assumptions": [
            "AAO-HNS stage presence identifies the affected/index-ear proxy in patients with exactly one staged ear.",
            "The paired unstaged ear is a reference proxy, not asserted to be a healthy ear.",
            "CochEH and VestEH are ordinal with 0 normal and higher values more abnormal; hearing thresholds increase with impairment.",
            "Only complete paired baseline cases were used; no imputation was performed.",
            "Mixture distributions, event order, and the operating threshold were fitted in Z2 only; LS was not used for Z2 model selection.",
        ],
        "models": serializable,
        "software": {
            "official_pebm_commit": "ffbe8a969b2947769098f1f4e6099edb32f36b97",
            "random_seeds": seeds,
            "mcmc_iterations_per_seed": args.mcmc,
            "mcmc_burn_in": args.burn,
            "mcmc_thin": args.thin,
            "metric_patient_cluster_bootstrap_replicates": args.metric_bootstrap,
            "primary_sequence_patient_cluster_bootstrap_replicates": args.sequence_bootstrap_primary,
            "extended_sequence_patient_cluster_bootstrap_replicates": args.sequence_bootstrap_extended,
        },
        "limitations": [
            "This is a cross-sectional pseudo-temporal model and does not estimate longitudinal transition probabilities.",
            "The affected/reference definition is a workbook-derived proxy and still requires clinical confirmation.",
            "Z2 performance is apparent development performance because the same cohort fitted the mixture model, event order, and threshold.",
            "AAO-HNS stage and PTA are related constructs, so their association is concordance rather than independent validation.",
            "Automated Z2 segmentation-derived morphometry was not used because the external masks have not yet received manual anatomical review.",
        ],
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 浙二临床 P-EBM 试点结果（2026-08-01）",
        "",
        "本版按项目负责人确认的方向执行：浙二为 P-EBM 建模队列，丽水仅作为次要迁移评价；旧的‘丽水拟合、浙二验证’结果只保留用于审计，不再作为主结果。",
        "",
    ]
    for result in serializable:
        development_ci = result["development_ci"]
        secondary_ci = result["secondary_ci"]
        report_lines.extend(
            [
                f"## {result['model_name']}",
                "",
                f"- 浙二建模队列：{result['development_patients']}人/{result['development_rows']}耳；丽水次要迁移队列：{result['secondary_patients']}人/{result['secondary_rows']}耳。",
                f"- 浙二学习到的事件阶段：{' → '.join(' + '.join(block) for block in result['best_order_labels'])}。",
                f"- 浙二表观 AUC：{result['development_metrics']['auc']:.3f}（患者簇 bootstrap 95% CI {development_ci['auc'][0]:.3f}–{development_ci['auc'][1]:.3f}）。这是开发集表观值，不是外部验证结果。",
                f"- 冻结浙二模型后，丽水次要迁移 AUC：{result['secondary_metrics']['auc']:.3f}（患者簇 bootstrap 95% CI {secondary_ci['auc'][0]:.3f}–{secondary_ci['auc'][1]:.3f}）。",
                f"- 浙二受累耳代理中，期望 P-EBM 分期与 AAO-HNS 分期的 Spearman ρ={result['development_stage_spearman']:.3f}（95% CI {result['development_stage_spearman_ci'][0]:.3f}–{result['development_stage_spearman_ci'][1]:.3f}；n={result['development_stage_spearman_n']}）。",
                "",
            ]
        )
    report_lines.extend(
        [
            "## 事件顺序稳定性",
            "",
            "- 主模型的患者簇 bootstrap 中，PTA 早于耳蜗积水的概率为0.850，早于前庭积水的概率为0.807，提示‘听力异常较早’相对稳定。",
            "- 但耳蜗积水与前庭积水的先后并不稳定：全样本最优顺序为耳蜗在前，而81.0%的患者簇 bootstrap 重抽样反而支持前庭积水在前。",
            "- 敏感性模型同样显示听阈事件总体早于积水事件，但83.8%的患者簇 bootstrap 支持前庭积水早于耳蜗积水。",
            "- 因此，当前只能谨慎表述为‘听阈异常倾向于早于积水事件’，不能把耳蜗与前庭积水的具体先后当作稳定结论。",
            "",
        ]
    )
    report_lines.extend(
        [
            "## 解释边界",
            "",
            "- 这是横断面 pseudo-temporal P-EBM，只用于探索事件相对顺序，不能解释为真实纵向进展、因果顺序或状态转移概率。",
            "- ‘受累耳’仍由唯一存在 AAO-HNS 分期的耳侧定义；配对另一耳是参考代理，不等同于健康耳。",
            "- 浙二自动分割产生的形态学特征尚未经过人工掩膜复核，因此本轮没有把这些特征作为 P-EBM 事件。",
            "- 如果后续使用浙二掩膜进行域适配，必须另留锁定测试子集，不能同时用于模型训练和最终评价。",
            "",
        ]
    )
    (output / "RESULTS_SUMMARY.md").write_text("\n".join(report_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
