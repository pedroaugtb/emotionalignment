#!/usr/bin/env python3
"""Rebuild the audited appendix and aggregate result ledger.

This script performs no model inference and reads only the aggregate CSV tables
included with this anonymous artifact. Paths are relative to this directory.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT


def read_csv(rel: str) -> list[dict[str, str]]:
    with (ROOT / rel).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: str | float | int | None, digits: int = 3) -> str:
    if value in (None, ""):
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "--"
    if number != 0 and abs(number) < 0.001:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def esc(text: str) -> str:
    for old, new in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def table(caption: str, label: str, columns: str, header: str, rows: list[str], *, star: bool = False,
          size: str = r"\small", resize: bool = False) -> str:
    env = "table*" if star else "table"
    body = "\n".join(rows)
    tab = f"\\begin{{tabular}}{{{columns}}}\n\\toprule\n{header} \\\\\n+\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}"
    tab = tab.replace("\n+\\midrule", "\n\\midrule")
    if resize:
        width = r"\textwidth" if star else r"\linewidth"
        tab = f"\\resizebox{{{width}}}{{!}}{{%\n{tab}\n}}"
    return f"""\\begin{{{env}}}[t]
\\caption{{{caption}}}
\\label{{{label}}}
\\centering
{size}
{tab}
\\end{{{env}}}
"""


def primary_discrimination_table() -> str:
    path = "outputs/isear_compact_gpt_oss_v1/evaluation/selected_layer_baseline_comparison.csv"
    data = read_csv(path)
    rows = []
    for emotion in ("joy", "fear", "anger", "sadness", "disgust", "shame", "guilt", "__macro__"):
        x = next(r for r in data if r["method"] == "functional_direction_primary" and r["pooling"] == "final_token"
                 and r["layer"] == "16" and r["emotion"] == emotion)
        n = "1,545" if emotion == "__macro__" else x["n"]
        p = x["welch_p_fdr"] or "--"
        rows.append(
            f"{('Macro' if emotion == '__macro__' else emotion.capitalize())} & {n} & "
            f"{fmt(x['auroc'])} [{fmt(x['auroc_ci_low'])}, {fmt(x['auroc_ci_high'])}] & "
            f"{fmt(x['average_precision'])} [{fmt(x['average_precision_ci_low'])}, {fmt(x['average_precision_ci_high'])}] & "
            f"{fmt(x['mean_difference'])} & {fmt(x['cohen_d'])} [{fmt(x['cohen_d_ci_low'])}, {fmt(x['cohen_d_ci_high'])}] & {fmt(p, 4)} \\\\"
        )
    return table(
        "Held-out one-versus-rest discrimination for the frozen GPT-OSS block-16 final-token primary directions. "
        "The unit is the report (1,545 reports; 221 participant clusters; 220 target reports for joy/fear and 221 for the other emotions). "
        "Intervals are percentile 95\\% CIs from 2,000 participant-cluster bootstrap resamples. $q$ is the BH-adjusted two-sided Welch target--rest test; the macro row averages emotion-level estimands and has no Welch test.",
        "tab:app-discrimination", "lrrrrrr", "Emotion & $n$ & AUROC [95\\% CI] & AP [95\\% CI] & $\\Delta$ score & Cohen $d$ [95\\% CI] & $q$", rows,
        star=True, size=r"\scriptsize", resize=True)


def comparator_table() -> str:
    path = "outputs/isear_compact_gpt_oss_v1/evaluation/selected_layer_baseline_comparison.csv"
    data = read_csv(path)
    specs = [
        ("Primary final token, block 16", "functional_direction_primary", "final_token", "16", "__macro__"),
        ("Primary mean content, block 16", "functional_direction_primary", "mean_content", "16", "__macro__"),
        ("Related final token, block 16", "functional_direction_related", "final_token", "16", "__macro__"),
        ("Fixed lexical baseline", "lexical", "baseline", "-1", "__macro__"),
        ("DistilRoBERTa (five classes)", "distilroberta_emotion", "baseline", "-1", "__macro__"),
    ]
    rows = []
    for label, method, pooling, layer, emotion in specs:
        candidates = [r for r in data if r["method"] == method and r["emotion"] == emotion]
        if pooling != "baseline":
            candidates = [r for r in candidates if r["pooling"] == pooling and r["layer"] == layer]
        x = candidates[0]
        rows.append(f"{label} & {int(float(x['n_emotions']))} & {fmt(x['auroc'])} [{fmt(x['auroc_ci_low'])}, {fmt(x['auroc_ci_high'])}] & {fmt(x['average_precision'])} [{fmt(x['average_precision_ci_low'])}, {fmt(x['average_precision_ci_high'])}] & {fmt(x['cohen_d'])} [{fmt(x['cohen_d_ci_low'])}, {fmt(x['cohen_d_ci_high'])}] \\\\"
                    )
    return table(
        "Macro discrimination comparisons on their supported categories. Intervals use the same 2,000 participant-cluster bootstrap draws. "
        "The classifier row covers anger, disgust, fear, joy, and sadness only and is not a seven-emotion comparator; paired method-difference intervals were not archived.",
        "tab:app-comparators", "lrrrr", "Method & Classes & AUROC [95\\% CI] & AP [95\\% CI] & Cohen $d$ [95\\% CI]", rows,
        star=True, size=r"\scriptsize", resize=True)


def baseline_by_emotion_table() -> str:
    data = read_csv("outputs/isear_compact_gpt_oss_v1/evaluation/selected_layer_baseline_comparison.csv")
    rows = []
    for emotion in ("joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"):
        lexical = next(r for r in data if r["method"] == "lexical" and r["emotion"] == emotion)
        classifier = next((r for r in data if r["method"] == "distilroberta_emotion" and r["emotion"] == emotion), None)
        classifier_cell = "--" if classifier is None or classifier["status"] != "ok" else f"{fmt(classifier['auroc'])} [{fmt(classifier['auroc_ci_low'])}, {fmt(classifier['auroc_ci_high'])}] / {fmt(classifier['average_precision'])}"
        rows.append(f"{emotion.capitalize()} & {fmt(lexical['auroc'])} [{fmt(lexical['auroc_ci_low'])}, {fmt(lexical['auroc_ci_high'])}] / {fmt(lexical['average_precision'])} & {classifier_cell} \\\\"
                    )
    return table(
        "Per-emotion baselines shown as AUROC [95\\% participant-bootstrap CI] / average precision. The fixed lexical score supports all seven categories. "
        "DistilRoBERTa supports five only; guilt and shame were not scored and are not zero-valued results.",
        "tab:app-baseline-emotion", "lrr", "Emotion & Lexical & DistilRoBERTa", rows,
        star=False, size=r"\scriptsize", resize=True)


def robustness_pca_tables() -> str:
    source = read_csv("outputs/construct_validity_robustness_audit_v1/supplementary_reporting_v1/MISSING_ROBUSTNESS_RESULTS.csv")
    source = [x for x in source if x["analysis"] == "PCA threshold and direction family"]
    order = [(variant, threshold) for variant in ("primary", "related", "family_average") for threshold in ("0%", "25%", "50%", "75%", "90%")]
    outputs = []
    for model, suffix in (("GPT-OSS-20B", "gpt"), ("Llama-3.1-8B-Instruct", "llama")):
        rows = []
        for variant, threshold in order:
            key = f"{variant}; pca={threshold}"
            items = {x["metric"]: x for x in source if x["model"] == model and x["variant"] == key}
            def cell(metric: str) -> str:
                x = items[metric]
                return f"{fmt(x['value'])} [{fmt(x['ci_low'])},{fmt(x['ci_high'])}]" if x["ci_low"] else fmt(x["value"])
            cos = "--" if variant == "family_average" else f"{fmt(items['cosine_to_frozen_mean']['value'])}/{fmt(items['cosine_to_frozen_min']['value'])}"
            rows.append(f"{variant.replace('_',' ').title()} & {threshold} & {int(float(items['neutral_components_removed']['value']))} & {cos} & "
                        f"{cell('macro_ovr_auroc')} & {cell('exact_agreement')} & {cell('group_average_human_llm_rsa')} & {cell('participant_rsa_mean')} & {cell('participant_rsa_median')} \\\\"
                        )
        outputs.append(table(
            f"PCA-threshold and direction-family sensitivity for {model}. Each cell is estimate [95\\% participant-bootstrap CI]; cosine is mean/minimum over seven role-specific directions against the frozen 50\\% vector. "
            "AUROC/agreement/RSA use 1,545 reports/221 participants or the 220 complete-RDM participants as appropriate; reporting additions use 500 resamples.",
            f"tab:app-pca-{suffix}", "llrllllll", "Directions & PCA & $k$ & Cosine & AUROC & Agreement & Group RSA & Mean RSA & Median RSA", rows,
            star=True, size=r"\tiny", resize=True))
    return "\n".join(outputs)


def ridge_tables() -> str:
    source = read_csv("outputs/construct_validity_robustness_audit_v1/supplementary_reporting_v1/MISSING_ROBUSTNESS_RESULTS.csv")
    source = [x for x in source if x["analysis"] == "Out-of-sample emotion-appraisal decomposition"]
    outputs = []
    for model, suffix in (("GPT-OSS-20B", "gpt"), ("Llama-3.1-8B-Instruct", "llama")):
        rows = []
        for emotion in ("joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"):
            items = {x["metric"]: x for x in source if x["model"] == model and x["emotion"] == emotion}
            def cell(metric: str) -> str:
                x = items[metric]
                return f"{fmt(x['value'])} [{fmt(x['ci_low'])},{fmt(x['ci_high'])}]"
            rows.append(f"{emotion.capitalize()} & {cell('R2_C')} & {cell('R2_E')} & {cell('R2_A')} & {cell('R2_F')} & {cell('Delta_E')} & {cell('Delta_A')} & {cell('phi_E')} & {cell('phi_A')} \\\\"
                        )
        outputs.append(table(
            f"Development-CV ridge decomposition for {model} (1,545 held-out reports, 221 participant clusters). Values are predictive $R^2$ or contributions [95\\% participant-bootstrap CI]. "
            "Five-fold grouped development CV selected regularization; preprocessing was inside the pipeline; CIs condition on fixed fitted predictions (500 resamples).",
            f"tab:app-ridge-{suffix}", "lrrrrrrrr", "Direction & $R_C^2$ & $R_E^2$ & $R_A^2$ & $R_F^2$ & $\\Delta_E$ & $\\Delta_A$ & $\\phi_E$ & $\\phi_A$", rows,
            star=True, size=r"\tiny", resize=True))
    return "\n".join(outputs)


def confusion_table() -> str:
    data = [r for r in read_csv("outputs/isear_compact_gpt_oss_v1/explanation_v1/tables/confusion_matrix.csv")
            if r["score_type"] == "train_standardized"]
    emotions = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
    rows = []
    for human in emotions:
        row = [next(r for r in data if r["human_emotion"] == human and r["predicted_emotion"] == pred) for pred in emotions]
        diag = next(r for r in row if r["predicted_emotion"] == human)
        rows.append(f"{human.capitalize()} ({sum(int(r['count']) for r in row)}) & " + " & ".join(r["count"] for r in row) +
                    f" & {100*float(diag['row_proportion']):.1f} [{100*float(diag['row_proportion_ci_low']):.1f}, {100*float(diag['row_proportion_ci_high']):.1f}] \\\\"
                    )
    return table(
        "Frozen seven-way argmax confusion matrix after development-set $z$ calibration. Rows are Human labels and columns are dominant functional directions; cells are report counts. "
        "The last column gives diagonal row percentages and 95\\% participant-bootstrap CIs (2,000 resamples). Row denominators are printed in parentheses.",
        "tab:app-confusion", "lrrrrrrrr", "Human ($n$) & Joy & Fear & Anger & Sad. & Disg. & Shame & Guilt & Match \\% [CI]", rows,
        star=True, size=r"\scriptsize", resize=True)


def construct_table() -> str:
    data = read_csv("outputs/isear_compact_gpt_oss_v1/regression/construct_validity_summary.csv")
    rows = []
    for emotion in ("joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"):
        x = next(r for r in data if r["method"] == "functional_direction_primary" and r["pooling"] == "final_token"
                 and r["layer"] == "16" and r["direction_emotion"] == emotion)
        rows.append(f"{emotion.capitalize()} & {fmt(x['target_beta'])} [{fmt(x['target_ci_low'])}, {fmt(x['target_ci_high'])}] & "
                    f"{fmt(x['target_p_fdr'],4)} & {fmt(x['target_lr_p_fdr'],4)} & {fmt(x['target_delta_marginal_r2'])} & {fmt(x['appraisal_delta_marginal_r2'])} & "
                    f"{fmt(x['appraisal_block_lr_p_fdr'],4)} & {fmt(x['full_marginal_r2'])} & {esc(x['warnings'] or 'none')} \\\\"
                    )
    return table(
        "Target-versus-rest construct-validity mixed models for seven GPT-OSS primary scores (1,545 reports; 221 participants). "
        "Models contain categorical country and nine categorical appraisals as fixed effects and participant random intercepts, and were fitted by maximum likelihood. "
        "$q$ values are BH-adjusted within the archived test families; $\\Delta R_m^2$ is the unique marginal contribution of the indicated block.",
        "tab:app-construct", "lrrrrrrrl", "Direction & Target $\\beta$ [95\\% CI] & Coef. $q$ & Target LR $q$ & $\\Delta R^2_T$ & $\\Delta R^2_A$ & Appraisal LR $q$ & Full $R_m^2$ & Warning", rows,
        star=True, size=r"\scriptsize", resize=True)


def secondary_construct_table() -> str:
    data = read_csv("outputs/isear_compact_gpt_oss_v1/explanation_v1/tables/categorical_emotion_appraisal_decomposition.csv")
    rows = []
    for emotion in ("joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"):
        x = next(r for r in data if r["direction_emotion"] == emotion)
        rows.append(
            f"{emotion.capitalize()} & {fmt(x['country_r2'])} & {fmt(x['emotion_country_r2'])} & {fmt(x['appraisal_country_r2'])} & {fmt(x['full_r2'])} & "
            f"{fmt(x['emotion_after_appraisal_delta_r2'])} & {fmt(x['emotion_block_wald_p_fdr'],4)} & "
            f"{fmt(x['appraisal_after_emotion_delta_r2'])} & {fmt(x['appraisal_block_wald_p_fdr'],4)} \\\\"
        )
    return table(
        "Secondary seven-category construct-validity specification (1,545 reports; 221 participant clusters). Models include country and categorical appraisals; OLS Wald inference uses participant-cluster-robust covariance. "
        "$R_C^2,R_E^2,R_A^2,R_F^2$ are country-only, emotion-plus-country, appraisal-plus-country, and full-model marginal $R^2$. The two $\\Delta R^2$ columns are unique full-minus-reduced contributions; $q$ is BH-adjusted across seven directions separately by block.",
        "tab:app-secondary-construct", "lrrrrrrrr", "Direction & $R_C^2$ & $R_E^2$ & $R_A^2$ & $R_F^2$ & $\\Delta R_E^2$ & Emotion $q$ & $\\Delta R_A^2$ & Appraisal $q$", rows,
        star=True, size=r"\scriptsize", resize=True)


def appraisal_drop_table() -> str:
    data = read_csv("outputs/isear_compact_gpt_oss_v1/explanation_v1/tables/direction_appraisal_signatures.csv")
    emotions = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
    apps = ["EXPC", "PLEA", "PLAN", "FAIR", "CAUS", "COPING", "MORL", "SELF", "RELA"]
    rows = []
    for e in emotions:
        cells = []
        for a in apps:
            x = next(r for r in data if r["direction_emotion"] == e and r["appraisal"] == a)
            cells.append(f"{float(x['drop_one_delta_r2']):.4f}/{fmt(x['joint_wald_p_fdr_global'],3)}")
        rows.append(f"{e.capitalize()} & " + " & ".join(cells) + r" \\")
    return table(
        "Unique drop-one marginal $R^2$ and global BH-adjusted joint-Wald $q$ for each categorical appraisal in each direction-specific OLS model ($\\Delta R_m^2/q$; 1,545 reports, 221 participant clusters). "
        "Covariance is participant-cluster robust. Appraisal abbreviations follow the ISEAR archive: expectedness, pleasantness, plan/goal relevance, fairness, causation, coping, morality, self responsibility, and relationship relevance.",
        "tab:app-appraisal-drop", "lrrrrrrrrr", "Direction & EXPC & PLEA & PLAN & FAIR & CAUS & COPING & MORL & SELF & RELA", rows,
        star=True, size=r"\tiny", resize=True)


def layer_table() -> str:
    data = read_csv("outputs/isear_compact_gpt_oss_v1/rsa/layerwise/tables/layerwise_rsa_statistics.csv")
    rows = []
    for layer in range(24):
        a = next(r for r in data if r["layer"] == str(layer) and r["reference"] == "appraisal")
        c = next(r for r in data if r["layer"] == str(layer) and r["reference"] == "emotion_category")
        rows.append(f"{layer} & {fmt(a['rsa_spearman'])} [{fmt(a['bootstrap_ci_low'])}, {fmt(a['bootstrap_ci_high'])}] & {fmt(a['permutation_p_two_sided_fdr_bh'],4)} & "
                    f"{fmt(c['rsa_spearman'])} [{fmt(c['bootstrap_ci_low'])}, {fmt(c['bootstrap_ci_high'])}] & {fmt(c['permutation_p_two_sided_fdr_bh'],4)} \\\\"
                    )
    return table(
        "Layer-wise GPT-OSS report-pair RSA against appraisal and emotion-category reference distances (1,545 reports; 221 participants; 1,192,740 report pairs). "
        "Intervals use participant-cluster bootstrap resampling; $q$ is BH-adjusted across 24 layers separately by reference. These report-pair estimands are not the 220-participant group-RDM RSA.",
        "tab:app-layers", "rrrrr", "Block & Appraisal $\\rho$ [95\\% CI] & $q$ & Category $\\rho$ [95\\% CI] & $q$", rows,
        star=False, size=r"\tiny", resize=True)


def within_overlap_table() -> str:
    data = read_csv("outputs/within_emotion_variability_alignment_v1/RESULTS.csv")
    rows = []
    for analysis, coding in (("categorical_original", "Original"), ("categorical_missing", "Missing")):
        for model_name, model in (("GPT-OSS-20B", "GPT"), ("Llama-3.1-8B-Instruct", "Llama")):
            for emotion in ("joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"):
                x = next(r for r in data if r["analysis"] == analysis and r["scope"] == "emotion"
                         and r["model"] == model_name and r["emotion"] == emotion)
                rows.append(
                    f"{coding} & {model} & {emotion.capitalize()} & {x['n_reports']} ({int(float(x['n_pairs_valid'])):,}) & "
                    f"{fmt(x['overlap_fraction'])} [{fmt(x['overlap_fraction_ci_low'])}, {fmt(x['overlap_fraction_ci_high'])}] & "
                    f"{fmt(x['overlap_permutation_null_mean'])} & {fmt(x['overlap_permutation_p'],4)} / {fmt(x['overlap_holm_p'],4)} \\\\"
                )
    return table(
        "Human-high/model-close overlap by reported emotion. Human-high means the upper quartile of within-emotion Human distances and model-close the lower quartile of model distances; overlap is the conditional fraction among Human-high pairs. "
        "The sample is 1,545 reports/221 eligible participant clusters. Intervals use 10,000 participant bootstraps; null means and one-sided raw $p$/Holm $q$ use 10,000 bijective participant mappings, adjusted across seven emotions separately by model and coding. Undefined pairwise-missing Gower pairs are excluded.",
        "tab:app-within-overlap", "lllrrrr", "Coding & Model & Emotion & $n$ reports (valid pairs) & Overlap [95\\% CI] & Null mean & $p/q$", rows,
        star=True, size=r"\tiny", resize=True)


def prototype_missing_table() -> str:
    data = [r for r in read_csv("outputs/individual_vs_population_alignment_v1/RESULTS.csv")
            if r["record_type"] == "aggregate" and r["analysis"] == "categorical_missing"]
    labels = (
        ("mean_individualized_rsa", r"Mean $r_{ind}$"),
        ("median_individualized_rsa", r"Median $r_{ind}$"),
        ("mean_generic_development_rsa", r"Mean $r_{generic}$"),
        ("median_generic_development_rsa", r"Median $r_{generic}$"),
        ("mean_delta_development", r"Mean $\Delta$"),
        ("median_delta_development", r"Median $\Delta$"),
        ("fraction_delta_development_positive", r"Fraction $\Delta>0$"),
    )
    rows = []
    for model_name, model in (("GPT-OSS-20B", "GPT"), ("Llama-3.1-8B-Instruct", "Llama")):
        for estimand, label in labels:
            x = next(r for r in data if r["model"] == model_name and r["estimand"] == estimand)
            rows.append(f"{model} & {label} & {fmt(x['estimate'])} [{fmt(x['ci_low'])}, {fmt(x['ci_high'])}] \\\\"
                        )
    return table(
        "Individual-versus-development-prototype sensitivity when ISEAR code 0 is pairwise missing ($P=203$ held-out participants; 799 participant-disjoint development prototypes). "
        "Intervals are 10,000 participant-bootstrap percentile CIs conditional on the fixed prototype. $r_{ind}$, $r_{generic}$, and $\\Delta$ are defined in the text; the final estimand is a participant fraction.",
        "tab:app-prototype-missing", "llr", "Model & Estimand & Estimate [95\\% CI]", rows,
        star=False, size=r"\scriptsize", resize=True)


def clause_emotion_interval_table() -> str:
    data = read_csv("outputs/isear_compact_gpt_oss_v1/clause_dynamics_v1/tables/terminal_mechanism_prevalence.csv")
    rows = []
    for emotion in ("joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"):
        flip = next(r for r in data if r["grouping"] == "human_emotion" and r["human_emotion"] == emotion
                    and r["mechanism"] == "final_clause_flip")
        loss = next(r for r in data if r["grouping"] == "human_emotion" and r["human_emotion"] == emotion
                    and r["mechanism"] == "target_loss")
        rows.append(
            f"{emotion.capitalize()} & {flip['n_reports']} & {fmt(flip['prevalence'])} [{fmt(flip['ci_low'])}, {fmt(flip['ci_high'])}] & "
            f"{fmt(loss['prevalence'])} [{fmt(loss['ci_low'])}, {fmt(loss['ci_high'])}] \\\\"
        )
    return table(
        "Saved uncertainty for final-clause flip and target-loss prevalence by reported emotion. The unit is the report (618 reports from 174 participant clusters overall); within an emotion there is at most one report per participant. "
        "Intervals are percentile 95\\% participant-bootstrap CIs from the archived clause audit. Per-emotion intervals for mean margin change and full-match prevalence were not archived and are therefore not supplied.",
        "tab:app-clause-emotion-ci", "lrrr", "Emotion & $n$ & Flip [95\\% CI] & Target loss [95\\% CI]", rows,
        star=False, size=r"\scriptsize", resize=True)


def strong_feature_table() -> str:
    data = read_csv("outputs/isear_compact_gpt_oss_v1/explanation_v1/tables/strong_match_mismatch_features.csv")
    rows = []
    for x in data:
        rows.append(
            f"{esc(x['feature'])} & {fmt(x['strong_agreement_mean'])} & {fmt(x['strong_disagreement_mean'])} & "
            f"{fmt(x['standardized_adjusted_difference'])} [{fmt(x['ci_low'])}, {fmt(x['ci_high'])}] & "
            f"{fmt(x['p_value'],4)} & {fmt(x['p_value_fdr'],4)} \\\\"
        )
    return table(
        "All prespecified linguistic-feature contrasts between the upper and lower 10\\% tails of the development-standardized Human-direction margin (159 strong agreements and 159 strong disagreements; $n=318$ reports). "
        "The estimate is a standardized adjusted difference from a model controlling reported emotion and country with participant-cluster-robust covariance; intervals are cluster-robust 95\\% CIs and $q$ is BH-adjusted across 32 features. Tail selection makes these explanatory contrasts, not population prevalence estimates.",
        "tab:app-strong-features", "lrrrrr", "Feature & Match mean & Mismatch mean & Adjusted difference [95\\% CI] & Raw $p$ & BH $q$", rows,
        star=True, size=r"\tiny", resize=True)


def terminal_feature_table() -> str:
    data = read_csv("outputs/isear_compact_gpt_oss_v1/clause_dynamics_v1/tables/terminal_feature_models.csv")
    order = ("contrastive_marker", "negation", "resolution_safety_relief", "positive_outcome_reversal",
             "negative_outcome_reversal", "temporal_reversal", "explicit_emotion_word", "narrator_agent_shift",
             "responsibility_blame_moral")
    rows = []
    for feature in order:
        margin = next(r for r in data if r["feature"] == feature and r["outcome"] == "delta_last_final_token")
        loss = next(r for r in data if r["feature"] == feature and r["outcome"] == "target_loss_final_token")
        rows.append(
            f"{esc(feature)} & {fmt(margin['estimate'],4)} [{fmt(margin['ci_low_cluster'],4)}, {fmt(margin['ci_high_cluster'],4)}] & "
            f"{fmt(margin['p_value_cluster'],4)} & {fmt(margin['p_fdr'],4)} & "
            f"{fmt(loss['estimate'],4)} [{fmt(loss['ci_low_cluster'],4)}, {fmt(loss['ci_high_cluster'],4)}] & "
            f"{fmt(loss['odds_ratio'],3)} & {fmt(loss['p_value_cluster'],4)} & {fmt(loss['p_fdr'],4)} \\\\"
        )
    return table(
        "Final-clause lexical-feature models for all nine features ($n=618$ reports; 174 participant clusters). Each feature was tested in a separate model with report and clause-length controls. "
        "Margin estimates are development-standardized units; target-loss estimates are log odds with odds ratios shown. Intervals and raw $p$ values use participant-cluster-robust covariance; BH $q$ values are adjusted separately across nine features for each outcome. Participant-bootstrap margin CIs are archived in the master ledger.",
        "tab:app-terminal-features", "lrrrrrrr", "Feature & Margin $\\beta$ [95\\% CI] & $p$ & $q$ & Loss log-odds [95\\% CI] & OR & $p$ & $q$", rows,
        star=True, size=r"\tiny", resize=True)


def calibration_tables() -> str:
    data = read_csv("outputs/construct_validity_robustness_audit_v1/tables/calibration_robustness.csv")
    rows = []
    for x in data:
        model = "GPT" if x["model"] == "GPT-OSS-20B" else "Llama"
        rows.append(f"{model} & {esc(x['calibration'])} & {x['n_reports']} & {fmt(x['exact_agreement'])} & {fmt(x['macro_auroc'])} & {fmt(x['mean_top_two_gap'])} \\\\"
                    )
    outputs = [table(
        "Development-only cross-direction calibration sensitivity (1,545 held-out reports; 221 participants). Agreement is frozen unsupervised argmax, AUROC is macro one-versus-rest, and the final column is the mean largest-minus-second-largest calibrated score. Gap scales differ by calibration and are not probabilities. No CI was archived for these six mean gaps.",
        "tab:app-calibration", "llrrrr", "Model & Calibration & $n$ & Agreement & Macro AUROC & Mean top-two gap", rows,
        star=False, size=r"\scriptsize", resize=True)]
    curves = read_csv("outputs/construct_validity_robustness_audit_v1/tables/selective_agreement_curves.csv")
    for model_name, model, suffix in (("GPT-OSS-20B", "GPT-OSS", "gpt"), ("Llama-3.1-8B-Instruct", "Llama", "llama")):
        curve_rows = []
        for x in curves:
            if x["model"] != model_name:
                continue
            curve_rows.append(
                f"{esc(x['calibration'])} & {fmt(x['target_retention'],1)} & {fmt(x['observed_retention'])} & "
                f"{fmt(x['gap_threshold'])} & {fmt(x['agreement'])} [{fmt(x['ci_low'])}, {fmt(x['ci_high'])}] \\\\"
            )
        outputs.append(table(
            f"Selective-agreement curve for {model}. Reports are ranked by the top-two calibrated-score gap and the requested upper fraction is retained (1,545 reports; 221 participant clusters before selection). "
            "Thresholds are calibration-scale-specific; agreement intervals use the archived participant bootstrap. Selection is descriptive and does not turn the gap into a calibrated probability.",
            f"tab:app-selective-{suffix}", "lrrrr", "Calibration & Target retained & Observed retained & Gap threshold & Agreement [95\\% CI]", curve_rows,
            star=True, size=r"\tiny", resize=True))
    return "\n".join(outputs)


def decoder_cv_table() -> str:
    data = read_csv("outputs/construct_validity_robustness_audit_v1/tables/supervised_decoder_development_cv.csv")
    rows = []
    for x in data:
        model = "GPT" if x["model"] == "GPT-OSS-20B" else "Llama"
        rows.append(f"{model} & {fmt(x['C'],2)} & {fmt(x['grouped_cv_accuracy'])} & {fmt(x['grouped_cv_sd'])} \\\\"
                    )
    return table(
        "Development-only five-fold participant-grouped cross-validation for the exploratory multinomial decoder. The selected $C$ values were 1 for GPT and 10 for Llama; the held-out outcomes and participant-bootstrap CIs are reported in the text. This supervised estimand is separate from frozen argmax.",
        "tab:app-decoder-cv", "lrrr", "Model & $C$ & Development CV accuracy & Fold SD", rows,
        star=False, size=r"\scriptsize", resize=True)


LEDGER: list[dict[str, str]] = []


def add_result(*, experiment: str, claim: str, statistic: str, model: str = "", n_reports: str = "",
               n_participants: str = "", condition: str = "", comparator: str = "", estimate: str = "",
               ci_low: str = "", ci_high: str = "", ci_method: str = "", p_value: str = "",
               alternative: str = "", correction: str = "", source: str = "", source_location: str = "",
               appendix: str = "", status: str = "verified") -> None:
    LEDGER.append(locals().copy())


def add_core_ledger() -> None:
    # Completion effects and neutral PCA counts.
    for x in read_csv("artifacts/compact_gpt_oss_v1_frozen/matching_effects.csv"):
        for metric in ("matching_delta", "mean_nonmatching_delta", "max_nonmatching_delta"):
            add_result(experiment="synthetic_functional_assay", claim=f"{x['prompt_id']} {x['emotion']} {metric}",
                       statistic=metric, model="GPT-OSS-20B", n_reports="1 synthetic prompt completion",
                       condition=x["prompt_id"], estimate=x[metric], source="artifacts/compact_gpt_oss_v1_frozen/matching_effects.csv",
                       source_location=f"prompt_id={x['prompt_id']};emotion={x['emotion']}", appendix="app:frozen-directions")
    for x in read_csv("artifacts/compact_gpt_oss_v1_frozen/neutral_pca.csv"):
        add_result(experiment="direction_construction", claim=f"block {x['layer']} neutral PCA", statistic="components_removed",
                   model="GPT-OSS-20B", condition="50% neutral variance", estimate=x["n_components"],
                   source="artifacts/compact_gpt_oss_v1_frozen/neutral_pca.csv", source_location=f"layer={x['layer']}", appendix="tab:app-pca-components")
        add_result(experiment="direction_construction", claim=f"block {x['layer']} neutral PCA", statistic="variance_explained",
                   model="GPT-OSS-20B", condition="50% neutral variance", estimate=x["variance_explained"],
                   source="artifacts/compact_gpt_oss_v1_frozen/neutral_pca.csv", source_location=f"layer={x['layer']}", appendix="tab:app-pca-components")
    # Primary discrimination, dominance, construct and appraisal drop-one rows.
    p = "outputs/isear_compact_gpt_oss_v1/evaluation/selected_layer_baseline_comparison.csv"
    for x in read_csv(p):
        if x["status"] != "ok":
            continue
        for metric, low, high in (("auroc", "auroc_ci_low", "auroc_ci_high"),
                                  ("average_precision", "average_precision_ci_low", "average_precision_ci_high"),
                                  ("cohen_d", "cohen_d_ci_low", "cohen_d_ci_high"),
                                  ("mean_difference", "", "")):
            if not x.get(metric):
                continue
            add_result(experiment="heldout_discrimination", claim=f"{x['method']} {x['pooling']} block {x['layer']} {x['emotion']}", statistic=metric,
                       model="GPT-OSS-20B", n_reports=x.get("n", "") or "1545 macro", n_participants="221",
                       condition=x["emotion"], estimate=x[metric], ci_low=x.get(low, ""), ci_high=x.get(high, ""),
                       ci_method="2000 participant-cluster percentile bootstrap" if low else "",
                       p_value=x.get("welch_p_fdr", ""), alternative="two-sided target-rest" if x.get("welch_p_fdr") else "",
                       correction="BH FDR" if x.get("welch_p_fdr") else "", source=p,
                       source_location=f"method={x['method']};pooling={x['pooling']};layer={x['layer']};emotion={x['emotion']}", appendix="tab:app-discrimination/tab:app-comparators")
    p = "outputs/isear_compact_gpt_oss_v1/explanation_v1/tables/confusion_matrix.csv"
    for x in read_csv(p):
        add_result(experiment="seven_way_dominance", claim=f"{x['score_type']} {x['human_emotion']} to {x['predicted_emotion']}", statistic="row_proportion",
                   model="GPT-OSS-20B", n_reports=x["count"], n_participants="221 clusters", condition=x["human_emotion"], comparator=x["predicted_emotion"],
                   estimate=x["row_proportion"], ci_low=x["row_proportion_ci_low"], ci_high=x["row_proportion_ci_high"],
                   ci_method="2000 participant-cluster percentile bootstrap", source=p,
                   source_location=f"score_type={x['score_type']};human={x['human_emotion']};predicted={x['predicted_emotion']}", appendix="tab:app-confusion")
    p = "outputs/isear_compact_gpt_oss_v1/regression/construct_validity_summary.csv"
    for x in read_csv(p):
        if not (x["method"] == "functional_direction_primary" and x["pooling"] == "final_token" and x["layer"] == "16"):
            continue
        for metric in ("target_beta", "target_delta_marginal_r2", "appraisal_delta_marginal_r2", "full_marginal_r2",
                       "target_lr_p_fdr", "appraisal_block_lr_p_fdr"):
            add_result(experiment="construct_validity", claim=x["direction_emotion"], statistic=metric, model="GPT-OSS-20B",
                       n_reports=x["n"], n_participants=x["n_participants"], estimate=x[metric],
                       ci_low=x["target_ci_low"] if metric == "target_beta" else "", ci_high=x["target_ci_high"] if metric == "target_beta" else "",
                       p_value=x["target_p_fdr"] if metric == "target_beta" else x["target_lr_p_fdr"] if metric == "target_delta_marginal_r2" else x["appraisal_block_lr_p_fdr"] if metric == "appraisal_delta_marginal_r2" else "",
                       correction="BH FDR", source=p, source_location=f"direction_emotion={x['direction_emotion']}", appendix="tab:app-construct")
    p = "outputs/isear_compact_gpt_oss_v1/explanation_v1/tables/direction_appraisal_signatures.csv"
    for x in read_csv(p):
        add_result(experiment="appraisal_drop_one", claim=f"{x['direction_emotion']} {x['appraisal']}", statistic="drop_one_delta_marginal_r2",
                   model="GPT-OSS-20B", n_reports=x["n"], n_participants=x["n_participants"], estimate=x["drop_one_delta_r2"],
                   p_value=x["joint_wald_p_fdr_global"], alternative="two-sided joint Wald", correction="global BH over 63 tests",
                   source=p, source_location=f"direction={x['direction_emotion']};appraisal={x['appraisal']}", appendix="tab:app-appraisal-drop")
    p = "outputs/isear_compact_gpt_oss_v1/rsa/tables/rsa_participant_emotion_mismatch.csv"
    for x in read_csv(p):
        if x["scope"] != "group_average":
            continue
        add_result(experiment="group_rsa_local_mismatch", claim=x["emotion"], statistic="mean_absolute_rank_mismatch",
                   model="GPT-OSS-20B", n_reports="1540", n_participants="220", estimate=x["mean_absolute_rank_mismatch"],
                   source=p, source_location=f"scope=group_average;emotion={x['emotion']}", appendix="app:rsa")


def add_numeric_csv(rel: str, experiment: str, appendix: str) -> None:
    """Normalize every numeric cell in a compact authoritative result table."""
    rows = read_csv(rel)
    for row_number, x in enumerate(rows, start=2):
        identity = ";".join(f"{k}={v}" for k, v in x.items() if v and not _is_number(v))
        model = x.get("model", "") or x.get("source", "")
        for key, value in x.items():
            if not value or not _is_number(value):
                continue
            add_result(experiment=experiment, claim=identity[:500] or f"row {row_number}", statistic=key,
                       model=model, n_reports=x.get("n_reports", "") or x.get("n", ""),
                       n_participants=x.get("n_participants", ""), condition=x.get("condition", ""),
                       estimate=value, source=rel, source_location=f"CSV row {row_number}; column {key}", appendix=appendix)


def _is_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def write_ledger() -> None:
    add_core_ledger()
    for rel, experiment, appendix in (
        ("outputs/isear_compact_gpt_oss_v1/rsa/tables/rsa_statistics.csv", "primary_rsa", "app:rsa"),
        ("outputs/isear_compact_llama31_8b_v1/rsa/tables/rsa_statistics.csv", "primary_rsa", "app:rsa"),
        ("outputs/isear_compact_gpt_oss_v1/rsa/cross_participant_explanation_v1/tables/crossed_random_effects_variance.csv", "crossed_rsa", "app:rsa"),
        ("outputs/isear_compact_gpt_oss_v1/rsa/cross_participant_explanation_v1/tables/candidate_feature_associations.csv", "crossed_rsa_features", "app:rsa"),
        ("outputs/isear_compact_gpt_oss_v1/rsa/cross_participant_explanation_v1/tables/multiple_regression_rsa_summary.csv", "multiple_regression_rsa", "app:rsa"),
        ("outputs/isear_compact_gpt_oss_v1/rsa/layerwise/tables/layerwise_rsa_statistics.csv", "layerwise_rsa", "tab:app-layers"),
        ("outputs/isear_compact_gpt_oss_v1/rsa/layerwise/tables/layerwise_rsa_contrasts.csv", "layerwise_rsa_contrast", "app:rsa"),
        ("outputs/construct_validity_robustness_audit_v1/tables/direction_pca_pooling_robustness.csv", "pca_direction_robustness", "app:robustness-external"),
        ("outputs/construct_validity_robustness_audit_v1/supplementary_reporting_v1/MISSING_ROBUSTNESS_RESULTS.csv", "robustness_reporting_complete", "app:robustness-external"),
        ("outputs/construct_validity_robustness_audit_v1/tables/calibration_robustness.csv", "calibration_robustness", "app:dominance"),
        ("outputs/construct_validity_robustness_audit_v1/tables/selective_agreement_curves.csv", "selective_agreement", "tab:app-selective-gpt/tab:app-selective-llama"),
        ("outputs/construct_validity_robustness_audit_v1/tables/supervised_decoder_development_cv.csv", "supervised_decoder_cv", "tab:app-decoder-cv"),
        ("outputs/construct_validity_robustness_audit_v1/tables/supervised_decoder_test.csv", "supervised_decoder", "app:dominance"),
        ("outputs/construct_validity_robustness_audit_v1/tables/rsa_pairing_advantage.csv", "participant_pairing", "app:mapping-coding"),
        ("outputs/construct_validity_robustness_audit_v1/supplementary_reporting_v1/same_203_appraisal_coding_comparison.csv", "appraisal_coding", "app:mapping-coding"),
        ("outputs/isear_compact_gpt_oss_v1/clause_dynamics_v1/tables/advanced_diagnostics/clause_result_overall.csv", "clause_dynamics", "app:narrative"),
        ("outputs/isear_compact_gpt_oss_v1/clause_dynamics_v1/tables/advanced_diagnostics/clause_result_by_emotion.csv", "clause_dynamics_by_emotion", "tab:app-clause-emotion"),
        ("outputs/isear_compact_gpt_oss_v1/clause_dynamics_v1/tables/clause_position_distribution.csv", "clause_position", "app:narrative"),
        ("outputs/isear_compact_gpt_oss_v1/clause_dynamics_v1/tables/clause_position_regression.csv", "clause_position_regression", "app:narrative"),
        ("outputs/isear_compact_gpt_oss_v1/clause_dynamics_v1/tables/terminal_feature_models.csv", "clause_features", "app:narrative"),
        ("outputs/isear_compact_gpt_oss_v1/clause_dynamics_v1/tables/terminal_mechanism_prevalence.csv", "clause_mechanisms", "app:narrative"),
        ("outputs/isear_compact_gpt_oss_v1/explanation_v1/tables/strong_match_mismatch_features.csv", "strong_match_mismatch_features", "tab:app-strong-features"),
        ("outputs/isear_compact_gpt_oss_v1/explanation_v1/tables/categorical_emotion_appraisal_decomposition.csv", "seven_category_construct", "tab:app-secondary-construct"),
        ("notebooks/paper_figures/clause_position_pooling_by_emotion_gpt_vs_llama.csv", "clause_mechanism_selected", "app:narrative"),
        ("notebooks/paper_figures/naming_emotion_specificity_gpt.csv", "explicit_emotion_naming", "app:narrative"),
        ("notebooks/paper_figures/naming_emotion_specificity_llama.csv", "explicit_emotion_naming", "app:narrative"),
        ("outputs/complete_appendix_audit/recomputed_clause_pair_cis.csv", "named_clause_pair", "app:narrative"),
    ):
        add_numeric_csv(rel, experiment, appendix)
    # Already normalized audit result files: preserve every row and its source columns as a separate estimand.
    for rel, experiment, appendix in (
        ("outputs/individual_variability_attenuation_v1/RESULTS.csv", "between_person_dispersion", "app:dispersion"),
        ("outputs/individual_vs_population_alignment_v1/RESULTS.csv", "individual_vs_prototype", "app:individual-prototype"),
        ("outputs/within_emotion_variability_alignment_v1/RESULTS.csv", "within_emotion_correspondence", "app:within-emotion"),
    ):
        for i, x in enumerate(read_csv(rel), start=2):
            keys = list(x)
            statistic = x.get("estimand") or x.get("metric") or x.get("analysis") or "archived_estimand"
            estimate = x.get("estimate") or x.get("value") or x.get("rho") or ""
            add_result(experiment=experiment, claim=x.get("emotion") or x.get("source") or x.get("model") or f"row {i}",
                       statistic=statistic, model=x.get("model", ""), n_reports=x.get("n_reports", ""),
                       n_participants=x.get("n_participants", "") or x.get("n", ""), condition=x.get("condition", "") or x.get("coding", ""),
                       estimate=estimate, ci_low=x.get("ci_low", ""), ci_high=x.get("ci_high", ""),
                       ci_method=x.get("ci_method", ""), p_value=x.get("p_value", "") or x.get("p", ""),
                       correction=x.get("correction", ""), source=rel, source_location=f"CSV row {i}; raw={json.dumps(x, sort_keys=True)}", appendix=appendix)
    fields = list(LEDGER[0])
    with (OUT / "RESULTS_MASTER.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(LEDGER)


def appendix_text() -> str:
    matching = read_csv("artifacts/compact_gpt_oss_v1_frozen/matching_effects.csv")
    match_rows = []
    for label in ("happy", "joyful", "afraid", "frightened", "angry", "furious", "sad", "depressed", "disgusted", "contemptuous", "ashamed", "embarrassed", "guilty", "remorseful"):
        a = next(x for x in matching if x["emotion"] == label and x["prompt_id"] == "self_report")
        b = next(x for x in matching if x["emotion"] == label and x["prompt_id"] == "third_person")
        match_rows.append(f"{label.capitalize()} & {fmt(a['matching_delta'])} & {fmt(a['max_nonmatching_delta'])} & {fmt(b['matching_delta'])} & {fmt(b['max_nonmatching_delta'])} \\\\"
                          )
    pca = read_csv("artifacts/compact_gpt_oss_v1_frozen/neutral_pca.csv")
    pca_rows = []
    for left, right in zip(pca[:12], pca[12:]):
        pca_rows.append(f"{left['layer']} & {left['n_components']} & {fmt(left['variance_explained'])} & {right['layer']} & {right['n_components']} & {fmt(right['variance_explained'])} \\\\"
                        )
    completion = table(
        "Synthetic steering completion assay at strength $+0.5$ versus 0, block 16. Each effect is the change in teacher-forced completion score for the matching label; Max nonmatch is the largest nonmatching-label change in the same prompt. "
        "The unit is one fixed synthetic prompt/label completion, so no participant CI or $p$ value applies.",
        "tab:app-steering", "lrrrr", "Direction & Self match & Self max nonmatch & Third-person match & Third-person max nonmatch", match_rows,
        star=True, size=r"\scriptsize", resize=True)
    pcat = table(
        "Neutral PCA removal by GPT-OSS transformer block. $k$ is the minimum component count reaching the prespecified 50\\% neutral-variance threshold; variance is the achieved cumulative fraction. The unit is the 1,200-dialogue frozen neutral corpus.",
        "tab:app-pca-components", "rrrrrr", "Block & $k$ & Variance & Block & $k$ & Variance", pca_rows, size=r"\scriptsize")

    return r"""\appendix
\section{Complete quantitative and reproducibility appendix}
\label{app:complete}

This appendix is deliberately redundant with the main paper: it preserves the numerical evidence if the main text is shortened. Confirmatory status is inherited from the frozen analysis plan. Unless stated otherwise, GPT-OSS means \texttt{openai/gpt-oss-20b}, the prespecified block-16, 50\%-neutral-PCA, primary-direction, final-token readout. Llama-3.1-8B-Instruct analyses are post hoc robustness audits at its frozen block 21. No result below is evidence of model sentience, human-reader performance, generated-advice quality, or user harm.

\subsection{Samples, estimands, and analysis status}
\label{app:denominators}

The report-level held-out set has 1,545 eligible reports from 221 participants (220 joy, 220 fear, and 221 reports for each other emotion). Participant RDM analyses require one usable report for all seven emotions and therefore use 1,540 reports from 220 participants. Treating ISEAR code 0 as pairwise missing yields a common 203-person sensitivity cohort; complete-case coding yields 17. Development contains 6,121 reports from 875 participant IDs, but complete seven-emotion model prototypes use 871 participants (799 for the pairwise-missing sensitivity). Clause analyses use 618 multi-clause reports from 174 participants and 1,799 report--clause deletion comparisons. These populations are not interchangeable.

Primary, prespecified tests are the GPT-OSS one-versus-rest discrimination, development-standardized frozen argmax, target/appraisal mixed models, group and participant RSA, and the frozen clause analysis. Alternative PCA thresholds, direction families, calibration rules, pooling/layers, Llama replication, alternative appraisal coding, matched mapping, dispersion, within-emotion correspondence, individual-versus-prototype analyses, and named-emotion pair diagnostics are exploratory or sensitivity analyses.

\subsection{Frozen directions, synthetic construction, and functional assay}
\label{app:frozen-directions}

Construction used only synthetic material: 14 labels $\times$ 100 topics $\times$ 12 narratives = 16,800 emotion-conditioned stories, plus 1,200 neutral dialogues (18,000 texts; 1,500 generation cells). Story representations average model tokens from ordinal 50 onward. At every block the label mean was contrasted with the unweighted mean of all 14 label means; the minimum neutral principal components explaining at least 50\% of neutral variance were removed, and vectors were normalized. The seven primary/related pairs were happy/joyful, afraid/frightened, angry/furious, sad/depressed, disgusted/contemptuous, ashamed/embarrassed, and guilty/remorseful. Synthetic texts are never counted as Human data.

""" + pcat + "\n" + completion + r"""
All 14 self-report matching effects were positive and exceeded the within-prompt nonmatching mean; eight also exceeded the maximum nonmatching effect. In third-person prompts, nine matching effects were positive and exceeded the maximum nonmatch, while afraid, frightened, angry, sad, and depressed were nonpositive. Thus functionality is prompt-dependent, not prompt-independent. The audit status is ``assay complete; not automatically validated'': manual story-quality certification was not archived.

The frozen GPT artifact uses model revision \texttt{6cee5e81ee83917806bbde320786a8fb61efebee}; direction tensor SHA-256 \texttt{8ddf6cab92ab6bc8a41816e98345a68ae4061587c985a192c8f4b77d84904c2d}; protocol SHA-256 \texttt{9d01cbcf4b1b386a74f862ad45a6059cb23901291c7819f64f6a042b69c90bde}; seed 7; bfloat16; temperature 0.9; top-$p$ 0.95; intervention strengths 0 and $+0.5$. The Llama audit uses revision \texttt{0e9e39f249a16976918f6564b8830bc894c89659}, direction SHA-256 \texttt{3583da57978dcb850f85c3027c37b57444d17bac983d9300282351d1fdcbda59}, and 21 components at its 50\% block-21 threshold.

\subsection{Held-out emotion discrimination and baselines}
\label{app:discrimination}
""" + primary_discrimination_table() + comparator_table() + baseline_by_emotion_table() + r"""
The lexical baseline supports all seven categories. Its macro AUROC was 0.599753 [0.582779, 0.618950] and AP 0.288530 [0.261721, 0.323853]. DistilRoBERTa supports five categories only and reached macro AUROC 0.918467 [0.906977, 0.929215] and AP 0.756883 [0.730544, 0.785129]. Functional-score AUROC point estimates exceeded the lexical baseline for every emotion, but paired method-difference inference was not run; direct statistical superiority is therefore not claimed. Mean-content pooling reached AUROC 0.662429 [0.649143, 0.676045] and AP 0.307469, and the exploratory block-23 final-token maximum was approximately 0.734; neither reselects the frozen block 16.

\subsection{Seven-way dominance, calibration, and supervised decoding}
\label{app:dominance}

Raw-cosine argmax agreed in 362/1,545 reports (0.234304; 95\% participant-bootstrap CI [0.221646, 0.248218]); development-$z$ argmax agreed in 574/1,545 (0.371521 [0.348210, 0.394825]). Per-emotion $z$-match rates and CIs were joy 0.750 [0.691, 0.805], fear 0.400 [0.336, 0.462], anger 0.335 [0.271, 0.394], sadness 0.262 [0.208, 0.326], disgust 0.290 [0.226, 0.353], shame 0.271 [0.213, 0.330], and guilt 0.294 [0.235, 0.357]. Ties use the fixed emotion order; no randomized tie-breaking is used.

""" + confusion_table() + calibration_tables() + r"""
Among mismatches, the largest destinations were joy$\rightarrow$guilt (38.2\%), fear$\rightarrow$joy (20.5\%), anger$\rightarrow$sadness (25.2\%), sadness$\rightarrow$guilt (38.0\%), disgust$\rightarrow$sadness (29.9\%), shame$\rightarrow$guilt (28.6\%), and guilt$\rightarrow$shame (34.6\%). Development-only alternative calibration changed GPT-OSS agreement to 0.377 for median/MAD and 0.364 for percentile calibration (AUROC remains 0.726); Llama values were 0.519, 0.517, and 0.502 (AUROC 0.800). A separate supervised seven-way decoder reached GPT-OSS accuracy 0.476 [0.452, 0.500] and macro AUROC 0.819 [0.806, 0.831] ($C=1$), and Llama accuracy 0.588 [0.566, 0.612] and AUROC 0.882 [0.871, 0.893] ($C=10$). These supervised outcomes do not replace or validate the frozen argmax estimand.

""" + decoder_cv_table() + r"""

The top-score margin is the largest minus second-largest calibrated score; it is descriptive confidence, not a probability. In the 220-person crossed analysis, smaller mean top-score gaps were associated with larger mismatch ($\rho=-0.361$), but this association is not a calibration test.

\subsection{Construct validity: category and appraisal information}
\label{app:construct}
""" + construct_table() + r"""
Appraisals added more unique marginal variance than the exact target indicator in six of seven directions; joy was the exception. The secondary seven-category, cluster-robust specification retained unique category/appraisal contributions respectively: joy 0.077316/0.042790; fear 0.134734/0.058875; anger 0.041995/0.025786; sadness 0.060668/0.052852; disgust 0.069310/0.030749; shame 0.097952/0.055312; guilt 0.094007/0.046181. Both blocks were BH-significant for every direction; medians were 0.077 and 0.046. This full-factor estimand is not interchangeable with the binary target--rest model.

""" + secondary_construct_table() + r"""

""" + appraisal_drop_table() + ridge_tables() + r"""
Only GPT anger's conditional $\Delta_A$ interval includes zero; all Llama incremental intervals exclude zero. These predictive decompositions are post hoc and their 500-bootstrap CIs condition on development-fitted models rather than including refit uncertainty.

\subsection{Human--model representational similarity}
\label{app:rsa}

For each of 220 complete held-out participants, Human $D_p^H$ is a $7\times7$ categorical-Gower RDM over nine ISEAR appraisal variables: each pair distance is the proportion of codes that differ. Model $D_p^M$ is Euclidean distance between seven-dimensional primary-score profiles standardized by participant-disjoint development means and SDs. RSA is Spearman correlation over the 21 upper-triangle emotion pairs. Group RSA correlates RDMs averaged before correlation; participant-average RSA averages 220 within-person correlations. These are distinct estimands.

GPT-OSS group RSA was 0.711688 (10,000-participant-bootstrap CI [0.571429, 0.805195]; two-sided permutation $p=0.003800$, positive-tail $p=0.001800). Participant RSA mean was 0.174300, median 0.178068, IQR [-0.015152, 0.362729], with 0.727273 positive. Llama group RSA was 0.642857 [0.533766, 0.748052] ($p=0.016298$ two-sided; 0.008899 positive-tail); its participant mean was 0.260726, median 0.281307, and 0.854545 were positive. The Human-to-Human leave-one-out population reference mean was 0.398 [0.365, 0.427] (median .440); GPT and Llama participant-model to other-Human means were .286 [.250,.324] and .399 [.364,.434]. This is a ceiling-like population comparator, not reliability correction.

For the GPT group-average RDM, mean absolute rank mismatch over the six relations involving each focal emotion was joy .142857, fear .222222, anger .150794, sadness .206349, disgust .198413, shame .134921, and guilt .214286. These local descriptive values are neither participant-level RSA nor category accuracy; no separate uncertainty interval was archived.

In the $220\times220=48{,}400$ crossed mismatch matrix, crossed random effects assigned 0.299 of variance to model columns, 0.117 to Human rows, and 0.583 to pair-specific/residual variation; cells were never treated as independent. Persistent mismatch correlated with Human group atypicality ($\rho=-0.810$, CI [-.856,-.756], BH $q=.00140$), model group atypicality ($\rho\approx-.807$), score entropy ($\rho=.375$), and smaller top-score gap ($\rho=-.361$). These characterize rows/columns and do not cause mismatch.

Multiple-regression RSA for GPT-OSS had full $R^2=.245040$ [.219916,.270759]. Unique effects were Human appraisal .083139 [.069380,.098517], permutation $p=.000200$; fixed sentence semantics .075547 [.061871,.089872], $p=.026795$; and TF--IDF .051650 [.042539,.061391], $p=.805439$. Llama values were full .279620 [.257457,.302294], Human .105216 [.090427,.120813], $p=.000200$; semantics .084413 [.071474,.097551], $p=.004599$; TF--IDF .057732 [.047953,.068270], $p=.518296$. A nonzero point estimate with a null permutation test is not called reliable unique evidence.

""" + layer_table() + r"""
At block 16, GPT report-pair appraisal/category correlations were .157825 [.135113,.183654] and .126228 [.113431,.138559]; their difference was .031597 [.009943,.056381], two-sided permutation $p=.000999$ (BH $q=.001262$). Final-token versus mean-content pooling produced GPT macro AUROC .726/.662, agreement .372/.342, group RSA .712/.655, and participant mean RSA .174/.159; corresponding Llama values were .800/.746, .519/.438, .643/.630, and .261/.240. Pooling sensitivity, layer-wise report-pair RSA, and group-RDM RSA must not be conflated.

\subsection{Matched mapping, appraisal coding, and population prototypes}
\label{app:mapping-coding}

Matched-minus-all-unmatched participant RSA was .025668 [.001827,.052359], mapping $p=.025987$ for GPT-OSS and .048101 [.021810,.071210], $p=.000500$ for Llama (10,000 participant bootstrap draws; 2,000 mappings in the original audit). After partialling a participant-disjoint development prototype the estimates were .056547 [.024950,.087018], $p=.018491$, and .108657 [.080127,.144833], $p=.000500$. These model-specific exploratory tests were reported without an additional cross-model adjustment.

Alternative Human coding gave the following participant-mean/group RSA [95\% CI]. GPT-OSS: original categorical, $P=220$, .174 [.134,.214]/.712 [.570,.811]; code-0 pairwise missing, $P=203$, .162 [.124,.201]/.694 [.584,.797]; mixed ordinal/nominal, $P=203$, .186 [.153,.220]/.712 [.603,.808]; complete case, $P=17$, .308 [.179,.423]/.777 [.473,.879]. Llama: .261 [.225,.292]/.643 [.543,.752]; .265 [.228,.299]/.643 [.549,.743]; .301 [.268,.334]/.626 [.566,.765]; and .265 [.134,.392]/.771 [.459,.851]. On exactly the same 203 participants, original categorical coding was GPT .161 [.126,.200]/.734 [.593,.807] and Llama .256 [.222,.289]/.677 [.569,.769], separating coding from selection. Code 0 means ``not applicable''; pairwise missing excludes a pair only when no jointly observed nonzero appraisal remains, without imputation. Mixed coding treats ordered 1--3 responses as ordinal and agency/coping-like variables as nominal.

\subsection{Between-person dispersion and the matched-deviation identity}
\label{app:dispersion}

For centered, unit-normalized rank vectors, $V_s=P^{-1}\sum_p\|u_{p,s}-\bar u_s\|^2$. Joint 10,000-participant bootstrap CIs resample identical participant indices across sources. Under original categorical Gower ($P=220$), Human $V=.797853$ [.764182,.824265], GPT $V=.869008$ [.831764,.895317], and Llama $V=.736326$ [.691913,.771697]. GPT minus Human was +.071155 [.025522,.114570], ratio 1.089183 [1.031278,1.148246], directional lower-dispersion raw/Holm $p=.9991/.9991$. Llama minus Human was -.061527 [-.112225,-.012371], ratio .922884 [.861874,.984262], $p=.0075/.0150$.

With code 0 pairwise missing ($P=203$), Human/GPT/Llama $V$ were .760071 [.722548,.789436], .870714 [.832436,.897678], and .739988 [.694085,.776826]. GPT's difference was +.110643 [.063644,.156855], ratio 1.145570 [1.081544,1.214313], Holm $p=1$; Llama's was -.020083 [-.071605,.032057], ratio .973577 [.907325,1.043380], Holm $p=.4478$. GPT-OSS is therefore more dispersed, and Llama's lower dispersion is coding-sensitive: there is no model-general homogenization result.

Matched deviations were GPT $K=.025551$ [-.001280,.052683], $R=.030686$ [-.001545,.063661], mapping $p=.0304$; Llama $K=.047883$ [.023239,.072675], $R=.062472$ [.030407,.095056], $p=.0002$. Exactly, $K=(P-1)A/P$, where $A$ is matched-minus-unmatched RSA (maximum identity error $1.388\times10^{-17}$). This is a reparameterization, not independent evidence.

\subsection{Within-emotion correspondence}
\label{app:within-emotion}

Within each Human emotion, pairwise categorical-Gower distance was correlated with pairwise model Euclidean distance. Global participant bootstraps and bijective mappings (10,000 each) preserve participant dependence; report pairs are not independent. Ties receive average ranks. Original valid pair counts are 24,090 for joy/fear and 24,310 for each other emotion. Pairwise-missing valid counts are 24,090, 23,652, 24,310, 24,090, 22,578, 22,366, and 22,791 in emotion order; undefined pairs are excluded, not imputed. Emotion $p$ values are two-sided mapping tests Holm-adjusted across seven emotions per model. The Human-high/model-close overlap family is adjusted separately. Pooled summaries equally weight the seven Fisher-$z$ emotion correlations.

\begin{table*}[t]
\caption{Within-emotion Spearman correspondence under original and pairwise-missing Gower. The sample is 1,545 reports/221 eligible participants, not necessarily the 220-person complete-RDM cohort. CIs use 10,000 participant bootstraps; $q$ is Holm-adjusted within model/coding.}
\label{tab:app-within-emotion}
\centering\scriptsize
\begin{tabular}{lllrrr}
\toprule Coding & Model & Emotion & $n$ reports (valid pairs) & $\rho$ [95\% CI] & Holm $q$ \\
\midrule
Original & GPT & Joy & 220 (24,090) & .031 [-.012,.103] & .7067 \\
 & & Fear & 220 (24,090) & .034 [.004,.096] & .3168 \\
 & & Anger & 221 (24,310) & .028 [-.013,.097] & .7067 \\
 & & Sadness & 221 (24,310) & .026 [-.012,.094] & .7067 \\
 & & Disgust & 221 (24,310) & .057 [.010,.134] & .1288 \\
 & & Shame & 221 (24,310) & .064 [-.001,.159] & .1288 \\
 & & Guilt & 221 (24,310) & .067 [.010,.154] & .1288 \\
Original & Llama & Joy & 220 (24,090) & .006 [-.045,.086] & .8518 \\
 & & Fear & 220 (24,090) & .026 [-.005,.088] & .3786 \\
 & & Anger & 221 (24,310) & .064 [.015,.141] & .1412 \\
 & & Sadness & 221 (24,310) & .065 [.022,.137] & .0306 \\
 & & Disgust & 221 (24,310) & .043 [.004,.112] & .1839 \\
 & & Shame & 221 (24,310) & .096 [.047,.174] & .0042 \\
 & & Guilt & 221 (24,310) & .075 [.027,.151] & .0306 \\
Missing & GPT & Joy & 220 (24,090) & .029 [-.015,.103] & 1.0000 \\
 & & Fear & 220 (23,652) & .041 [.006,.106] & .4760 \\
 & & Anger & 221 (24,310) & .031 [-.009,.101] & 1.0000 \\
 & & Sadness & 221 (24,090) & .006 [-.034,.078] & 1.0000 \\
 & & Disgust & 221 (22,578) & .001 [-.036,.073] & 1.0000 \\
 & & Shame & 221 (22,366) & -.016 [-.053,.055] & 1.0000 \\
 & & Guilt & 221 (22,791) & -.009 [-.050,.065] & 1.0000 \\
Missing & Llama & Joy & 220 (24,090) & -.007 [-.054,.070] & 1.0000 \\
 & & Fear & 220 (23,652) & .036 [-.004,.107] & .4372 \\
 & & Anger & 221 (24,310) & .062 [.016,.134] & .1752 \\
 & & Sadness & 221 (24,090) & .046 [-.001,.122] & .3365 \\
 & & Disgust & 221 (22,578) & .017 [-.024,.088] & 1.0000 \\
 & & Shame & 221 (22,366) & .062 [.022,.130] & .1274 \\
 & & Guilt & 221 (22,791) & .009 [-.028,.077] & 1.0000 \\
\bottomrule\end{tabular}
\end{table*}

""" + within_overlap_table() + r"""

Original pooled $\rho$ was GPT .044 [.032,.085], $p=.0002$, and Llama .054 [.044,.090], $p=.0001$. Pairwise-missing pooled values were GPT .012 [.005,.047], $p=.2337$, and Llama .032 [.025,.067], $p=.0017$. Original pooled Human-high/model-close overlap was GPT .238 [.226,.243] versus null .250 ($p=.9990$), and Llama .232 [.220,.237] versus .250 ($p=1$). Missing-code overlaps were .252 [.238,.258] ($p=.3221$) and .243 [.230,.249] ($p=.9549$). No emotion-specific overlap exceeded its null after Holm correction. Small positive distance correspondence is not evidence that the model literally averages people; null correspondence would not by itself prove homogenization.

\subsection{Individual versus population-prototype correspondence}
\label{app:individual-prototype}

For held-out participant $p$, $r_{ind}=\rho_S(H_p,M_p)$, $r_{generic}=\rho_S(H_p,\bar M_{dev})$, and $\Delta=r_{ind}-r_{generic}$. The fixed model prototype averages 871 participant-disjoint development RDMs (799 under pairwise-missing coding); it is never fitted to test participants. Percentile CIs use 10,000 held-out-participant bootstraps conditional on the prototype. The positive-direction test uses 10,000 paired sign flips; two-sided Wilcoxon tests are secondary.

GPT mean/median $r_{ind}$ were .174 [.137,.211]/.178 [.108,.210], and $r_{generic}$ .347 [.312,.381]/.379 [.343,.419]. Mean/median $\Delta$ were -.173 [-.217,-.128]/-.155 [-.217,-.105]; fraction $\Delta>0=.300$ [.241,.364], positive one-sided $p=1.0000$, two-sided Wilcoxon $p<.0001$, $d_z=-.517$. Llama values were $r_{ind}$ .261 [.228,.294]/.281 [.206,.302], $r_{generic}$ .339 [.304,.374]/.381 [.337,.417], $\Delta$ -.078 [-.115,-.044]/-.041 [-.090,-.018], fraction .414 [.350,.477], one-sided $p=1.0000$, Wilcoxon $p<.0001$, $d_z=-.290$.

Leave-one-out test prototypes yielded GPT generic .344 and $\Delta=-.170$ [-.213,-.127] (positive fraction .291), and Llama .336 and $\Delta=-.075$ [-.111,-.041] (.414). Under pairwise-missing coding, GPT/Llama mean individualized/generic/$\Delta$ were .162/.381/-.219 [-.261,-.177] and .265/.371/-.106 [-.142,-.069]; positive fractions .227/.345 and $d_z=-.721/-.400$.

""" + prototype_missing_table() + r"""

For the pairwise-missing table, the one-sided sign-flip alternative is $E[\Delta]>0$: $p=1.0000$ for both models. Secondary two-sided Wilcoxon $p$ values were $1.0331\times10^{-17}$ (GPT) and $1.1849\times10^{-7}$ (Llama), with paired standardized means $d_z=-.721$ and $-.400$. Leave-one-out test-prototype sensitivities were GPT mean/median generic .380 [.344,.414]/.421 [.392,.460], mean/median $\Delta=-.218$ [-.258,-.177]/-.208 [-.254,-.157], fraction positive .227 [.172,.286]; and Llama .371 [.335,.406]/.420 [.372,.458], $-.106$ [-.142,-.070]/-.072 [-.135,-.041], fraction .345 [.281,.414].

Correct matching versus arbitrary mappings was +.026 ($p=.0287$) for GPT and +.048 ($p<.0001$) for Llama using 10,000 mappings. These are numerically the same estimands as the earlier mapping audit; Monte Carlo $p$ differences are not replications. Human typicality $T=\rho_S(H_p,\bar H_{dev})$ correlated with $r_{ind}$ at GPT .199 [.063,.330] and Llama .339 [.205,.461], but mapping-control $p=.1378/.1411$; correlations with $\Delta$ were -.371 [-.486,-.247] and -.337 [-.453,-.212], mapping-control $p=.2931/.2224$. The mapping-null means were .284/.413 for $r_{ind}$ and -.318/-.272 for $\Delta$.

Fixed equal-count typicality quartiles ($n=55$ each) had GPT mean typicality/$r_{ind}$/$r_{generic}$/$\Delta$: Q1 .073/.097/.069/+.028; Q2 .350/.175/.356/-.181; Q3 .489/.195/.423/-.228; Q4 .668/.230/.540/-.310. Llama: Q1 .073/.107/.051/+.056; Q2 .350/.300/.348/-.048; Q3 .489/.285/.419/-.134; Q4 .668/.351/.538/-.187. The generic prototype benefits from averaging away model-specific noise. Better generic RSA does not prove a literal internal averaging operation and must not be confused with $\rho(H,\bar H_{dev})$.

\subsection{Narrative and lexical sensitivity}
\label{app:narrative}

The three displayed word-deletion cases were selected from ten frozen cases (five strongest readable matches and five mismatches, without repeated reported emotion within each set); they do not estimate prevalence. Display chips use the emotion-specific 90th percentile of positive standardized deletion effects and cap saturation at the 99th percentile. In the fear$\rightarrow$joy bicycle case, $z_{joy}=2.435$, $z_{fear}=-1.978$, margin $-4.430$; deleting ``no'' lowered joy by 1.563 and deleting ``injuries'' by 2.006. In the shame$\rightarrow$joy case the margin was $-4.220$ and deleting ``cheering'' lowered joy by .868 while raising the relative shame evidence (shame deletion effect $-.865$). In the anger$\rightarrow$sadness relationship case the margin was $-3.569$. Deletions are nonadditive and alter syntax, position, and contextualization.

""" + strong_feature_table() + r"""

The strong-tail table is a post hoc explanatory comparison distinct from the ten word-deletion display cases. The four BH-significant contrasts were explicit target-emotion wording, prototype margin, any explicit emotion count, and intensity cues; perspective, tense, negation, length, and syntactic-complexity measures did not survive that 32-feature family.

Across 618 reports/174 participants, the final clause changed the dominant direction in 437/618=.707120 [95\% participant-bootstrap CI .671254,.742822]. The match-state table was: mismatch$\rightarrow$match 155/618=.250809, match$\rightarrow$mismatch 95/618=.153722, match retained 93/618=.150485, and mismatch retained 275/618=.444984. Mean margin change was +.413496, positive in .601942; match rose from .304207 to .401294. The separately named leave-one-final-clause ``mismatch restoration'' diagnostic was 96/618=.155340 [.128160,.182403]; it is not the 155 prefix-to-full transition.

\begin{table*}[t]
\caption{Final-clause results by reported emotion. The unit is the multi-clause report; $n$ totals 618 from 174 participant clusters. Means are development-standardized margin changes. Match/loss are report proportions. The saved per-emotion prevalence intervals use participant bootstrap resampling; exact interval rows remain in the master ledger.}
\label{tab:app-clause-emotion}
\centering\small
\begin{tabular}{lrrrrrr}
\toprule Emotion & $n$ & Mean $\Delta m$ & Flip & Prefix match & Full match & Target loss \\
\midrule
Joy & 64 & +1.1336 & .6094 & .5469 & .7344 & .1563 \\
Fear & 94 & +.3631 & .6809 & .3085 & .5106 & .1170 \\
Anger & 109 & +.4319 & .6972 & .2294 & .3761 & .1193 \\
Sadness & 78 & -.1193 & .7436 & .3846 & .2564 & .2308 \\
Disgust & 83 & +.5121 & .6627 & .2289 & .3494 & .1205 \\
Shame & 82 & +.8970 & .8049 & .1951 & .4878 & .0976 \\
Guilt & 108 & -.0461 & .7315 & .3148 & .2130 & .2315 \\
Overall & 618 & +.4135 & .7071 & .3042 & .4013 & .1537 \\
\bottomrule\end{tabular}
\end{table*}

""" + clause_emotion_interval_table() + r"""

Across 1,799 clause deletions, earlier clauses ($n=1,181$) had mean human-margin influence .073755 [.019867,.128391], versus .412961 [.290879,.530606] for final clauses ($n=618$). Controlling clause and report length, the final-position coefficient was +.300302; participant-bootstrap CI [.133379,.467192], cluster-robust CI [.129877,.470727], $p=.000553$. In the selected GPT diagnostic subset ($n=548$), final-token clause-last mean was .463907 [.319343,.603498], mean-content .215969 [.137709,.298000], and moved-first final-token .111402 [.046985,.179389], reductions of 53.4\% and 76.0\%; sign persisted in .600365 and original/moved correlation was .291466. The Llama selected subset is $n=484$, with .584634 [.462610,.704801], .283084 [.212924,.353611], and .217686 [.161892,.275032]. Selected-subset magnitudes are diagnostic, not population rates.

In nine separately adjusted feature models, explicit emotion wording predicted $\Delta m$ by +.941562, cluster-robust CI [.674898,1.208225], participant-bootstrap CI [.672393,1.208312], raw $p=4.50279\times10^{-12}$, BH $q=4.05251\times10^{-11}$. No target-loss feature survived its separate BH family; raw significance for explicit wording and narrator shift did not survive correction.

""" + terminal_feature_table() + r"""

Exploratory whole-clause naming contrasts used 5,000 participant bootstraps. Anger clauses ($n=25$) changed anger +.8549 [.3769,1.3024] and sadness -1.6968 [-2.1920,-1.2244], ending anger-dominant in 21. Sadness clauses ($n=17$) changed anger +.1935 [-.1784,.5452] and sadness -.4692 [-.9953,-.0363], ending sadness-dominant in 3 and shame-dominant in 8. Guilt clauses ($n=28$) changed guilt +.3657 [-.0533,.8063] and shame +.8236 [.3539,1.2807], ending guilt in 4 and shame in 16. Shame clauses ($n=26$) changed guilt -.0969 [-.4992,.2767] and shame +1.4893 [1.0299,1.9030], ending shame in 20. These were selected after observed confusions and are mechanism-generating, not confirmatory isolated-word effects.

\subsection{PCA-family robustness and external scope}
\label{app:robustness-external}

""" + robustness_pca_tables() + r"""
Alternative thresholds do not move every estimand in the same direction; participant RSA is more sensitive than AUROC/agreement. Family-average vector cosines were not stored and are not inferred. Reconstructed 50\% primary vectors reproduced both models' frozen vectors and RSA points to reported numerical tolerance.

Applying the unchanged GPT block-16 measurement and ISEAR development calibration to EmpatheticDialogues produced 3,498 exact matches among 10,493 listener opportunities from 5,090 dialogues (.3334). This is a descriptive domain transfer, not an ISEAR-comparable performance change. Response generation/intervention was run at $\alpha\in\{.1,.25,.5,1\}$, but blinded response-quality evaluation is incomplete; no generated-advice or user-outcome effect is reported.

\subsection{Inference, integrity, and unavailable quantities}
\label{app:integrity}

Participant is the resampling unit throughout. Primary discrimination/confusion and clause prevalence use 2,000 participant bootstraps (analysis seed 20260928 for the main evaluation); primary RSA uses 10,000 participant bootstraps and 10,000 participant permutations; post hoc geometry audits use 10,000 joint participant bootstraps/mappings; PCA/ridge reporting additions use 500 bootstraps; pair-specific named-clause intervals use 5,000. Permutation $p$ values use plus-one corrections where archived. BH families are kept separate for target terms, appraisal blocks, 63 appraisal drop-one tests, layer tests, and clause-feature outcomes; Holm covers the two dispersion hypotheses and, separately, seven within-emotion tests per model/coding and their overlap family. Undefined pairwise-missing distances are excluded; Human ties use average ranks.

No new model inference was performed for this appendix. Tests not run are reported as such: paired functional-versus-lexical method differences; family-average direction cosines; last-32/64/128-token pooling (token-level states were not retained); unconditional/refit bootstrap uncertainty for ridge and decoder models; a Human-reader benchmark; AIPsy-Affect validation (dataset unavailable); and sufficiently complete blinded response-quality or downstream-harm analysis. Controlled narrative stimuli were exported but unscored, because scoring would require new model inference. No result from another project is imported as manuscript evidence.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    appendix = appendix_text().rstrip() + "\n"
    (OUT / "APPENDIX_COMPLETE.tex").write_text(appendix, encoding="utf-8")
    write_ledger()


if __name__ == "__main__":
    main()
