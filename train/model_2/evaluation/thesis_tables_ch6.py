"""
thesis_tables_ch6.py - every number in thesis Chapter 6 (Results) comes from
here. Reads the saved evaluation files in results/ and the per-run
metrics.json, and writes LaTeX tables to papers/thesis/Chapters/table-and-figure/:

Appendix (development runs, validation, two seeds):
  tab-app-configs-cls.tex   class F1 (plain average), per class F1, mAP
  tab-app-configs-angle.tex keypoint-branch angle MAE, stretch / y / shear slopes,
                            mean shift increase (as in the acceptance rules)
  tab-app-mist.tex          the mist-map runs: state by class head / gate
                            reader, angle by map rule / angle reader, slopes, AUC
Chapter 6 (the final model = run 7 = abl_heads):
  tab-ch6-masking.tex       RQ1: without masking / without the observability loss
  tab-ch6-rq2.tex           RQ2: keypoint branch and angle reader under the probes
  tab-ch6-test.tex          RQ3: the final model on the test set (touched once)
  tab-ch6-pi5.tex           RQ3: the final model on the Raspberry Pi 5 (results/pi5_s*.json)
  tab-ch6-clsw.tex          check: class weights of the classification loss
  tab-ch6-kptw.tex          check: keypoint loss weight 20 and 10 instead of 30

Run from train/model_2:  python evaluation/thesis_tables_ch6.py
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
CKPT = HERE.parent / "checkpoint"
GEN = HERE.parent.parent.parent / "papers" / "thesis" / "Chapters" / "table-and-figure"

# (thesis label, run prefix, seeds)
CONFIGS = [
    ("stage 1", "v9_fixes_reverse_seed42", [None]),
    ("reference", "v10_ref", [42, 123]),
    ("2c", "abl_Foff", [42, 123]),
    ("3a", "abl_angle", [42, 123]),
    ("3b", "abl_tipedges", [42, 123]),
    ("3c", "abl_angleaug", [42, 123]),
    ("2G", "abl_photoaug", [42, 123]),
    ("B", "abl_mist", [42, 123]),
    ("7", "abl_heads", [42, 123]),
    ("7b", "abl_headscc", [42, 123]),
    ("5h", "abl_miststem", [42, 123]),
    ("5i-a", "abl_nomask", [42, 123]),
    ("5i-b", "abl_obsoff", [42, 123]),
    ("5j $p{=}0$", "abl_clsw_p0", [42, 123]),
    ("5j $p{=}0.5$", "abl_clsw_p05", [42, 123]),
    ("5j $p{=}1$", "abl_clsw_p1", [42, 123]),
    ("5k", "abl_kptw10", [42, 123]),
    ("5l", "abl_kptw20", [42, 123]),
]


def load(p):
    return json.load(open(p, encoding="utf-8")) if Path(p).exists() else None


def f(x, nd=2, dash="--"):
    return dash if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def run_name(prefix, seed):
    return prefix if seed is None else f"{prefix}_s{seed}"


def best_epoch_metrics(run):
    h = load(CKPT / run / "metrics.json")
    if not h:
        return {}
    comp = h["composite"]
    b = max(range(len(comp)), key=lambda i: comp[i] if comp[i] == comp[i] else -1)
    return {k: (v[b] if isinstance(v, list) and len(v) > b else None) for k, v in h.items()}


def full_map(split, run):
    """mAP@0.5 with the full score sweep of Section 5 (decode threshold 0.05),
    from map_threshold_probe.py. Training logged the AP on detections decoded at
    the 0.35 operating point only; that value is not used in the tables."""
    d = load(RES / f"map_threshold_{split}_{run}.json")
    if d is None:
        raise FileNotFoundError(f"run evaluation/map_threshold_probe.py for {run} on {split} first")
    return d["by_threshold"]["0.05"]["mAP_50"]


def six_cell(cf):
    """mean increase of the keypoint-branch MAE over the four shifts of six cells
    (outside the augmentation range, the main reading of the shift probe in Ch5)"""
    v = [d["MAE_increase_vs_base"] for k, d in (cf or {}).get("shift_rot", {}).items() if k.startswith("shift") and k.endswith("6")]
    return sum(v) / len(v) if v else None


def reader_hx(split, label):
    """horizontal stretch slope of the angle reader, from reader_hstretch.py"""
    d = load(RES / f"reader_hstretch_{split}_{label}.json")
    return d["stretch_x"]["slope"] if d else None


def rot10(cf):
    v = [d["MAE_increase_vs_base"] for k, d in (cf or {}).get("shift_rot", {}).items() if k.startswith("rot")]
    return sum(v) / len(v) if v else None


def compare_row(run):
    """the row for `run` in any compare_*.json that contains it"""
    for p in RES.glob("compare_*.json"):
        d = load(p)
        rows = d.get("rows", d) if isinstance(d, dict) else d
        if isinstance(rows, list):
            for r in rows:
                if r.get("tag") == run:
                    return r
    return {}


def hd(text):
    """a header cell in the classicthesis table style (spaced small caps);
    a cell that is only math is left as it is"""
    t = text.strip()
    return text if t.startswith("$") and t.endswith("$") else f"\\spacedlowsmallcaps{{{text}}}"


def grouped_header(first, groups, seeds=(42, 123)):
    """two header lines: the run names over their seed columns, then the seeds"""
    n = len(seeds)
    top = "  " + first + " & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{hd(g)}}}" for g in groups) + " \\\\"
    rules = "  " + "".join(f"\\cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}" for i in range(len(groups)))
    sub = "  " + hd("seed") + " & " + " & ".join(str(s) for _ in groups for s in seeds) + " \\\\"
    return "\n".join([top, rules, sub])


# short captions for the list of tables, by label
SHORT_CAPTIONS = {
    "tab:app-configs-cls": "Detection and state of the development runs",
    "tab:app-configs-angle": "Keypoint branch angle of the development runs",
    "tab:app-mist": "Development runs with the mist map",
    "tab:ch6-masking": "The final model without masking or observability loss",
    "tab:ch6-clsw": "Class weights of the classification loss",
    "tab:ch6-rq2": "The two angle outputs under the probes",
    "tab:ch6-kptw": "Keypoint loss weight",
    "tab:ch6-test": "Detection and state on the test set",
    "tab:ch6-test-angle": "Apex angle error on the test set",
    "tab:ch6-test-errors": "Where the errors on the test set fall",
    "tab:ch6-pi5": "The final model on the Raspberry Pi 5",
}


def latex_table(title, caption, label, colspec, header, body, footer=(), stretch=1.15, size="\\footnotesize", note="", fit=False, placement="htbp", scale=None):
    """fit=True shrinks the tabular to the text width when it is wider.
    header: a list of cells, or a ready string of header lines (grouped_header)
    body / footer: lists of cells; a string is written as it is (a rule, or a line with \\multicolumn)"""
    rows = [r for r in list(body) + list(footer) if not isinstance(r, str)]
    widths = [max(len(r[i]) for r in rows) if rows else 0 for i in range(len(rows[0]))]

    def line(r, pad=True):
        if isinstance(r, str):
            return "  " + r
        cells = [c.ljust(w) if pad else c for c, w in zip(r, widths)]
        return "  " + " & ".join(cells).rstrip() + " \\\\"

    out = [f"% {title}", "% Generated by train/model_2/evaluation/thesis_tables_ch6.py from results/*.json;",
           "% do not edit by hand - rerun the script instead.",
           "\\begin{table}[" + placement + "]", "\\centering", size, f"\\renewcommand{{\\arraystretch}}{{{stretch}}}",
           f"\\caption[{SHORT_CAPTIONS[label]}]{{{caption}}}", f"\\label{{{label}}}"]
    if scale:
        out += ["\\makebox[\\linewidth][c]{\\resizebox{" + str(scale) + "\\linewidth}{!}{%"]
    elif fit:
        out += ["\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%"]
    # header cells written as \textbf{...} are set in the classicthesis table style
    head = header if isinstance(header, str) else line(header, pad=False).replace("\\textbf{", "\\spacedlowsmallcaps{")
    out += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", head, "\\midrule", *[line(r) for r in body]]
    if footer:
        out += ["\\midrule", *[line(r) for r in footer]]
    out += ["\\bottomrule", "\\end{tabular}" + ("}}" if scale else "}" if fit else "")]
    if note:
        out += ["\\\\[0.3em]", "\\parbox{\\linewidth}{\\footnotesize " + note + "}"]
    out += ["\\end{table}", ""]
    return "\n".join(out)


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    summary = {}
    lookup_val = load(RES / "angle_reeval_val_abl_heads_s42.json")["pixel"]["baselines"]["position_lookup"]["MAE_deg"]

    # ---------------- appendix tables A / B: every development run on validation ----------------
    body_cls, body_ang = [], []
    dev = CONFIGS[:[c[0] for c in CONFIGS].index("5h") + 1]  # development runs only; 5i-5k are in Chapter 6
    for label, prefix, seeds in dev:
        for seed in seeds:
            run = run_name(prefix, seed)
            cr = compare_row(run)
            cf = load(RES / f"counterfactual_val_{run}.json") or {}
            bm = best_epoch_metrics(run)
            pc = cr.get("per_class_F1", {})
            row = {
                "run": label, "seed": "--" if seed is None else str(seed),
                "cls_f1": cr.get("cls_f1"), "no_spray": pc.get("nozzle_no_spray"), "facing": pc.get("nozzle_facing"),
                "spray": pc.get("nozzle_spray"), "mAP": None if seed is None else full_map("val", run),
                "mae": cr.get("angle_mae"), "n": cr.get("angle_n"), "stretch": cr.get("stretch_slope"),
                "slope_y": (cf.get("stretch_y") or {}).get("slope"), "shear": (cf.get("shear") or {}).get("slope"),
                "shift": cr.get("shift_increase"),
            }
            summary[run] = row
            body_cls.append([label, row["seed"], f(row["cls_f1"]), f(row["spray"]), f(row["no_spray"]), f(row["facing"]), f(row["mAP"])])
            body_ang.append([label, row["seed"], f(row["mae"]) + ("" if row["n"] is None else f" ({row['n']})"),
                             f(row["stretch"]), f(row["slope_y"]), f(row["shear"]), f(row["shift"], 1)])
    # big is left out of the table; the note says why, from the same compare rows
    big = [(label, seed, compare_row(run_name(prefix, seed)).get("per_class_F1", {}).get("big_nozzle"))
           for label, prefix, seeds in dev for seed in seeds]
    low = [(l, s, v) for l, s, v in big if v is not None and round(v, 2) < 1.0]
    big_note = ("\\texttt{big} is not listed, since its $F_1$ is $1.00$ in every run"
                + (" except " + ", ".join(f"run {l}{'' if s is None else f' on seed ${s}$'}, at ${v:.2f}$" for l, s, v in low) if low else "")
                + ".")
    (GEN / "tab-app-configs-cls.tex").write_text(latex_table(
        "Appendix: detection and state of every development run on the validation set",
        "Detection and state of every development run on the validation set, both seeds, read from the grid heads. "
        "$F_1^{\\text{cls}}$ is the plain average of the classification head over the four classes, and the next three "
        "columns give its $F_1$ for each small lance class.",
        "tab:app-configs-cls", "llrrrrr",
        ["\\textbf{Run}", "\\textbf{seed}", "$F_1^{\\text{cls}}$", "\\texttt{spray}", "\\texttt{no\\_spray}", "\\texttt{facing}", "$\\mathrm{mAP}_{50}$"],
        body_cls, note="stage 1 was scored with the same instance level metrics. Its $\\mathrm{mAP}_{50}$ was logged under the "
                       "earlier metric definition and is not listed." + "\\\\ " +
                       "Two seeds of the same run can differ by a collapse of \\texttt{no\\_spray} to zero, which is a "
                       "property of the training." + "\\\\ " + big_note), encoding="utf-8")
    (GEN / "tab-app-configs-angle.tex").write_text(latex_table(
        "Appendix: the angle of every development run on the validation set, keypoint branch",
        "The apex angle read from the keypoint branch, every development run on the validation set, both seeds. "
        "{\\danny MAE$_\\psi$} in degrees on the $N$ tip matched spray lances out of $96$; the position lookup baseline is "
        f"{f(lookup_val)}$^{{\\circ}}$ on the same lances.",
        "tab:app-configs-angle", "llrrrrr",
        ["\\textbf{Run}", "\\textbf{seed}", "{\\danny MAE$_\\psi$} ($N$)", "\\textbf{horizontal}", "\\textbf{vertical}",
         "\\textbf{shear}", "\\textbf{shift} $+^{\\circ}$"],
        body_ang, note="\\spacedlowsmallcaps{horizontal}, \\spacedlowsmallcaps{vertical}, \\spacedlowsmallcaps{shear}: slope of the "
                       "change of the prediction on the change of the true angle under the horizontal stretch, the vertical "
                       "stretch, and the shear. The vertical stretch and the shear were added at run 3b and were not run on the "
                       "earlier checkpoints." + "\\\\ " +
                       "\\spacedlowsmallcaps{shift}: mean increase of {\\danny MAE$_\\psi$} over all twelve shifts of two, four, and six "
                       "cells.", fit=True), encoding="utf-8")

    # ---------------- table 2: the mist-map configurations on validation ----------------
    # transposed: one column per run x seed, one row per reader and quantity
    cols, runs = [], []
    for label, prefix in [("B", "abl_mist"), ("7", "abl_heads"), ("7b", "abl_headscc"), ("5h", "abl_miststem")]:
        for seed in (42, 123):
            run = f"{prefix}_s{seed}"
            m = load(RES / f"mist_hybrid_val_{run}.json")
            if not m:
                continue
            st, an, pr, ph = m["state"], m["angle"], m["probes"], m.get("probes_learned_head") or {}
            lh = st.get("learned_head") or {}
            cols.append({
                "head": f"{label} / {seed}",
                "ns_head": f(st["nozzle_no_spray"]["F1_network"]), "ns_gate": f(st["nozzle_no_spray"]["F1_mist"]), "ns_learn": f(lh.get("nozzle_no_spray")),
                "fa_head": f(st["nozzle_facing"]["F1_network"]), "fa_gate": f(st["nozzle_facing"]["F1_mist"]), "fa_learn": f(lh.get("nozzle_facing")),
                "mae_rule": f(an.get("MAE_from_map_deg")), "sy_rule": f(pr["stretch_y"]["slope"]), "sh_rule": f(pr["shear"]["slope"]),
                "mae_learn": f(an.get("MAE_learned_head_deg")), "sx_learn": f(reader_hx("val", run)) if ph else "--",
                "sy_learn": f(ph["stretch_y"]["slope"]) if ph else "--", "sh_learn": f(ph["shear"]["slope"]) if ph else "--",
                "auc": f(m["map"]["AUC_vs_pseudo_in_roi"], 3)})
            summary[run + "_mist"] = {"state": st, "angle": an, "probes": pr, "probes_head": ph, "map_auc": m["map"]["AUC_vs_pseudo_in_roi"]}
    rows_spec = [
        ("map AUC vs target in $\\mathcal{R}$", "auc"),
        ("\\texttt{no\\_spray} $F_1$, class head", "ns_head"), ("\\texttt{no\\_spray} $F_1$, mist gate", "ns_gate"),
        ("\\texttt{facing} $F_1$, class head", "fa_head"), ("\\texttt{facing} $F_1$, mist gate", "fa_gate"),
        ("{\\danny MAE$_\\psi$}, map rule", "mae_rule"), ("vertical stretch slope, map rule", "sy_rule"), ("shear slope, map rule", "sh_rule"),
        ("{\\danny MAE$_\\psi$}, angle reader", "mae_learn"), ("horizontal stretch slope, angle reader", "sx_learn"),
        ("vertical stretch slope, angle reader", "sy_learn"), ("shear slope, angle reader", "sh_learn"),
    ]
    body = [[name] + [c[key] for c in cols] for name, key in rows_spec]
    header = grouped_header(hd("run"), list(dict.fromkeys(c["head"].split(" / ")[0] for c in cols)))
    note = ("State: $F_1$ of the classification head and the mist gate, on the same detections." + "\\\\ " +
            "Angle: {\\danny MAE$_\\psi$} in degrees with the cone edges read from the map by a fixed threshold rule and by the "
            "angle reader, with slopes from the vertical stretch and shear probes." + "\\\\ " +
            "map AUC: agreement of the map with the automatic target inside $\\mathcal{R}$." + "\\\\ " +
            "Run 7 is the final model.")
    (GEN / "tab-app-mist.tex").write_text(latex_table(
        "Appendix: readers of the mist map on the validation set",
        "The development runs with the mist map on the validation set: the nozzle lance state and the angle read from the map, next to "
        "the same quantities read from the grid heads of the same network.",
        "tab:app-mist", "l" + "r" * len(cols), header, body, size="\\footnotesize", note=note, fit=True), encoding="utf-8")

    # ---------------- table 2b: removing the masking / the observability loss (5i) ----------------
    cols = []
    for label, prefix in [("final", "abl_heads"), ("no mask", "abl_nomask"), ("no obs.", "abl_obsoff")]:
        for seed in (42, 123):
            run = f"{prefix}_s{seed}"
            m, cr, bm = load(RES / f"mist_hybrid_val_{run}.json"), compare_row(run), best_epoch_metrics(run)
            cf = load(RES / f"counterfactual_val_{run}.json")
            if not m or not cr:
                continue
            st, an, ph = m["state"], m["angle"], m.get("probes_learned_head") or {}
            n = cr.get("angle_n")
            cols.append({
                "head": f"{label} / {seed}",
                # OBS_OFF keeps the output channel untrained, so its score is not a result of the model
                "obs": "--" if prefix == "abl_obsoff" else f(cr.get("obs_f1")),
                "tips": "--" if n is None else f"{n}",
                "mae_kpt": f(cr.get("angle_mae")), "mae_learn": f(an.get("MAE_learned_head_deg")),
                "sx_kpt": f(cf["stretch"]["slope"]), "sy_kpt": f(cf["stretch_y"]["slope"]), "sh_kpt": f(cf["shear"]["slope"]),
                "shift": f(six_cell(cf)), "sx_learn": f(reader_hx("val", run)),
                "sy_learn": f(ph["stretch_y"]["slope"]) if ph else "--", "sh_learn": f(ph["shear"]["slope"]) if ph else "--",
                "ns_head": f(st["nozzle_no_spray"]["F1_network"]), "fa_head": f(st["nozzle_facing"]["F1_network"]),
                "sp_head": f(st["nozzle_spray"]["F1_network"]), "cls": f(cr.get("cls_f1")), "map": f(full_map("val", run)),
                "ns_gate": f(st["nozzle_no_spray"]["F1_mist"]), "fa_gate": f(st["nozzle_facing"]["F1_mist"]),
                "auc": f(m["map"]["AUC_vs_pseudo_in_roi"], 3)})
            summary[run + "_5i"] = {"compare": cr, "state": st, "angle": an, "probes_head": ph, "mAP_50": full_map("val", run)}
    rows_spec = [
        ("$F_1^{\\text{obs}}$", "obs"), ("spray lances paired, of 96", "tips"),
        # RQ1 reads the accuracy of the two angle outputs; their probe slopes belong to RQ2 (tab-ch6-rq2)
        ("{\\danny MAE$_\\psi$}, keypoint branch", "mae_kpt"), ("{\\danny MAE$_\\psi$}, angle reader", "mae_learn"),
        ("\\texttt{no\\_spray} $F_1$, class head", "ns_head"), ("\\texttt{facing} $F_1$, class head", "fa_head"),
        ("\\texttt{spray} $F_1$, class head", "sp_head"), ("$F_1^{\\text{cls}}$", "cls"), ("$\\mathrm{mAP}_{50}$", "map"),
        ("\\texttt{no\\_spray} $F_1$, mist gate", "ns_gate"), ("\\texttt{facing} $F_1$, mist gate", "fa_gate"),
    ]
    body = [[name] + [c[key] for c in cols] for name, key in rows_spec]
    header = grouped_header(hd("run"), list(dict.fromkeys(c["head"].split(" / ")[0] for c in cols)))
    note = ("\\spacedlowsmallcaps{final}: the final model.\\\\ "
            "\\spacedlowsmallcaps{no mask}: the class determined masking switched off.\\\\ "
            "\\spacedlowsmallcaps{no obs.}: the observability loss removed, so $F_1^{\\text{obs}}$ is not reported.\\\\ "
            "Class head $F_1$ per class on the same detections as the mist gate.")
    (GEN / "tab-ch6-masking.tex").write_text(latex_table(
        "Chapter 6: removing the masking and the observability loss from the final model (RQ1)",
        "The final model and two variants without a part of it, on the validation set and both seeds, {\\danny one without the class determined "
        "masking and one without the observability loss}.",
        "tab:ch6-masking", "l" + "r" * len(cols), header, body, size="\\footnotesize", note=note, fit=True), encoding="utf-8")

    # ---------------- table 2c: class weights of the classification loss (5j) ----------------
    cols = []
    for label, prefix in [("final", "abl_heads"), ("$p{=}0$", "abl_clsw_p0"), ("$p{=}0.5$", "abl_clsw_p05"), ("$p{=}1$", "abl_clsw_p1")]:
        for seed in (42, 123):
            run = f"{prefix}_s{seed}"
            m, cr, bm = load(RES / f"mist_hybrid_val_{run}.json"), compare_row(run), best_epoch_metrics(run)
            if not m or not cr:
                continue
            st, an, ph = m["state"], m["angle"], m.get("probes_learned_head") or {}
            n = cr.get("angle_n")
            cols.append({
                "head": f"{label} / {seed}",
                "cls": f(cr.get("cls_f1")), "sp_head": f(st["nozzle_spray"]["F1_network"]),
                "ns_head": f(st["nozzle_no_spray"]["F1_network"]), "fa_head": f(st["nozzle_facing"]["F1_network"]),
                "map": f(full_map("val", run)), "tips": "--" if n is None else f"{n}",
                "mae_kpt": f(cr.get("angle_mae")), "mae_learn": f(an.get("MAE_learned_head_deg")),
                "sx_learn": f(reader_hx("val", run)),
                "sy_learn": f(ph["stretch_y"]["slope"]) if ph else "--", "sh_learn": f(ph["shear"]["slope"]) if ph else "--",
                "ns_gate": f(st["nozzle_no_spray"]["F1_mist"]), "auc": f(m["map"]["AUC_vs_pseudo_in_roi"], 3)})
            summary[run + "_5j"] = {"compare": cr, "state": st, "angle": an, "probes_head": ph, "mAP_50": full_map("val", run)}
    rows_spec = [
        ("$F_1^{\\text{cls}}$", "cls"), ("\\texttt{spray} $F_1$, class head", "sp_head"),
        ("\\texttt{no\\_spray} $F_1$, class head", "ns_head"), ("\\texttt{facing} $F_1$, class head", "fa_head"),
        ("$\\mathrm{mAP}_{50}$", "map"), ("spray lances paired, of 96", "tips"),
        ("{\\danny MAE$_\\psi$}, keypoint branch", "mae_kpt"), ("{\\danny MAE$_\\psi$}, angle reader", "mae_learn"),
        ("horizontal stretch slope, angle reader", "sx_learn"),
        ("vertical stretch slope, angle reader", "sy_learn"), ("shear slope, angle reader", "sh_learn"),
        ("\\texttt{no\\_spray} $F_1$, mist gate", "ns_gate"),
    ]
    body = [[name] + [c[key] for c in cols] for name, key in rows_spec]
    header = grouped_header(hd("run"), list(dict.fromkeys(c["head"].split(" / ")[0] for c in cols)))
    note = ("$\\lambda_k = (n_{\\max} / n_k)^p$ over the positive training cells, in the order (\\texttt{spray}, "
            "\\texttt{no\\_spray}, \\texttt{facing}, \\texttt{big})." + "\\\\ " +
            "Everything else is as in the final model.")
    (GEN / "tab-ch6-clsw.tex").write_text(latex_table(
        "Chapter 6: class weights of the classification loss swept as a one parameter family",
        "The class weight of the classification loss swept as one family on the validation set, both seeds, next to the final model.",
        "tab:ch6-clsw", "l" + "r" * len(cols), header, body, size="\\footnotesize", note=note, fit=True), encoding="utf-8")

    # ---------------- table RQ2: the angle readers of the final model under the probes ----------------
    rq2 = []
    for head, ar_f, cf_f, mh_f, hx in [
        ("val / 42", "angle_reeval_val_abl_heads_s42.json", "counterfactual_val_abl_heads_s42.json", "mist_hybrid_val_abl_heads_s42.json",
         reader_hx("val", "abl_heads_s42")),
        ("val / 123", "angle_reeval_val_abl_heads_s123.json", "counterfactual_val_abl_heads_s123.json", "mist_hybrid_val_abl_heads_s123.json",
         reader_hx("val", "abl_heads_s123")),
        ("test / 42", "angle_reeval_test_final_heads_s42.json", "counterfactual_test_abl_heads_s42.json", "mist_hybrid_test_final_heads_s42.json",
         reader_hx("test", "final_heads_s42")),
        ("test / 123", "angle_reeval_test_final_heads_s123.json", "counterfactual_test_abl_heads_s123.json", "mist_hybrid_test_final_heads_s123.json",
         reader_hx("test", "final_heads_s123")),
    ]:
        ar, cf, mh = load(RES / ar_f)["pixel"], load(RES / cf_f), load(RES / mh_f)
        k = [x for x in ar if x != "baselines"][0]
        mc, mb = cf["mask"]["cone"], cf["mask"]["body"]
        ph = mh.get("probes_learned_head") or {}
        rq2.append({
            "head": head,
            "const": f(ar["baselines"]["constant"]["MAE_deg"]),
            "lookup": f(ar["baselines"]["position_lookup"]["MAE_deg"]),
            "k_mae": f"{f(ar[k]['MAE_deg'])} ({ar[k]['N']})",
            "k_sx": f(cf["stretch"]["slope"]), "k_sy": f(cf["stretch_y"]["slope"]), "k_sh": f(cf["shear"]["slope"]),
            "k_shift": f(six_cell(cf)), "k_rot": f(rot10(cf)),
            "k_cone": f"{mc['N'] - mc['no_spray_prediction']} / {mc['N']}, {f(mc['MAE_when_predicted'])}",
            "k_body": f"{mb['N'] - mb['no_spray_prediction']} / {mb['N']}, {f(mb['MAE_when_predicted'])}",
            "h_mae": f(mh["angle"]["MAE_learned_head_deg"]), "h_sx": f(hx),
            "h_sy": f(ph["stretch_y"]["slope"]), "h_sh": f(ph["shear"]["slope"]),
        })
        summary["rq2_" + head.replace(" / ", "_")] = rq2[-1]
    rows_spec = [
        ("constant baseline, {\\danny MAE$_\\psi$}", "const"),
        ("position lookup baseline, {\\danny MAE$_\\psi$}", "lookup"),
        ("keypoint branch, {\\danny MAE$_\\psi$} ($N$)", "k_mae"),
        ("\\quad slope, horizontal stretch", "k_sx"), ("\\quad slope, vertical stretch", "k_sy"), ("\\quad slope, shear", "k_sh"),
        ("\\quad error increase, shift of six cells", "k_shift"), ("\\quad error increase, rotation", "k_rot"),
        ("\\quad cone hidden: kept, {\\danny MAE$_\\psi$}", "k_cone"), ("\\quad body hidden: kept, {\\danny MAE$_\\psi$}", "k_body"),
        ("angle reader, {\\danny MAE$_\\psi$}", "h_mae"), ("\\quad slope, horizontal stretch", "h_sx"),
        ("\\quad slope, vertical stretch", "h_sy"), ("\\quad slope, shear", "h_sh"),
    ]
    body = [[name] + [c[key] for c in rq2] for name, key in rows_spec]
    # two header lines as in tab-ch6-masking: the set over its seed columns, then the seeds
    assert [c["head"] for c in rq2] == ["val / 42", "val / 123", "test / 42", "test / 123"]
    header = grouped_header(hd("output"), ["validation", "test"])
    note = ("Errors in degrees." + "\\\\ " +
            "Slope: change of the prediction per degree of change of the true angle." + "\\\\ " +
            "Error increase: mean over the four shifts of six cells, and over the two rotations." + "\\\\ " +
            "Hidden cone or body: lances that keep a \\texttt{spray} prediction out of all lances, and their error." + "\\\\ " +
            "The angle reader was not run on the shift, rotation, and mask probes.")
    (GEN / "tab-ch6-rq2.tex").write_text(latex_table(
        "Chapter 6: the two angle outputs of the final model under the probes (RQ2)",
        "The two angle outputs of the final model under the probes, next to the constant and position lookup baselines, "
        "on the validation and the test set, for both seeds.",
        "tab:ch6-rq2", "lrrrr", header, body, size="\\footnotesize", note=note, fit=True), encoding="utf-8")

    # ---------------- table 2d: keypoint loss weight 30 -> 20 (5l) and 30 -> 10 (5k) ----------------
    cols = []
    for label, prefix in [("final", "abl_heads"), ("$\\lambda_{\\text{kpt}}{=}20$", "abl_kptw20"), ("$\\lambda_{\\text{kpt}}{=}10$", "abl_kptw10")]:
        for seed in (42, 123):
            run = f"{prefix}_s{seed}"
            m, cr = load(RES / f"mist_hybrid_val_{run}.json"), compare_row(run)
            cf = load(RES / f"counterfactual_val_{run}.json")
            if not m or not cr:
                continue
            st, an, ph = m["state"], m["angle"], m.get("probes_learned_head") or {}
            n = cr.get("angle_n")
            cols.append({
                "head": f"{label} / {seed}",
                "cls": f(cr.get("cls_f1")), "sp_head": f(st["nozzle_spray"]["F1_network"]),
                "ns_head": f(st["nozzle_no_spray"]["F1_network"]), "fa_head": f(st["nozzle_facing"]["F1_network"]),
                "map": f(full_map("val", run)), "tips": "--" if n is None else f"{n}",
                "mae_kpt": f(cr.get("angle_mae")), "shift": f(six_cell(cf)),
                "mae_learn": f(an.get("MAE_learned_head_deg")), "sx_learn": f(reader_hx("val", run)),
                "sy_learn": f(ph["stretch_y"]["slope"]) if ph else "--", "sh_learn": f(ph["shear"]["slope"]) if ph else "--",
                "ns_gate": f(st["nozzle_no_spray"]["F1_mist"]), "auc": f(m["map"]["AUC_vs_pseudo_in_roi"], 3)})
            summary[run + "_5k"] = {"compare": cr, "state": st, "angle": an, "probes_head": ph, "mAP_50": full_map("val", run)}
    rows_spec = [
        ("$F_1^{\\text{cls}}$", "cls"), ("\\texttt{spray} $F_1$, class head", "sp_head"),
        ("\\texttt{no\\_spray} $F_1$, class head", "ns_head"), ("\\texttt{facing} $F_1$, class head", "fa_head"),
        ("$\\mathrm{mAP}_{50}$", "map"), ("spray lances paired, of 96", "tips"),
        ("{\\danny MAE$_\\psi$}, keypoint branch", "mae_kpt"), ("{\\danny error increase, shift of six cells, keypoint branch}", "shift"),
        ("{\\danny MAE$_\\psi$}, angle reader", "mae_learn"), ("horizontal stretch slope, angle reader", "sx_learn"),
        ("vertical stretch slope, angle reader", "sy_learn"), ("shear slope, angle reader", "sh_learn"),
        ("\\texttt{no\\_spray} $F_1$, mist gate", "ns_gate"),
    ]
    body = [[name] + [c[key] for c in cols] for name, key in rows_spec]
    header = grouped_header(hd("run"), list(dict.fromkeys(c["head"].split(" / ")[0] for c in cols)))
    note = ("Shift of six cells: mean increase of the keypoint branch {\\danny MAE$_\\psi$} over the four shifts of six cells.")
    have20 = any(c["head"].startswith("$\\lambda_{\\text{kpt}}{=}20$") for c in cols)
    (GEN / "tab-ch6-kptw.tex").write_text(latex_table(
        "Chapter 6: keypoint loss weight lowered from 30 to 20 and to 10" if have20 else "Chapter 6: keypoint loss weight lowered from 30 to 10",
        ("The keypoint loss weight lowered from $30$ to $20$ and to $10$ on the validation set, both seeds, next to the final model." if have20 else
         "The keypoint loss weight lowered from $30$ to $10$ on the validation set, both seeds, next to the final model."),
        "tab:ch6-kptw", "l" + "r" * len(cols), header, body, size="\\footnotesize", note=note, fit=True), encoding="utf-8")

    # ---------------- table 3: the final model on test, both seeds ----------------
    # Seed 42 was scored on test first (step 5); seed 123 of the same design was added on 2026-09-27
    # (run_step5_s123.sh) so that the test results show the seed spread. Nothing is chosen from test.
    SEEDS = (42, 123)
    T = {}
    for sd in SEEDS:
        tag = f"final_heads_s{sd}"
        ar_ = load(RES / f"angle_reeval_test_{tag}.json")["pixel"]
        mh_ = load(RES / f"mist_hybrid_test_{tag}.json")
        T[sd] = {"tm": load(RES / f"metrics_test_{tag}.json"), "ar": ar_, "key": [k for k in ar_ if k != "baselines"][0],
                 "mh": mh_, "ap": load(RES / f"map_threshold_test_{tag}.json")["by_threshold"]["0.05"],
                 "abl": load(RES / f"angle_by_lance_test_{tag}.json"), "err": load(RES / f"test_errors_{tag}.json"), "gate": load(RES / f"map_gate_test_{tag}.json")["reported_state"]}
    classes = ("nozzle_spray", "nozzle_no_spray", "nozzle_facing", "big_nozzle")
    body = []
    for name, fn in [
        ("reported state, $F_1$", lambda d: [d["tm"]["overall"]["det_F1_nozzle_spray"], d["mh"]["state"]["nozzle_no_spray"]["F1_mist"],
                                             d["mh"]["state"]["nozzle_facing"]["F1_mist"], d["tm"]["overall"]["det_F1_big_nozzle"]]),
        ("reported state, $\\mathrm{AP}_{50}$", lambda d: [d["gate"]["per_class"][k] for k in classes]),
        ("classification head, $F_1$", lambda d: [d["tm"]["overall"][f"det_F1_{k}"] for k in classes]),
        ("classification head, $\\mathrm{AP}_{50}$", lambda d: [d["ap"]["per_class"][k] for k in classes]),
    ]:
        for i, sd in enumerate(SEEDS):
            vals = fn(T[sd])
            body.append([name if i == 0 else "", str(sd)] + [f(v, 3) for v in vals] + [f(sum(vals) / 4, 3)])
    header = ["\\textbf{output}", "\\textbf{seed}", "\\texttt{spray}", "\\texttt{no\\_spray}", "\\texttt{facing}", "\\texttt{big}", "\\textbf{Mean}"]
    note = ("Reported state: \\texttt{spray} and \\texttt{big} from the classification head, \\texttt{no\\_spray} and \\texttt{facing} from the mist gate." + "\\\\ " +
            "Classification head: the same detections before the mist gate, so the difference to the reported state is what the mist gate corrects." + "\\\\ " +
            "Mean: the plain average over the four classes, which for the $\\mathrm{AP}_{50}$ rows is $\\mathrm{mAP}_{50}$." + "\\\\ " +
            f"Observability head: $F_1^{{\\text{{obs}}}} = {f(T[42]['tm']['overall']['obs_f1'], 3)}$ and {f(T[123]['tm']['overall']['obs_f1'], 3)}.")
    (GEN / "tab-ch6-test.tex").write_text(latex_table(
        "Chapter 6: the final model on the test set, detection and state (RQ3)",
        "Detection and nozzle lance state per class on the test set, as reported by the final model and by its "
        "classification head alone, for both seeds.",
        "tab:ch6-test", "llrrrrr", header, body, note=note, fit=True), encoding="utf-8")

    rows = {"reader": [], "kpt": [], "lookup": [], "const": []}
    for sd in SEEDS:
        d = T[sd]
        an_, ar_, k_, kept = d["mh"]["angle"], d["ar"], d["key"], d["abl"]["score_at_least_operating"]
        assert abs(d["abl"]["all"]["MAE_reader"] - an_["MAE_learned_head_deg"]) < 1e-6 and d["abl"]["all"]["N"] == ar_[k_]["N"]
        assert abs(d["abl"]["all"]["MAE_lookup"] - ar_["baselines"]["position_lookup"]["MAE_deg"]) < 1e-6
        rows["reader"] += [f(an_["MAE_learned_head_deg"]), f(kept["MAE_reader"])]
        rows["kpt"] += [f(ar_[k_]["MAE_deg"]), f(kept["MAE_kpt"])]
        rows["lookup"] += [f(ar_["baselines"]["position_lookup"]["MAE_deg"]), f(kept["MAE_lookup"])]
        rows["const"] += [f(ar_["baselines"]["constant"]["MAE_deg"]), "--"]
    body = [["angle reader (reported)"] + rows["reader"], ["keypoint branch"] + rows["kpt"],
            ["position lookup baseline"] + rows["lookup"], ["constant baseline"] + rows["const"]]
    top = "  " + hd("output") + " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{hd(f'seed {sd}')}}}" for sd in SEEDS) + " \\\\"
    rules = "  \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}"
    sub = "  & " + " & ".join(f"{hd('all')} (${T[sd]['ar'][T[sd]['key']]['N']}$) & {hd('kept')} (${T[sd]['abl']['score_at_least_operating']['N']}$)"
                              for sd in SEEDS) + " \\\\"
    note = ("{\\danny MAE$_\\psi$} in degrees." + "\\\\ " +
            "\\spacedlowsmallcaps{all}: every \\texttt{spray} instance of the test set {\\danny with annotated keypoints}." + "\\\\ " +
            "\\spacedlowsmallcaps{kept}: the instances whose paired detection reaches the operating threshold, which "
            "are the ones the deployed system reports.")
    (GEN / "tab-ch6-test-angle.tex").write_text(latex_table(
        "Chapter 6: the final model on the test set, apex angle (RQ3)",
        "Apex angle error of the two angle outputs of the final model and of the two baselines on the test set, "
        "on all \\texttt{spray} instances and on the reported ones, for both seeds.",
        "tab:ch6-test-angle", "lrrrr", "\n".join([top, rules, sub]), body, note=note), encoding="utf-8")

    # ---------------- table 3c: where the errors of the final model on the test set fall ----------------
    order = [("CAM1 V2", "big"), ("CAM1 V2", "no_spray"), ("CAM1 V2", "facing"),
             ("CAM1 V3", "big"), ("CAM1 V3", "no_spray"), ("CAM1 V3", "facing"),
             ("CAM3 V2", "big"), ("CAM3 V2", "facing"), ("CAM3 V2", "spray"),
             ("CAM3 V3", "big"), ("CAM3 V3", "no_spray")]

    def split(e):
        # wrong state: every "reported <other class>" outcome of a detected lance
        return e.get("correct", 0), e.get("not detected", 0), sum(v for k, v in e.items() if k.startswith("reported"))

    def wrong_cell(e):
        # the wrong states with the state that was reported, e.g. "105 big, 70 facing"
        parts = [f"{k.split()[1].replace('_', chr(92) + '_')} {v}" for k, v in sorted(e.items(), key=lambda kv: -kv[1]) if k.startswith("reported")]
        return ", ".join(parts) if parts else "0"

    body, kinds = [], {sd: set() for sd in SEEDS}
    for rec, cls in order:
        cells, n = [], None
        for sd in SEEDS:
            e = T[sd]["err"].get(f"{rec} {cls}", {})
            c, nd, ws = split(e)
            n = c + nd + ws if n is None else n
            assert n == c + nd + ws, (rec, cls)
            cells += [str(c), str(nd), wrong_cell(e)]
            kinds[sd] |= {f"\\texttt{{{cls.replace('_', chr(92) + '_')}}} as \\texttt{{{k.split()[1].replace('_', chr(92) + '_')}}}"
                          for k in e if k.startswith("reported")}
        body.append([rec, "\\texttt{" + cls.replace("_", "\\_") + "}", str(n)] + cells)
    footer_cells = []
    for sd in SEEDS:
        tt = T[sd]["err"]["total"]
        footer_cells += [str(tt.get("correct", 0)), str(tt.get("not detected", 0)), wrong_cell(tt)]
    footer = [["total", "", str(sum(int(r[2]) for r in body))] + footer_cells]
    top = ("  " + hd("recording") + " & " + hd("class") + " & " + hd("lances") + " & "
           + " & ".join(f"\\multicolumn{{3}}{{c}}{{{hd(f'seed {sd}')}}}" for sd in SEEDS) + " \\\\")
    rules = "  \\cmidrule(lr){4-6}\\cmidrule(lr){7-9}"
    sub = "  & & & " + " & ".join(f"{hd('correct')} & {hd('missed')} & {hd('wrong')}" for _ in SEEDS) + " \\\\"
    note = ("Every annotated lance of the test set falls into one of the three columns of each seed." + "\\\\ " +
            "\\spacedlowsmallcaps{correct}: detected with the correct reported state." + "\\\\ " +
            "\\spacedlowsmallcaps{missed}: not detected." + "\\\\ " +
            "\\spacedlowsmallcaps{wrong}: detected with a wrong reported state, given with the state that was reported." + "\\\\ " +
            "Detections that match no annotated lance: "
            + ", ".join(f"{T[sd]['err']['total'].get('unmatched detection', 0)} for seed {sd}" for sd in SEEDS) + ".")
    (GEN / "tab-ch6-test-errors.tex").write_text(latex_table(
        "Chapter 6: where the errors of the final model on the test set fall (RQ3)",
        "The annotated lances of the test set by recording and class, and how the final model handles them, for both seeds.",
        "tab:ch6-test-errors", "lll" + "r" * 6, "\n".join([top, rules, sub]), body, footer=footer, note=note, fit=True, stretch=1.0, placement="p", scale=1.2),
        encoding="utf-8")
    summary["test_errors"] = {sd: T[sd]["err"] for sd in SEEDS}

    # ---------------- table 3d: the final model on the Raspberry Pi 5 (RQ3) ----------------
    P = {sd: load(RES / f"pi5_s{sd}.json") for sd in SEEDS}
    if all(P.values()):
        # upper part: one column per statistic of the time per frame, grouped by seed;
        # lower part: one value per seed across its three columns, the statistic named in the row
        stats = ("mean", "median", "p95")
        stages = [("Frame decoding", "decode"), ("Preprocessing", "preprocess"), ("Forward pass", "forward"),
                  ("Decoding with NMS", "decode_nms"), ("Mist gate and angle reader", "readers")]
        body = [[name] + [f"{P[sd]['time_ms'][k][s]:.1f}" for sd in SEEDS for s in stats] for name, k in stages]
        body += ["\\midrule", ["Total time per frame $t_{\\text{inf}}$"] + [f"{P[sd]['time_ms']['total'][s]:.1f}" for sd in SEEDS for s in stats]]

        def res_row(name, vals):
            return f"{name} & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{v}}}" for v in vals) + " \\\\"

        def join(sd, key, parts, nd=1):
            return " / ".join(f"{P[sd][key][p]:.{nd}f}" for p in parts)

        footer = [
            res_row("Frames per second", [f"{P[sd]['fps']['sustained']:.2f}" for sd in SEEDS]),
            res_row("CPU load (\\%)", [join(sd, "cpu_percent_4_cores", ("mean", "peak")) for sd in SEEDS]),
            res_row("Memory (MB)", [join(sd, "rss_mb", ("mean", "peak")) for sd in SEEDS]),
            res_row("Temperature ($^{\\circ}$C)", [join(sd, "soc_temp_c", ("start", "end", "peak")) for sd in SEEDS]),
            res_row("CPU clock (MHz)", [join(sd, "cpu_mhz", ("min", "max"), 0) for sd in SEEDS]),
        ]
        pr = P[42]["protocol"]
        assert all(P[sd]["protocol"] == pr for sd in SEEDS)
        ag = {sd: P[sd]["agreement_with_pc"] for sd in SEEDS}
        assert all(a["frames_det_count_differs"] == 0 and a["states_differ"] == 0 for a in ag.values())
        # the quantities and the protocol are defined in Chapter 5, and the agreement with the PC is given in the text
        assert (pr["count"], pr["warmup"], pr["source_resolution"]) == (100, 20, [2048, 1536])
        note = ("Measured by the protocol of Section~\\ref{sec:ch5-deployment}, with ONNX Runtime on "
                f"${pr['threads']}$ threads, on a Raspberry Pi 5 with ${P[42]['platform']['ram_total_mb'] / 1024:.0f}$ GB of memory."
                "\\\\ The frame rate is sustained over the whole run. The CPU load and the memory are given as mean / peak, "
                "the temperature as start / end / peak, and the CPU clock as minimum / maximum.")
        header = "\n".join([
            "  & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{hd(f'seed {sd}')}}}" for sd in SEEDS) + " \\\\",
            "  \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
            "  " + hd("stage (ms)") + " & " + " & ".join(hd(s) for _ in SEEDS for s in ("mean", "median", "95th")) + " \\\\"])
        (GEN / "tab-ch6-pi5.tex").write_text(latex_table(
            "Chapter 6: the final model on the Raspberry Pi 5 (RQ3)",
            "The final model on the Raspberry Pi 5, over one hundred timed frames, for both seeds.",
            "tab:ch6-pi5", "l" + "r" * 6, header, body, footer=footer, note=note, fit=True,
            size="\\footnotesize\n\\setlength{\\tabcolsep}{4pt}"), encoding="utf-8")
        summary["pi5"] = {sd: {k: v for k, v in P[sd].items() if k != "per_frame_ms"} for sd in SEEDS}

    t = T[42]["tm"]["overall"]
    cf = load(RES / "counterfactual_test_abl_heads_s42.json")
    summary["test"] = {sd: {"overall": T[sd]["tm"]["overall"], "per_source": T[sd]["tm"].get("per_source"), "angle": T[sd]["ar"],
                            "mist": T[sd]["mh"]} for sd in SEEDS}
    json.dump(summary, open(RES / "thesis_ch6_summary.json", "w"), indent=1, default=str)
    print("wrote the tables to", GEN)


if __name__ == "__main__":
    main()
