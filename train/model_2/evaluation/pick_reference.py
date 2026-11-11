"""
pick_reference.py - decide, by a rule fixed BEFORE the results were seen,
which configuration is the clean v10 reference that steps 3-5 build on.

Candidates: v10_ref (INSTANCE_NORM on) vs abl_Foff (INSTANCE_NORM off).
Inputs: results/decode_tuning_val_<tag>.json for each seed (per-class F1 at
OBJ_THRESH, standard instance-level matching on VAL).

Rule (docs/notes/PLAN-V10.md step 2c):
  1. a config "collapses" if, in ANY seed, any small-nozzle class
     (spray / no_spray / facing) has F1 < 0.20 at the best checkpoint;
  2. prefer the config that does not collapse;
  3. if both or neither collapse, prefer the higher mean macro F1 over seeds;
  4. tie (|diff| < 0.01): keep INSTANCE_NORM on (fewer changes to the plan).

Prints ONE line with the env assignment for run_abl.sh ("" or
"AFGNET_INSTANCE_NORM=0"), writes results/pick_reference.json.
"""

import json
import sys

from common import RESULTS_DIR

SMALL = ("nozzle_spray", "nozzle_no_spray", "nozzle_facing")
CANDS = {"v10_ref": "", "abl_Foff": "AFGNET_INSTANCE_NORM=0"}
SEEDS = (42, 123)


def stats(name):
    per_seed = []
    for s in SEEDS:
        p = RESULTS_DIR / f"decode_tuning_val_{name}_s{s}.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        sw = d["obj_thresh_sweep"][str(d["current"]["OBJ_THRESH"])]
        per_seed.append({"macro": sw["macro_F1"], "cls": {k: v["F1"] for k, v in sw["per_class"].items()}})
    return {
        "per_seed": per_seed,
        "mean_macro": sum(x["macro"] for x in per_seed) / len(per_seed),
        "collapse": any(x["cls"][c] < 0.20 for x in per_seed for c in SMALL),
        "min_small_F1": min(x["cls"][c] for x in per_seed for c in SMALL),
    }


def main():
    st = {k: stats(k) for k in CANDS}
    missing = [k for k, v in st.items() if v is None]
    if missing:
        sys.stderr.write(f"missing results for {missing}\n")
        sys.exit(2)
    a, b = st["v10_ref"], st["abl_Foff"]
    if a["collapse"] != b["collapse"]:
        pick, why = ("abl_Foff" if a["collapse"] else "v10_ref"), "only one config keeps every small class above F1 0.20"
    elif abs(a["mean_macro"] - b["mean_macro"]) < 0.01:
        pick, why = "v10_ref", "macro F1 tie (<0.01): keep INSTANCE_NORM on"
    else:
        pick, why = ("abl_Foff" if b["mean_macro"] > a["mean_macro"] else "v10_ref"), "higher mean macro F1 over seeds"
    out = {"candidates": st, "pick": pick, "env": CANDS[pick], "why": why}
    (RESULTS_DIR / "pick_reference.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    sys.stderr.write(json.dumps({k: {"mean_macro": round(v["mean_macro"], 3), "collapse": v["collapse"], "min_small_F1": round(v["min_small_F1"], 3)} for k, v in st.items()}, indent=1) + f"\npick: {pick} ({why})\n")
    print(CANDS[pick])


if __name__ == "__main__":
    main()
