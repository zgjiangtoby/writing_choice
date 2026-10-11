"""Compare Base and Instruct choice models with matched resampling draws."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from multiple_testing import holm
from inference_utils import adjusted_family,normal_tail_probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--instruct",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    inputs = {name:json.loads((path/"summary.json").read_text())
              for name,path in [("base",args.base),("instruct",args.instruct)]}
    summaries = {name:s["results"]["mean"]["coauthor"] for name,s in inputs.items()}
    for key in ("seed","sorted_cluster_ids","valid_draw_indices"):
        assert summaries["base"][key]==summaries["instruct"][key]
    for key in ("seed","sorted_worker_ids","valid_draw_indices"):
        assert summaries["base"]["paired_worker_refits"][key]==summaries["instruct"]["paired_worker_refits"][key]
    predictions = {name:pd.read_csv(path/"coauthor_mean_heldout.csv").sort_values("menu_id")
                   for name,path in [("base",args.base),("instruct",args.instruct)]}
    keys = ["menu_id","condition","topic","worker_id","session","fold"]
    assert predictions["base"][keys].reset_index(drop=True).equals(predictions["instruct"][keys].reset_index(drop=True))
    primary = []
    for name,summary in summaries.items():
        for row in summary["comparisons"]:
            if row["primary_family"]:
                primary.append({"checkpoint":name,**row})
    assert len(primary)==4
    for row in primary:
        expected,status = normal_tail_probability(row["delta_nll"],row["primary_se_maximum_fixed_and_refit"])
        actual = row["primary_normal_p_before_Holm"]
        assert (actual is None and expected is None) or (actual is not None and expected is not None and np.isclose(actual,expected,rtol=1e-12,atol=0)), "Inconsistent primary max-SE p-value; regenerate the checkpoint analysis"
        row["primary_normal_inference_status"] = status
    primary_complete = adjusted_family(primary,"primary_normal_p_before_Holm","holm_p_four_primary",holm)
    fixed = {name:np.load(path/"coauthor_mean_bootstrap.npz")
             for name,path in [("base",args.base),("instruct",args.instruct)]}
    refit = {name:np.load(path/"coauthor_mean_worker_refits.npz")
             for name,path in [("base",args.base),("instruct",args.instruct)]}
    checkpoint = []
    for comparison in ("BUG-BU","BUG-BUW"):
        point = {r["checkpoint"]:r["delta_nll"] for r in primary if r["comparison"]==comparison}
        key = "equal_condition_pool__"+comparison
        assert fixed["base"][key].shape==fixed["instruct"][key].shape
        assert refit["base"][key].shape==refit["instruct"][key].shape
        fixed_delta = fixed["instruct"][key]-fixed["base"][key]
        refit_delta = refit["instruct"][key]-refit["base"][key]
        delta = point["instruct"]-point["base"]
        se = max(fixed_delta.std(ddof=1),refit_delta.std(ddof=1))
        p_value,inference_status = normal_tail_probability(delta,se)
        checkpoint.append({"comparison":comparison,"instruct_minus_base_increment":delta,
            "paired_fixed_fit_95_interval":np.quantile(fixed_delta,[.025,.975]).tolist(),
            "paired_refit_95_interval":np.quantile(refit_delta,[.025,.975]).tolist(),
            "se_maximum_fixed_and_refit":float(se),
            "normal_approximation_p":p_value,"normal_inference_status":inference_status,
            "primary_family":False})
    secondary_complete = adjusted_family(checkpoint,"normal_approximation_p","holm_p_two_secondary_checkpoint_comparisons",holm)
    args.output.mkdir(parents=True)
    pd.DataFrame(primary).to_csv(args.output/"primary_four.csv",index=False)
    pd.DataFrame(checkpoint).to_csv(args.output/"paired_checkpoint.csv",index=False)
    report = {"base_analysis":str(args.base),"instruct_analysis":str(args.instruct),"primary_family":primary,
        "secondary_checkpoint_comparisons":checkpoint,
        "scope":"descriptive checkpoint comparison with identical input token sequences; no causal identification of post-training",
        "uncertainty":"paired worker draws; training head refits retain the original fold-specific supply features and worker partitions",
        "zero_variance_policy":"SE and effect within 1e-12 nats: p=1; nonzero effect with degenerate SE: p=null, family input=1, adjusted p=null",
        "primary_inference_complete":primary_complete,"secondary_inference_complete":secondary_complete,
        "complete":True}
    (args.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":"complete","primary_comparisons":4,"paired_checkpoint_comparisons":2,
                      "primary_inference_complete":primary_complete,"secondary_inference_complete":secondary_complete}),flush=True)


if __name__=="__main__":
    main()
