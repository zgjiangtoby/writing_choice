"""Fit context-conditioned candidate-choice models and paired cluster refits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
from choice_design import FEATURES,pack,weights as hai_weights
from coauthor_features import menu_weights
from inference_utils import normal_tail_probability

MODEL_COLUMNS = {"B":list(range(7)),"BU":list(range(8)),"BUG":list(range(9)),
                 "BUW":list(range(8))+[9],"C_only":[10]}
NAMES = FEATURES+["U","C_minus_U","W_minus_U","C"]
PAIRS = [("BUG","BU"),("BUG","BUW"),("BU","B"),("C_only","B")]


def read_scores(path):
    result = {}
    for line in path.open():
        row = json.loads(line)
        assert row["id"] not in result
        assert np.isfinite(row["token_log_probabilities"]).all()
        assert len(row["token_log_probabilities"])==row["count"]
        assert np.isclose(sum(row["token_log_probabilities"]),row["sum_log_probability"])
        result[row["id"]] = row
    expected = set()
    for line in (HERE/"work/inputs/inputs.jsonl").open():
        key = json.loads(line)["id"]
        assert key not in expected
        expected.add(key)
    assert expected==set(result)
    rows = []
    for corpus in ("hai","coauthor"):
        for line in (HERE/"work/menus"/f"{corpus}_menus.jsonl").open():
            menu = json.loads(line)
            for position in range(5):
                row = {"corpus":corpus,"menu_id":menu["menu_id"],"position":position,
                       "recorded_choice":int(position==menu["chosen_position"])}
                for context in ("C","U","W"):
                    score = result[f"{corpus}:{menu['menu_id']}:p{position}:{context}"]
                    row[context+"_mean"] = score["mean_log_probability"]
                    row[context+"_total"] = score["sum_log_probability"]
                    row[context+"_canonical"] = score["canonical_boundary"]
                    row[context+"_truncated"] = score["prefix_tokens_original"]>score["prefix_tokens_retained"]
                    row[context+"_prefix_nonempty"] = score["prefix_tokens_original"]>0
                rows.append(row)
    return pd.DataFrame(rows)


def assemble(frame,scores,corpus,scope):
    frame = frame[frame.menu_size.eq(5)].merge(scores[scores.corpus.eq(corpus)],on=["menu_id","position"],validate="one_to_one")
    assert (frame.chosen==frame.recorded_choice).all()
    if scope=="canonical":
        good = frame[[c+"_canonical" for c in ("C","U","W")]].all(axis=1)
        good_menus = frame.assign(valid=good).groupby("menu_id").valid.all()
        frame = frame[frame.menu_id.isin(good_menus.index[good_menus])]
    elif scope=="nonempty_prefix":
        frame = frame[frame.C_prefix_nonempty]
    return frame


def packed(frame,variant):
    frame = frame.sort_values(["menu_id","position"])
    base,y,meta = pack(frame)
    suffix = "total" if variant=="total" else "mean"
    c,u,w = [frame[name+"_"+suffix].to_numpy().reshape(-1,5) for name in ("C","U","W")]
    extra = np.stack([u,c-u,w-u,c],axis=-1)
    extra -= extra.mean(axis=1,keepdims=True)
    raw = np.concatenate([base,extra],axis=-1)
    grouped = frame.groupby("menu_id",sort=True)
    for column in ("worker_id","condition","partition","outer_fold"):
        if column in frame:
            assert grouped[column].nunique().le(1).all()
            meta[column] = meta.menu_id.map(grouped[column].first())
    meta["selected_context_gain"] = (c-u)[np.arange(len(y)),y]-(c-u).mean(1)
    if "retention_fraction" in frame:
        chosen = frame[frame.chosen.eq(1)].set_index("menu_id")
        meta["retention_fraction"] = meta.menu_id.map(chosen.retention_fraction)
    return raw,y,meta


def weight(meta,corpus):
    result = menu_weights(meta) if corpus=="coauthor" else hai_weights(meta)
    assert np.isclose(result.sum(),1)
    return result


def fit(raw,y,w,name,menu_multiplicity=None):
    columns = MODEL_COLUMNS[name]
    scale = np.ones(raw.shape[-1])
    for k in [0,1,6,7,8,9,10]:
        if menu_multiplicity is None:
            scale[k] = np.std(raw[:,:,k]) or 1.
        else:
            # Match physically duplicated menus; choice-loss hierarchy weights
            # and the original unweighted candidate scaling are different.
            moment = np.average((raw[:,:,k]**2).mean(1),weights=menu_multiplicity)
            mean = np.average(raw[:,:,k].mean(1),weights=menu_multiplicity)
            scale[k] = np.sqrt(max(0.,moment-mean**2)) or 1.
    x = (raw/scale)[:,:,columns]
    def objective(beta):
        utility = x@beta
        nll = logsumexp(utility,axis=1)-utility[np.arange(len(x)),y]
        difference = softmax(utility,axis=1)
        difference[np.arange(len(x)),y] -= 1
        gradient = np.einsum("q,qj,qjk->k",w,difference,x)+.001*beta
        return float(w@nll+.0005*(beta@beta)),gradient
    bounds = [(0.,None)] if name=="C_only" else None
    result = minimize(objective,np.zeros(len(columns)),jac=True,method="L-BFGS-B",bounds=bounds,
                      options={"maxiter":1000,"ftol":1e-13,"gtol":1e-8,"maxls":40})
    value,gradient = objective(result.x)
    projected = gradient.copy()
    if name=="C_only" and result.x[0]<=1e-10 and gradient[0]>0:
        projected[0] = 0
    assert result.success and np.max(np.abs(projected))<1e-5,(name,result.message,gradient)
    return {"beta":result.x,"scale":scale,"columns":columns,
            "record":{"model":name,"features":[NAMES[k] for k in columns],"beta_scaled":result.x.tolist(),
                      "coefficients_original_units":(result.x/scale[columns]).tolist(),"scales":scale[columns].tolist(),
                      "iterations":int(result.nit),"objective":value,
                      "max_gradient":float(np.abs(gradient).max()),"max_projected_gradient":float(np.abs(projected).max()),
                      "message":str(result.message)}}


def predict(model,raw,y):
    utility = (raw/model["scale"])[:,:,model["columns"]]@model["beta"]
    p = softmax(utility,axis=1)
    selected = p[np.arange(len(y)),y]
    return {"nll":logsumexp(utility,axis=1)-utility[np.arange(len(y)),y],
            "accuracy":(p.argmax(1)==y).astype(float),"selected_probability":selected}


def evaluate(corpus,scores,variant):
    base = HERE/("work/coauthor_features/primary_candidate_features.csv" if corpus=="coauthor"
                        else "work/hai_features/candidate_features.csv")
    primary = assemble(pd.read_csv(base),scores,corpus,variant)
    fits,predictions,unsupported = [],[],[]
    for condition,frame in primary.groupby("condition"):
        raw,y,meta = packed(frame,variant)
        for name in MODEL_COLUMNS:
            model = fit(raw,y,weight(meta,corpus),name)
            fits.append({"fit":"full_descriptive","corpus":corpus,"condition":condition,
                         "train_menus":len(y),**model["record"]})
    folds = range(5) if corpus=="coauthor" else sorted(primary.topic.unique())
    for fold in folds:
        if corpus=="coauthor":
            data = assemble(pd.read_csv(HERE/f"work/coauthor_features/outer_fold_{fold}_candidate_features.csv"),scores,corpus,variant)
        else:
            data = primary.copy()
            data["partition"] = np.where(data.topic.eq(fold),"test","train")
        for condition,frame in data.groupby("condition"):
            train,test = frame[frame.partition.eq("train")],frame[frame.partition.eq("test")]
            if train.empty or test.empty:
                unsupported.append({"condition":condition,"fold":fold,"train_rows":len(train),"test_rows":len(test)})
                continue
            traw,ty,tmeta = packed(train,variant)
            raw,y,meta = packed(test,variant)
            if corpus=="coauthor":
                assert not set(tmeta.worker_id)&set(meta.worker_id)
            row = meta.copy()
            row["fold"] = fold
            for name in MODEL_COLUMNS:
                model = fit(traw,ty,weight(tmeta,corpus),name)
                fits.append({"fit":"heldout","corpus":corpus,"condition":condition,"fold":fold,
                             "train_menus":len(ty),"test_menus":len(y),**model["record"]})
                for metric,values in predict(model,raw,y).items():
                    row[name+"_"+metric] = values
            predictions.append(row)
    result = pd.concat(predictions,ignore_index=True)
    assert result.menu_id.is_unique
    assert set(result.menu_id)==set(primary.menu_id)
    return result,fits,unsupported


def aggregate(group,corpus,multiplicity=None):
    columns = [c for c in group.columns if c.endswith(("_nll","_accuracy","_selected_probability"))]
    for c in ("selected_context_gain","retention_fraction"):
        if c in group and group[c].notna().all():
            columns.append(c)
    # Aggregate the original hierarchy before cluster reweighting.
    if corpus=="coauthor":
        session = group.groupby(["topic","worker_id","session"])[columns].mean()
        cluster = session.groupby(["topic","worker_id"]).mean().reset_index()
        cluster["mult"] = cluster.worker_id.map(multiplicity).fillna(0) if multiplicity is not None else 1.
        totals = []
        for _,topic in cluster.groupby("topic"):
            if topic.mult.sum()==0:
                return None
            totals.append(np.average(topic[columns],axis=0,weights=topic.mult))
        values = np.mean(totals,axis=0)
    else:
        topic = group.groupby(["topic","session"])[columns].mean().groupby("topic").mean()
        w = topic.index.map(multiplicity).to_numpy(dtype=float) if multiplicity is not None else np.ones(len(topic))
        if w.sum()==0:
            return None
        values = np.average(topic,axis=0,weights=w)
    return dict(zip(columns,map(float,values)))


def refit_weight(meta,multiplicity):
    """Repeat workers while retaining the original topic/worker/session/menu hierarchy."""
    counts = meta.worker_id.map(multiplicity).to_numpy(dtype=float)
    unique = meta[["topic","worker_id"]].drop_duplicates()
    totals = unique.assign(mult=unique.worker_id.map(multiplicity)).groupby("topic").mult.sum()
    if (totals==0).any():
        return None,counts
    original_workers = unique.groupby("topic").size()
    factor = counts*meta.topic.map(original_workers/totals).to_numpy(dtype=float)
    w = weight(meta,"coauthor")*factor
    assert np.isclose(w.sum(),1)
    return w,counts


def paired_worker_refits(scores,predictions,draws):
    """Refit the three primary heads under paired worker resampling; fixed reference features/folds."""
    folds = []
    for fold in range(5):
        data = assemble(pd.read_csv(HERE/f"work/coauthor_features/outer_fold_{fold}_candidate_features.csv"),
                        scores,"coauthor","mean")
        for condition,frame in data.groupby("condition"):
            train = packed(frame[frame.partition.eq("train")],"mean")
            test = packed(frame[frame.partition.eq("test")],"mean")
            assert not set(train[2].worker_id)&set(test[2].worker_id)
            folds.append((fold,condition,train,test))
    clusters = sorted(predictions.worker_id.unique())
    conditions = sorted(predictions.condition.unique())
    rng = np.random.default_rng(20261019)
    values = {c:{f"{new}-{old}":[] for new,old in PAIRS[:2]} for c in conditions+["equal_condition_pool"]}
    invalid,max_gradient,fit_count = 0,0.,0
    valid_indices = []
    for draw in range(draws):
        multiplicity = dict(zip(clusters,rng.multinomial(len(clusters),np.full(len(clusters),1/len(clusters)))))
        prepared = [(entry,refit_weight(entry[2][2],multiplicity)) for entry in folds]
        if any(weights[0] is None for _,weights in prepared):
            invalid += 1
            continue
        heldout = []
        for (fold,condition,(traw,ty,tmeta),(raw,y,meta)),(w,count) in prepared:
            row = meta.copy()
            for name in ("BU","BUG","BUW"):
                model = fit(traw,ty,w,name,menu_multiplicity=count)
                max_gradient = max(max_gradient,model["record"]["max_projected_gradient"])
                fit_count += 1
                row[name+"_nll"] = predict(model,raw,y)["nll"]
            heldout.append(row)
        combined = pd.concat(heldout,ignore_index=True)
        assert combined.menu_id.is_unique and set(combined.menu_id)==set(predictions.menu_id)
        cells = {c:aggregate(combined[combined.condition.eq(c)],"coauthor",multiplicity) for c in conditions}
        if any(v is None for v in cells.values()):
            invalid += 1
            continue
        valid_indices.append(draw)
        for new,old in PAIRS[:2]:
            key = f"{new}-{old}"
            deltas = []
            for condition in conditions:
                delta = cells[condition][new+"_nll"]-cells[condition][old+"_nll"]
                values[condition][key].append(delta)
                deltas.append(delta)
            values["equal_condition_pool"][key].append(float(np.mean(deltas)))
        if (draw+1)%10==0:
            print(json.dumps({"worker_refit_draw":draw+1,"total":draws,"invalid_draws":invalid}),flush=True)
    assert draws-invalid>.9*draws
    comparisons = []
    for condition in conditions+["equal_condition_pool"]:
        groups = conditions if condition=="equal_condition_pool" else [condition]
        point = [aggregate(predictions[predictions.condition.eq(c)],"coauthor") for c in groups]
        for new,old in PAIRS[:2]:
            key = f"{new}-{old}"
            delta = np.mean([p[new+"_nll"]-p[old+"_nll"] for p in point])
            distribution = np.asarray(values[condition][key])
            se = distribution.std(ddof=1)
            comparisons.append({"condition":condition,"comparison":key,"delta_nll":float(delta),
                "paired_worker_refit_95_interval":np.quantile(distribution,[.025,.975]).tolist(),
                "bootstrap_se":float(se),"normal_approximation_p":float(2*norm.sf(abs(delta)/se)) if se else None,
                "valid_draws":len(distribution),"invalid_draws":invalid})
    return {"comparisons":comparisons,"fit_count":fit_count,"max_projected_gradient":max_gradient,
        "seed":20261019,"valid_draw_indices":valid_indices,"sorted_worker_ids":clusters,
        "scope":"paired worker multiplicities in training and held-out aggregation; choice heads/scales refitted; original fold-specific supply features and worker partitions fixed",
        "topic_support":"draws lacking any required training or evaluation topic support are counted and excluded"},values


def summarize(predictions,corpus,draws):
    conditions = sorted(predictions.condition.unique())
    point = {c:aggregate(predictions[predictions.condition.eq(c)],corpus) for c in conditions}
    cluster_key = "worker_id" if corpus=="coauthor" else "topic"
    clusters = sorted(predictions[cluster_key].unique())
    rng = np.random.default_rng(20261009)
    values = {condition:{f"{new}-{old}":[] for new,old in PAIRS} for condition in conditions+["equal_condition_pool"]}
    invalid = 0
    valid_indices = []
    for draw in range(draws):
        counts = rng.multinomial(len(clusters),np.full(len(clusters),1/len(clusters)))
        multiplicity = dict(zip(clusters,counts))
        fitted = {c:aggregate(predictions[predictions.condition.eq(c)],corpus,multiplicity) for c in conditions}
        if any(v is None for v in fitted.values()):
            invalid += 1
            continue
        valid_indices.append(draw)
        for new,old in PAIRS:
            key = f"{new}-{old}"
            deltas = []
            for c in conditions:
                delta = fitted[c][new+"_nll"]-fitted[c][old+"_nll"]
                deltas.append(delta)
                values[c][key].append(delta)
            values["equal_condition_pool"][key].append(float(np.mean(deltas)))
    point["equal_condition_pool"] = {key:float(np.mean([point[c][key] for c in conditions])) for key in point[conditions[0]]
                                      if all(key in point[c] for c in conditions)}
    comparisons = []
    for c,metrics in point.items():
        for new,old in PAIRS:
            key = f"{new}-{old}"
            bootstrap = np.asarray(values[c][key])
            assert len(bootstrap)>.9*draws
            delta = metrics[new+"_nll"]-metrics[old+"_nll"]
            se = bootstrap.std(ddof=1)
            record = {"condition":c,"comparison":key,"delta_nll":delta,"bootstrap_se":float(se),
                      "conditional_fixed_fit_95_interval":np.quantile(bootstrap,[.025,.975]).tolist(),
                      "valid_draws":len(bootstrap),"invalid_draws":invalid,
                      "primary_family":corpus=="coauthor" and c=="equal_condition_pool" and (new,old) in PAIRS[:2],
                      "normal_approximation_p":float(2*norm.sf(abs(delta)/se)) if corpus=="coauthor" and se else None}
            if corpus=="hai":
                deleted = []
                for topic in clusters:
                    sub = predictions[predictions.topic.ne(topic)]
                    cells = conditions if c=="equal_condition_pool" else [c]
                    estimates = [aggregate(sub[sub.condition.eq(cell)],corpus) for cell in cells]
                    if all(r is not None for r in estimates):
                        deleted.append(np.mean([r[new+"_nll"]-r[old+"_nll"] for r in estimates]))
                record["delete_one_topic_range"] = [float(min(deleted)),float(max(deleted))]
            comparisons.append(record)
    interactions = []
    for i,a in enumerate(conditions):
        for b in conditions[i+1:]:
            for new,old in PAIRS[:2]:
                key = f"{new}-{old}"
                delta = (point[a][new+"_nll"]-point[a][old+"_nll"])-(point[b][new+"_nll"]-point[b][old+"_nll"])
                bootstrap = np.asarray(values[a][key])-np.asarray(values[b][key])
                interactions.append({"condition_a":a,"condition_b":b,"comparison":key,"difference_of_increments":delta,
                                     "paired_cluster_95_interval":np.quantile(bootstrap,[.025,.975]).tolist()})
    return {"point":point,"comparisons":comparisons,"condition_interactions":interactions,
            "cluster_key":cluster_key,"clusters":len(clusters),"seed":20261009,
            "valid_draw_indices":valid_indices,"sorted_cluster_ids":clusters,
            "uncertainty":"fixed fitted choice models and supply references"},values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores",type=Path,required=True)
    parser.add_argument("--model-name",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--worker-refits",type=int,default=100)
    args = parser.parse_args()
    assert not args.output.exists() and args.bootstrap>=2000 and args.worker_refits>=100
    scores = read_scores(args.scores)
    args.output.mkdir(parents=True)
    results,fits_all,coverage = {},[],[]
    for variant in ("mean","total","canonical","nonempty_prefix"):
        results[variant] = {}
        for corpus in ("hai","coauthor"):
            predictions,fits,unsupported = evaluate(corpus,scores,variant)
            summary,draws = summarize(predictions,corpus,args.bootstrap)
            summary["unsupported_folds"] = unsupported
            results[variant][corpus] = summary
            if variant=="mean" and corpus=="coauthor":
                refits,refit_draws = paired_worker_refits(scores,predictions,args.worker_refits)
                summary["paired_worker_refits"] = refits
                np.savez_compressed(args.output/"coauthor_mean_worker_refits.npz",
                    **{c+"__"+key:np.asarray(v) for c,cell in refit_draws.items() for key,v in cell.items()})
                by_key = {(r["condition"],r["comparison"]):r for r in refits["comparisons"]}
                for comparison in summary["comparisons"]:
                    key = comparison["condition"],comparison["comparison"]
                    if key in by_key:
                        se = max(comparison["bootstrap_se"],by_key[key]["bootstrap_se"])
                        comparison["primary_se_maximum_fixed_and_refit"] = se
                        p_value,inference_status = normal_tail_probability(comparison["delta_nll"],se)
                        comparison["primary_normal_p_before_Holm"] = p_value
                        comparison["primary_normal_inference_status"] = inference_status
            predictions.to_csv(args.output/f"{corpus}_{variant}_heldout.csv",index=False)
            np.savez_compressed(args.output/f"{corpus}_{variant}_bootstrap.npz",
                                **{c+"__"+key:np.asarray(v) for c,cell in draws.items() for key,v in cell.items()})
            fits_all.extend({"variant":variant,**record} for record in fits)
            coverage.append({"variant":variant,"corpus":corpus,"menus":len(predictions),
                             "sessions":predictions.session.nunique(),"topics":predictions.topic.nunique()})
            print(json.dumps({"variant":variant,"corpus":corpus,"menus":len(predictions),"complete":True}),flush=True)
    pd.DataFrame(coverage).to_csv(args.output/"coverage.csv",index=False)
    (args.output/"fits.json").write_text(json.dumps(fits_all,indent=2)+"\n")
    (args.output/"summary.json").write_text(json.dumps({"model":args.model_name,"source_scores":str(args.scores),
        "results":results,"L2":.001,"solver":"L-BFGS-B","ftol":1e-13,"gtol":1e-8,
        "stationarity_tolerance":1e-5,"primary_variant":"mean",
        "primary_multiplicity":"two pooled contrasts per checkpoint, Holm across four after both models; raw p retained here",
        "scope":"observed accepted five-option menus; no simulated people or causal selection effect", "complete":True},indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
