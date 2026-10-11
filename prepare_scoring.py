"""Make outcome-blind conditional, unconditional, and matched-prefix scoring records."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--menus",type=Path,default=HERE/"work/menus")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261009)
    report = {}
    with (args.output/"inputs.jsonl").open("x") as out, (args.output/"pilot.jsonl").open("x") as pilot, (args.output/"mapping.jsonl").open("x") as mapping:
        for corpus in ("hai", "coauthor"):
            menus = [json.loads(s) for s in (args.menus/f"{corpus}_menus.jsonl").open()]
            cells, by_condition = defaultdict(list), defaultdict(list)
            for index, menu in enumerate(menus):
                cells[(menu["condition"],menu["topic"])].append(index)
                by_condition[menu["condition"]].append(index)
            technical = set()
            for condition, indices in sorted(by_condition.items()):
                technical.update(rng.choice(indices, size=min(4,len(indices)), replace=False).tolist())
            unsupported = 0
            for index, menu in enumerate(menus):
                pool = [j for j in cells[(menu["condition"],menu["topic"])]
                        if menus[j]["session"] != menu["session"]
                        and (corpus == "hai" or menus[j]["worker_id"] != menu["worker_id"])
                        and menus[j]["prefix"] != menu["prefix"]]
                donor = menus[int(rng.choice(pool))] if pool else None
                unsupported += donor is None
                contexts = {"C": menu["prefix"], "U": ""}
                if donor is not None:
                    contexts["W"] = donor["prefix"]
                item_id = f"{corpus}:{menu['menu_id']}"
                mapping.write(json.dumps({"item_id": item_id, "corpus": corpus,
                    "menu_id": menu["menu_id"], "donor_menu_id": donor["menu_id"] if donor else None,
                    "technical_pilot": index in technical},ensure_ascii=False)+"\n")
                for position, candidate in enumerate(menu["candidates"]):
                    for context, prefix in contexts.items():
                        record = {"id": f"{item_id}:p{position}:{context}", "prefix": prefix,
                                  "continuation": " "+candidate, "inserted_scoring_separator":" ","item_id": item_id,
                                  "position": position, "context": context}
                        line = json.dumps(record,ensure_ascii=False)+"\n"
                        out.write(line)
                        if index in technical:
                            pilot.write(line)
            report[corpus] = {"menus": len(menus), "technical_menus": len(technical),
                              "missing_matched_prefix": unsupported, "gold_written_to_scoring_input": False}
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
