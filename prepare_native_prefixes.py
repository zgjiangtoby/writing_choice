"""Recover source whitespace at existing menu cursors without changing any menu or choice."""

import argparse
import json
import re
from collections import Counter,defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
from replay_hai import units,as_text,apply_delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    records,summary = {},{}
    for corpus,replay,raw in [("hai","hai_replay","hai"),
                              ("coauthor","coauthor_replay","coauthor")]:
        selected = {r["menu_id"]:r for r in map(json.loads,(HERE/"work/menus_lexical"/f"{corpus}_menus.jsonl").open())}
        files = defaultdict(list)
        for row in map(json.loads,(HERE/"work"/replay/"menus.jsonl").open()):
            if row["menu_id"] in selected:
                files[row["path"].removeprefix("./")].append(row)
        result,counts = {},Counter()
        for number,(path,menus) in enumerate(sorted(files.items())):
            events = [json.loads(s) for s in (HERE/"data"/raw/path).open() if s.strip()]
            needed = {row["event_index"]:row for row in menus}
            doc = units(events[0].get("currentDoc") or "\n")
            owners = [None]*len(doc)
            for index,event in enumerate(events):
                if index in needed:
                    row = needed[index]
                    value = event.get("currentCursor")
                    cursor = int(value) if value not in ("",None) else len(doc)
                    prefix = as_text(doc[:cursor]).split("\n---\n")[-1]
                    words = list(re.finditer(r"\S+",prefix))
                    retained = prefix[words[-200].start():] if len(words)>200 else prefix.lstrip()
                    assert " ".join(retained.split())==row["prefix"],(corpus,row["menu_id"])
                    old = selected[row["menu_id"]]
                    assert old["prefix"]==row["prefix"]
                    new = dict(old,native_prefix=retained,normalized_lexical_prefix=old["prefix"],
                               prefix=retained.rstrip(),prefix_source_event=index,
                               fixed_scoring_separator=" ")
                    assert new["candidates"]==old["candidates"] and new["chosen_position"]==old["chosen_position"]
                    result[row["menu_id"]] = new
                    counts["menus"] += 1
                    counts["word_sequence_matches_original"] += 1
                    counts["original_native_prefix_ends_whitespace"] += bool(retained) and retained[-1].isspace()
                    counts["internal_whitespace_differs_from_lexical_prefix"] += new["prefix"]!=old["prefix"]
                    counts["menus_with_original_leading_candidate_whitespace"] += any(c[0].isspace() for c in old["candidates"])
                if index>=max(needed):
                    break
                delta = event.get("textDelta")
                if isinstance(delta,dict) and delta.get("ops"):
                    doc,owners,_,_ = apply_delta(doc,owners,delta["ops"],None,None)
            if (number+1)%50==0:
                print(json.dumps({"corpus":corpus,"sessions_replayed":number+1,"total_sessions":len(files)}),flush=True)
        assert set(result)==set(selected)
        records[corpus] = [result[key] for key in selected]
        summary[corpus] = {**dict(counts),"sessions":len(files),"original_menu_count":len(selected)}
    args.output.mkdir(parents=True)
    for corpus,rows in records.items():
        with (args.output/f"{corpus}_menus.jsonl").open("x") as output:
            for row in rows:
                output.write(json.dumps(row,ensure_ascii=False)+"\n")
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
