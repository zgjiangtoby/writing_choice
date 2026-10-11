"""Join replayed five-option menus to their selected candidates and folds."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "work"
SOURCES = {
    "hai": (ROOT / "hai_replay/menus.jsonl",
            ROOT / "hai_features/candidate_features.csv"),
    "coauthor": (ROOT / "coauthor_replay/menus.jsonl",
                 ROOT / "coauthor_features/primary_candidate_features.csv"),
}
FOLDS = ROOT / "coauthor_features/author_folds.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def feature_choices(path):
    choices = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["menu_size"]) != 5:
                continue
            menu_id = row["menu_id"]
            position = int(row["position"])
            if row["chosen"] == "True":
                if menu_id in choices:
                    raise ValueError(f"multiple chosen rows: {menu_id}")
                choices[menu_id] = position
    return choices


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    folds = json.loads(FOLDS.read_text())["worker_folds"]
    report = {"private_data_do_not_package": True, "source_sha256": {},
              "eligible_menus": {}, "conditions": {}, "empty_prefixes": {},
              "duplicate_candidate_text_menus": {}, "max_candidate_words": {}}
    for corpus, (source, feature_source) in SOURCES.items():
        choices = feature_choices(feature_source)
        seen, conditions = set(), Counter()
        empty_prefixes = duplicates = max_words = 0
        dest = output / f"{corpus}_menus.jsonl"
        with source.open() as stream, dest.open("w") as out:
            for line in stream:
                menu = json.loads(line)
                menu_id = menu["menu_id"]
                if menu_id not in choices:
                    continue
                if menu_id in seen or menu["status"] != "selected" or len(menu["candidates"]) != 5:
                    raise ValueError(f"invalid or duplicate menu: {menu_id}")
                seen.add(menu_id)
                candidates = sorted(menu["candidates"], key=lambda c: c["position"])
                if [c["position"] for c in candidates] != list(range(5)):
                    raise ValueError(f"invalid positions: {menu_id}")
                chosen = menu["selected_position"]
                if chosen != choices[menu_id] or candidates[chosen]["id"] != menu["selected_id"]:
                    raise ValueError(f"feature/replay choice mismatch: {menu_id}")
                texts = [c["text"] for c in candidates]
                if any(not isinstance(t, str) or not t for t in texts):
                    raise ValueError(f"empty candidate: {menu_id}")
                worker = menu.get("worker_id")
                if corpus == "coauthor" and worker not in folds:
                    raise ValueError(f"worker has no assigned fold: {menu_id}")
                record = {
                    "menu_id": menu_id, "session": menu["session"],
                    "condition": menu["condition"], "topic": menu["topic"],
                    "worker_id": worker, "worker_fold": folds[worker] if worker else None,
                    "prefix": menu["prefix"], "candidates": texts,
                    "chosen_position": chosen,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                conditions[menu["condition"]] += 1
                empty_prefixes += not bool(menu["prefix"])
                duplicates += len(set(texts)) < 5
                max_words = max(max_words, *(len(text.split()) for text in texts))
        if seen != set(choices):
            raise ValueError(f"feature/replay menu-set mismatch: {corpus}, missing {len(set(choices)-seen)}")
        report["source_sha256"][corpus] = {"replay": digest(source), "features": digest(feature_source)}
        report["eligible_menus"][corpus] = len(seen)
        report["conditions"][corpus] = dict(sorted(conditions.items()))
        report["empty_prefixes"][corpus] = empty_prefixes
        report["duplicate_candidate_text_menus"][corpus] = duplicates
        report["max_candidate_words"][corpus] = max_words
    report["source_sha256"]["worker_folds"] = digest(FOLDS)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))
