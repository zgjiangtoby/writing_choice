"""Build HAI TF-IDF candidate features with each target session excluded."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compute_centrality(menus):
    groups = defaultdict(list)
    for menu in menus:
        groups[(menu['condition'], menu['topic'])].append(menu)
    results, features, skipped = [], [], []
    for (condition, topic), group in groups.items():
        sessions = sorted({x['session'] for x in group})
        for session in sessions:
            reference = [x for x in group if x['session'] != session
                         and x['display_kind'] == 'suggestion-open']
            reference_sessions = {x['session'] for x in reference}
            if not reference_sessions:
                skipped.append({'session': session, 'reason': 'no other same-topic source log'})
                continue
            per_log = Counter(x['session'] for x in reference)
            texts, weights = [], []
            for menu in reference:
                weight = 1 / len(reference_sessions) / per_log[menu['session']] / len(menu['candidates'])
                texts.extend(c['text'] for c in menu['candidates'])
                weights.extend([weight] * len(menu['candidates']))
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b',
                lowercase=True, norm='l2', max_features=50000, dtype=np.float64)
            reference_vectors = vectorizer.fit_transform(texts)
            center = np.asarray(reference_vectors.multiply(np.array(weights)[:, None]).sum(axis=0)).ravel()
            for menu in group:
                if menu['session'] != session or menu['status'] != 'selected':
                    continue
                candidates = menu['candidates']
                vectors = vectorizer.transform([c['text'] for c in candidates])
                centrality = np.asarray(vectors @ center).ravel()
                prefix = vectorizer.transform([menu['prefix']])
                prefix_similarity = np.asarray((vectors @ prefix.T).toarray()).ravel()
                chosen = next(i for i, c in enumerate(candidates) if c['id'] == menu['selected_id'])
                result = {
                    'menu_id': menu['menu_id'], 'session': session,
                    'condition': condition, 'topic': topic,
                    'display_kind': menu['display_kind'], 'menu_size': len(candidates),
                    'reference_logs': len(reference_sessions),
                    'reference_menus': len(reference),
                    'selected_position': int(menu['selected_position']),
                    'chosen_centrality': float(centrality[chosen]),
                    'menu_mean_centrality': float(centrality.mean()),
                    'choice_shift': float(centrality[chosen] - centrality.mean()),
                    'chosen_prefix_similarity': float(prefix_similarity[chosen]),
                    'menu_mean_prefix_similarity': float(prefix_similarity.mean()),
                    'chosen_words': len(candidates[chosen]['text'].split()),
                    'menu_mean_words': float(np.mean([len(c['text'].split()) for c in candidates])),
                    'inserted_units': menu['inserted_units'],
                    'retained_units': menu['retained_units'],
                    'retention_fraction': menu['retention_fraction'],
                }
                results.append(result)
                for j, candidate in enumerate(candidates):
                    features.append({
                        'menu_id': menu['menu_id'], 'session': session,
                        'condition': condition, 'topic': topic,
                        'menu_size': len(candidates), 'position': candidate['position'],
                        'chosen': j == chosen, 'centrality': float(centrality[j]),
                        'prefix_similarity': float(prefix_similarity[j]),
                        'words': len(candidate['text'].split()),
                    })
    return pd.DataFrame(results), pd.DataFrame(features), skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    replay_summary = json.loads((args.replay / 'summary.json').read_text())
    if replay_summary['missing_logs'] or replay_summary['replay_failures'] or replay_summary['snapshot_mismatches']:
        raise ValueError('Resolve incomplete or inconsistent raw replay before choice analysis')
    menus = read_jsonl(args.replay / 'menus.jsonl')
    frame, features, skipped = compute_centrality(menus)
    frame.to_csv(args.output / 'accepted_menus.csv', index=False)
    features.to_csv(args.output / 'candidate_features.csv', index=False)
    (args.output / 'summary.json').write_text(json.dumps({
        'accepted_events_scored': len(frame), 'skipped_reference_cases': skipped,
        'accepted_events_in_replay': replay_summary['selected_menus'],
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
