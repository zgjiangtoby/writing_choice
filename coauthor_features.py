"""Build CoAuthor features with target writers and held-out writers excluded."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


SEED = 20261008
BOOTSTRAPS = 2000
STRATUM = ['writing_type', 'prompt_code', 'temperature', 'frequency_penalty']
FEATURE_COLUMNS = [
    'menu_id', 'session', 'worker_id', 'writing_type', 'prompt_code',
    'temperature', 'frequency_penalty', 'condition', 'topic',
    'display_kind', 'menu_size', 'position', 'candidate_id', 'chosen',
    'centrality', 'prefix_similarity', 'words', 'logged_id_agrees',
    'inserted_units', 'retained_units', 'retention_fraction',
    'reference_workers', 'reference_sessions', 'reference_menus',
    'outer_fold', 'partition',
]


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def author_folds(sessions):
    workers = sorted({row['worker_id'] for row in sessions})
    order = np.random.default_rng(SEED).permutation(workers)
    return {str(worker): int(i % 5) for i, worker in enumerate(order)}


def menu_weights(meta):
    """Equal topic, worker within topic, session within worker, then menu."""
    topics = meta['topic'].nunique()
    workers = meta.groupby('topic')['worker_id'].nunique()
    sessions = meta.groupby(['topic', 'worker_id'])['session'].nunique()
    menus = meta.groupby(['topic', 'worker_id', 'session']).size()
    return np.array([
        1 / topics / workers[row.topic] / sessions[(row.topic, row.worker_id)]
        / menus[(row.topic, row.worker_id, row.session)] for row in meta.itertuples()
    ])


def prepare_inputs(replay):
    summary = json.loads((replay / 'summary.json').read_text())
    sessions = read_jsonl(replay / 'sessions.jsonl')
    menus = read_jsonl(replay / 'menus.jsonl')
    eligible_sessions = set()
    coverage = {}
    for row in sessions:
        reason = None
        if row['replay_status'] != 'replayed':
            reason = row['replay_status']
        elif not row['event_count_matches_metadata']:
            reason = 'metadata_event_count_mismatch'
        elif row['noninitial_snapshot_mismatches']:
            reason = 'noninitial_snapshot_mismatch'
        if reason is None:
            eligible_sessions.add(row['session'])
        coverage[row['session']] = {
            key: row[key] for key in ['session', 'worker_id', *STRATUM]
        }
        coverage[row['session']].update({
            'replay_status': row['replay_status'], 'session_exclusion': reason,
            'all_menus': 0, 'accepted_menus': 0,
            'valid_new_supply_menus': 0,
            'eligible_choice_menus_at_least_two': 0,
            'eligible_choice_menus_five': 0, 'primary_feature_menus': 0,
            'menu_exclusions': {},
            'choice_exclusions': {},
        })
    groups = defaultdict(list)
    excluded_menus = []
    for menu in menus:
        row = coverage[menu['session']]
        row['all_menus'] += 1
        row['accepted_menus'] += menu['status'] == 'selected'
        if menu['session'] not in eligible_sessions:
            reason = row['session_exclusion']
        elif not menu['settings_match_metadata']:
            reason = 'menu_settings_disagree_with_metadata'
        else:
            reason = None
        if reason is not None:
            excluded_menus.append({'menu_id': menu['menu_id'], 'reason': reason})
            row['menu_exclusions'][reason] = row['menu_exclusions'].get(reason, 0) + 1
            continue
        groups[tuple(menu[key] for key in STRATUM)].append(menu)
    for key, group in list(groups.items()):
        other_settings = {tuple(menu['generation_settings'][field] for field in
                                ['n', 'max_tokens', 'top_p', 'presence_penalty']) for menu in group}
        if len(other_settings) != 1 or any(value is None for values in other_settings for value in values):
            for menu in group:
                reason = 'unresolved_generation_parameters_within_metadata_stratum'
                excluded_menus.append({'menu_id': menu['menu_id'], 'reason': reason})
                row = coverage[menu['session']]
                row['menu_exclusions'][reason] = row['menu_exclusions'].get(reason, 0) + 1
            del groups[key]
    for group in groups.values():
        for menu in group:
            coverage[menu['session']]['valid_new_supply_menus'] += menu['display_kind'] == 'suggestion-open'
            if menu['status'] != 'selected':
                continue
            if not menu['analysis_choice_eligible']:
                reason = menu['choice_alignment']['alignment_status']
            elif menu['insertion_match'] is not True:
                reason = 'selected_without_exact_attributable_insertion'
            elif len(menu['candidates']) < 2:
                reason = 'single_option_menu'
            else:
                continue
            row = coverage[menu['session']]
            row['choice_exclusions'][reason] = row['choice_exclusions'].get(reason, 0) + 1
            excluded_menus.append({'menu_id': menu['menu_id'], 'reason': reason,
                                   'exclusion_scope': 'choice_only_supply_retained'})
    return summary, sessions, groups, coverage, excluded_menus


def selected_choices(group):
    return [menu for menu in group if menu['status'] == 'selected'
            and len(menu['candidates']) >= 2 and menu['analysis_choice_eligible']
            and menu['insertion_match'] is True]


def build_features(replay, output):
    replay_summary, sessions, groups, coverage, excluded_menus = prepare_inputs(replay)
    folds = author_folds(sessions)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'author_folds.json').write_text(json.dumps({
        'seed': SEED, 'folds': 5, 'worker_folds': folds,
        'basis': 'all metadata workers, including unavailable sessions',
    }, indent=2) + '\n')
    choices = {key: selected_choices(group) for key, group in groups.items()}
    for group in choices.values():
        for menu in group:
            row = coverage[menu['session']]
            row['eligible_choice_menus_at_least_two'] += 1
            row['eligible_choice_menus_five'] += len(menu['candidates']) == 5
    contexts = []
    with (output / 'reference_blocks.jsonl').open('w') as block_out:
        for fold in [-1, 0, 1, 2, 3, 4]:
            name = 'primary' if fold == -1 else f'outer_fold_{fold}'
            held_out = {worker for worker, value in folds.items() if value == fold} if fold >= 0 else set()
            context_counts = Counter()
            path = output / f'{name}_candidate_features.csv'
            with path.open('w', newline='') as feature_out:
                writer = csv.DictWriter(feature_out, fieldnames=FEATURE_COLUMNS)
                writer.writeheader()
                for key in sorted(groups):
                    group = groups[key]
                    supply = [menu for menu in group if menu['display_kind'] == 'suggestion-open'
                              and menu['worker_id'] not in held_out]
                    by_target = defaultdict(list)
                    for menu in choices[key]:
                        by_target[menu['worker_id']].append(menu)
                    for worker, target in sorted(by_target.items()):
                        reference = [menu for menu in supply if menu['worker_id'] != worker]
                        ref_workers = {menu['worker_id'] for menu in reference}
                        ref_sessions = {menu['session'] for menu in reference}
                        block = dict(zip(STRATUM, key))
                        block.update({
                            'context': name, 'target_worker_id': worker,
                            'held_out_workers': sorted(held_out),
                            'reference_worker_ids': sorted(ref_workers),
                            'reference_workers': len(ref_workers),
                            'reference_sessions': len(ref_sessions),
                            'reference_menus': len(reference), 'target_menus': len(target),
                        })
                        if not reference:
                            block['status'] = 'no_other_worker_candidate_supply'
                            block_out.write(json.dumps(block) + '\n')
                            context_counts[block['status']] += len(target)
                            continue
                        sessions_by_worker = defaultdict(set)
                        menus_per_session = Counter(menu['session'] for menu in reference)
                        for menu in reference:
                            sessions_by_worker[menu['worker_id']].add(menu['session'])
                        texts, weights = [], []
                        for menu in reference:
                            weight = (1 / len(ref_workers) / len(sessions_by_worker[menu['worker_id']])
                                      / menus_per_session[menu['session']] / len(menu['candidates']))
                            texts.extend(candidate['text'] for candidate in menu['candidates'])
                            weights.extend([weight] * len(menu['candidates']))
                        vectorizer = TfidfVectorizer(
                            ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b',
                            lowercase=True, norm='l2', max_features=50000, dtype=np.float64)
                        try:
                            vectors = vectorizer.fit_transform(texts)
                        except ValueError as exc:
                            if 'empty vocabulary' not in str(exc):
                                raise
                            block['status'] = 'empty_reference_vocabulary'
                            block_out.write(json.dumps(block) + '\n')
                            context_counts[block['status']] += len(target)
                            continue
                        center = np.asarray(vectors.multiply(np.asarray(weights)[:, None]).sum(axis=0)).ravel()
                        block.update({'status': 'features_built', 'reference_candidates': len(texts),
                                      'vocabulary_size': len(vectorizer.vocabulary_),
                                      'reference_weight_sum': float(sum(weights))})
                        block_out.write(json.dumps(block) + '\n')
                        context_counts['reference_blocks'] += 1
                        for menu in target:
                            candidates = menu['candidates']
                            x = vectorizer.transform([candidate['text'] for candidate in candidates])
                            centrality = np.asarray(x @ center).ravel()
                            prefix = vectorizer.transform([menu['prefix']])
                            prefix_similarity = np.asarray((x @ prefix.T).toarray()).ravel()
                            common = {field: menu[field] for field in [
                                'menu_id', 'session', 'worker_id', *STRATUM, 'condition', 'topic',
                                'display_kind', 'inserted_units', 'retained_units', 'retention_fraction']}
                            common.update({
                                'menu_size': len(candidates),
                                'logged_id_agrees': menu['choice_alignment']['alignment_status'] == 'unique_api_text_confirms_logged_id',
                                'reference_workers': len(ref_workers), 'reference_sessions': len(ref_sessions),
                                'reference_menus': len(reference), 'outer_fold': fold,
                                'partition': ('descriptive' if fold == -1 else
                                              'test' if worker in held_out else 'train'),
                            })
                            for j, candidate in enumerate(candidates):
                                writer.writerow(dict(common, position=candidate['position'], candidate_id=candidate['id'],
                                                     chosen=candidate['id'] == menu['selected_id'],
                                                     centrality=float(centrality[j]), prefix_similarity=float(prefix_similarity[j]),
                                                     words=len(candidate['text'].split())))
                            context_counts['menus'] += 1
                            context_counts['candidate_rows'] += len(candidates)
                            if fold == -1:
                                coverage[menu['session']]['primary_feature_menus'] += 1
            contexts.append({'context': name, **dict(context_counts)})
            print(json.dumps(contexts[-1]), flush=True)
    with (output / 'session_coverage.jsonl').open('w') as handle:
        for row in coverage.values():
            handle.write(json.dumps(row) + '\n')
    with (output / 'excluded_menus.jsonl').open('w') as handle:
        for row in excluded_menus:
            handle.write(json.dumps(row) + '\n')
    report = {
        'stage': 'features_only_no_effect_summary_or_choice_model',
        'source_replay': str(replay.resolve()), 'source_ref': replay_summary['source_ref'],
        'metadata_sessions': len(sessions), 'metadata_workers': len(folds),
        'available_at_replay_start': replay_summary['available_at_run_start'],
        'acquisition_complete': replay_summary['acquisition_complete_at_run_start'],
        'replay_statuses': replay_summary['replay_statuses'],
        'session_exclusions': dict(Counter(row['session_exclusion'] for row in coverage.values() if row['session_exclusion'])),
        'all_recorded_menus': sum(row['all_menus'] for row in coverage.values()),
        'sessions_with_valid_new_supply': sum(row['valid_new_supply_menus'] > 0 for row in coverage.values()),
        'eligible_sessions_without_valid_new_supply': sum(
            row['session_exclusion'] is None and not row['valid_new_supply_menus'] for row in coverage.values()),
        'accepted_menus': sum(row['accepted_menus'] for row in coverage.values()),
        'eligible_choice_menus_at_least_two': sum(row['eligible_choice_menus_at_least_two'] for row in coverage.values()),
        'eligible_choice_menus_five': sum(row['eligible_choice_menus_five'] for row in coverage.values()),
        'menu_exclusions': dict(Counter(row['reason'] for row in excluded_menus)),
        'accepted_choice_exclusions': dict(sum((Counter(row['choice_exclusions']) for row in coverage.values()), Counter())),
        'contexts': contexts, 'seed': SEED,
        'reference_weighting': 'equal worker / session / original menu / candidate',
        'fold_reference_policy': 'exclude all outer-test workers, and additionally target worker for training menus',
    }
    (output / 'coverage.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build_features(args.replay, args.output)


if __name__ == '__main__':
    main()
