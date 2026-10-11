"""Replay metadata-defined CoAuthor sessions.

All 1,445 metadata members remain in sessions.jsonl, with each log's availability
and replay status. Replayed final text is derived from events.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from replay_hai import replay_log, units


SETTING_FIELDS = {
    'n': 'currentN', 'max_tokens': 'currentMaxToken',
    'temperature': 'currentTemperature', 'top_p': 'currentTopP',
    'presence_penalty': 'currentPresencePenalty',
    'frequency_penalty': 'currentFrequencyPenalty',
}


def settings(event):
    return {key: float(event[field]) if event.get(field) not in (None, '') else None
            for key, field in SETTING_FIELDS.items()}


def metadata_members(data):
    members = []
    seen = set()
    for kind in ['argumentative', 'creative']:
        source = data / 'data/CoAuthor/add_data' / f'metadata_{kind}.csv'
        with source.open(newline='') as handle:
            rows = sorted(csv.DictReader(handle), key=lambda row: row['session_id'])
        for row in rows:
            session = row['session_id']
            if session in seen:
                raise ValueError(f'duplicate metadata session: {session}')
            seen.add(session)
            if not row['worker_id'] or not row['prompt_code']:
                raise ValueError(f'missing metadata writer or prompt: {session}')
            members.append({
                'session': session, 'worker_id': row['worker_id'],
                'writing_type': kind, 'prompt_code': row['prompt_code'],
                'temperature': float(row['temperature']),
                'frequency_penalty': float(row['frequency_penalty']),
                'path': f'data/CoAuthor/coauthor-v1.0/{session}.jsonl',
                'metadata': row,
            })
    return members


def metadata_support(members):
    strata = {}
    for row in members:
        key = tuple(row[field] for field in [
            'writing_type', 'prompt_code', 'temperature', 'frequency_penalty'])
        strata.setdefault(key, Counter())[row['worker_id']] += 1
    output = []
    remaining_writer_distribution = Counter()
    unsupported = 0
    for key, writers in sorted(strata.items()):
        sessions = sum(writers.values())
        remaining = len(writers) - 1
        remaining_writer_distribution[str(remaining)] += sessions
        unsupported += sessions if remaining == 0 else 0
        output.append({
            'writing_type': key[0], 'prompt_code': key[1],
            'temperature': key[2], 'frequency_penalty': key[3],
            'metadata_sessions': sessions, 'metadata_workers': len(writers),
            'remaining_workers_after_excluding_current': remaining,
            'reference_sessions_after_excluding_current_worker_min': sessions - max(writers.values()),
            'reference_sessions_after_excluding_current_worker_max': sessions - min(writers.values()),
            'sessions_without_other_worker': sessions if remaining == 0 else 0,
        })
    return {
        'basis': 'all metadata members; no menu contents or outcomes used',
        'strata': output,
        'session_weighted_remaining_worker_distribution': dict(sorted(
            remaining_writer_distribution.items(), key=lambda pair: int(pair[0]))),
        'sessions_without_other_worker': unsupported,
        'limitation': 'metadata support does not imply available valid candidate supply after replay',
    }


def align_choices(events):
    """Observe actual inserted candidate text without assuming index semantics.

    Only a unique exact candidate, optionally surrounded by whitespace, resolves
    a choice. Raw event dictionaries are never changed. Hover is not consulted.
    """
    prepared = list(events)
    evidence = {}
    active = last = pending = None
    for ei, event in enumerate(events):
        name = event['eventName']
        if name in ['suggestion-open', 'suggestion-reopen']:
            candidates = (last if name == 'suggestion-reopen' and last is not None
                          else event.get('currentSuggestions', []))
            active = (ei, candidates) if candidates else None
            if candidates:
                last = candidates
        elif name == 'suggestion-close':
            active = None
        elif name == 'suggestion-select' and active is not None:
            menu_index, candidates = active
            raw = int(event['currentSuggestionIndex'])
            evidence[menu_index] = {
                'menu_event_index': menu_index, 'selection_event_index': ei,
                'logged_selected_index': raw,
                'candidate_ids_in_array_order': [int(candidate['index']) for candidate in candidates],
                'alignment_status': 'no_following_api_delta',
                'resolved_selected_id': None, 'resolved_selected_position': None,
                'logged_own_id_matches_api_text': None,
                'logged_array_position_matches_api_text': None,
                'index_interpretations_differ': None,
            }
            pending = (evidence[menu_index], candidates)
            active = None
        delta = event.get('textDelta')
        if pending is None or event['eventSource'] != 'api' or not isinstance(delta, dict) or not delta.get('ops'):
            continue
        row, candidates = pending
        inserted = ''.join(op['insert'] for op in delta['ops'] if isinstance(op.get('insert'), str))
        matches = []
        for position, candidate in enumerate(candidates):
            text = candidate.get('trimmed', candidate.get('original', ''))
            start = inserted.find(text) if text else -1
            if start >= 0 and not inserted[:start].strip() and not inserted[start + len(text):].strip():
                matches.append((position, int(candidate['index']), text, start))
        raw = row['logged_selected_index']
        own = next((position for position, candidate in enumerate(candidates) if int(candidate['index']) == raw), None)
        array = raw if 0 <= raw < len(candidates) else None
        row.update({
            'api_event_index': ei, 'api_inserted_text': inserted,
            'exact_candidate_match_count': len(matches),
            'logged_own_id_matches_api_text': any(position == own for position, _, _, _ in matches),
            'logged_array_position_matches_api_text': any(position == array for position, _, _, _ in matches),
            'index_interpretations_differ': own is not None and array is not None and own != array,
        })
        if len(matches) == 1:
            position, chosen, text, start = matches[0]
            row.update({
                'resolved_selected_id': chosen, 'resolved_selected_position': position,
                'alignment_status': ('unique_api_text_confirms_logged_id' if chosen == raw
                                     else 'unique_api_text_conflicts_with_logged_id'),
                'api_match_kind': ('whole_insert_exact' if inserted == text
                                   else 'exact_candidate_with_boundary_whitespace'),
                'api_prefix_outside_candidate': inserted[:start],
                'api_suffix_outside_candidate': inserted[start + len(text):],
            })
            si = row['selection_event_index']
            prepared[si] = dict(events[si], currentSuggestionIndex=chosen)
        else:
            row['alignment_status'] = 'ambiguous_api_text' if matches else 'no_matching_candidate_api_text'
        pending = None
    return prepared, evidence


def replay_member(events, member, choice_evidence):
    kind = member['writing_type']
    condition = f"coauthor_{kind}_t{member['temperature']:g}_fp{member['frequency_penalty']:g}"
    menus, summary, endpoint = replay_log(
        events, member['path'], condition, topic=member['prompt_code'])
    native = {key: member[key] for key in [
        'worker_id', 'writing_type', 'prompt_code', 'temperature', 'frequency_penalty']}
    configurations = Counter()
    setting_mismatches = []
    for menu in menus:
        # Reopened menus inherit the configuration of their source generation.
        observed = settings(events[menu['source_open_index']])
        menu.update(native)
        menu['generation_settings'] = observed
        menu['settings_match_metadata'] = all(
            observed[key] == member[key] for key in ['temperature', 'frequency_penalty'])
        menu['supply_stratum'] = {
            'writing_type': kind, 'prompt_code': member['prompt_code'],
            'temperature': member['temperature'],
            'frequency_penalty': member['frequency_penalty'],
        }
        if menu['display_kind'] == 'suggestion-open':
            configurations[tuple(observed.items())] += 1
        if not menu['settings_match_metadata']:
            setting_mismatches.append(menu['menu_id'])
        if menu['status'] == 'selected':
            alignment = choice_evidence[menu['event_index']]
            menu['choice_alignment'] = alignment
            menu['analysis_choice_eligible'] = alignment['resolved_selected_id'] is not None
            menu['selected_id'] = alignment['resolved_selected_id']
            menu['selected_position'] = alignment['resolved_selected_position']
    checkpoints = endpoint.pop('document_snapshot_checks')
    noninitial = [row for row in checkpoints if not row['initialization_anchor']]
    unmatched = [row for row in noninitial
                 if not row['matches_before_event'] and not row['matches_after_event']]
    recorded_events = int(float(member['metadata']['num_event']))
    summary.update(native)
    summary.update({
        'replay_status': 'replayed',
        'metadata_events': recorded_events,
        'event_count_matches_metadata': len(events) == recorded_events,
        'metadata_num_query': int(float(member['metadata']['num_query'])),
        'observed_suggestion_get_events': summary['event_counts'].get('suggestion-get', 0),
        'observed_new_menu_events': summary['event_counts'].get('suggestion-open', 0),
        'metadata_num_selected': int(float(member['metadata']['num_selected'])),
        'selected_count_matches_metadata': summary['selected'] == int(float(member['metadata']['num_selected'])),
        'menu_sizes': dict(Counter(str(len(menu['candidates'])) for menu in menus)),
        'selected_menu_sizes': dict(Counter(str(len(menu['candidates'])) for menu in menus if menu['status'] == 'selected')),
        'new_menus': sum(menu['display_kind'] == 'suggestion-open' for menu in menus),
        'reopened_menus': sum(menu['display_kind'] == 'suggestion-reopen' for menu in menus),
        'insertion_mismatches': sum(menu['insertion_match'] is False for menu in menus),
        'selected_without_matched_insertion': sum(menu['status'] == 'selected' and menu['insertion_match'] is not True for menu in menus),
        'choice_alignment_counts': dict(Counter(row['alignment_status'] for row in choice_evidence.values())),
        'choices_with_unique_api_text': sum(menu.get('analysis_choice_eligible', False) for menu in menus),
        'generation_configurations': [dict(values, new_menus=count) for values, count in configurations.items()],
        'menu_settings_mismatches': setting_mismatches,
        'document_snapshot_checks': checkpoints,
        'noninitial_document_snapshots': len(noninitial),
        'noninitial_snapshot_mismatches': len(unmatched),
        'final_event_name': events[-1]['eventName'],
        'final_event_source': events[-1]['eventSource'],
        'last_event_changes_text': endpoint['final_document'] != endpoint['before_last_event_document'],
        'last_event_changes_document': endpoint['last_event_changes_document'],
        'embed_insertions': endpoint['embed_insertions'],
        'final_embed_count': len(endpoint['final_embed_positions']),
        'final_text_utf16_units': endpoint['final_text_utf16_units'],
        'independent_final_text_available': False,
        'final_matches_independent_text': None,
        'endpoint_validation': 'replayed_all_events; no independently published final-text gold supplied',
    })
    endpoint.update({
        'session': member['session'], **native,
        'artifact_kind': 'derived_replay_text_not_independent_gold',
        'text_export_excludes_embeds': True,
        'before_last_event_utf16_units': len(units(endpoint['before_last_event_document'])),
    })
    return menus, summary, endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    members = metadata_members(args.data)
    started = datetime.now(timezone.utc).isoformat()
    available = {row['session'] for row in members if (args.data / row['path']).is_file()}
    args.output.mkdir(parents=True, exist_ok=True)
    support = metadata_support(members)
    (args.output / 'metadata_support.json').write_text(json.dumps(support, ensure_ascii=False, indent=2) + '\n')
    rows = []
    menu_statuses = Counter()
    with (args.output / 'menus.jsonl').open('w') as menu_out, (args.output / 'endpoints.jsonl').open('w') as endpoint_out, (args.output / 'choice_alignment.jsonl').open('w') as alignment_out:
        for member in members:
            row = {key: value for key, value in member.items() if key != 'metadata'}
            row['metadata'] = member['metadata']
            if member['session'] not in available:
                row['replay_status'] = 'not_available_at_run_start'
                rows.append(row)
                continue
            events = None
            try:
                events = [json.loads(line) for line in (args.data / member['path']).read_text().splitlines() if line.strip()]
                prepared, evidence = align_choices(events)
                row['choice_alignment_counts'] = dict(Counter(item['alignment_status'] for item in evidence.values()))
                for item in evidence.values():
                    alignment_out.write(json.dumps(dict(item, session=member['session']), ensure_ascii=False) + '\n')
                menus, result, endpoint = replay_member(prepared, member, evidence)
            except (OSError, ValueError, KeyError, TypeError, UnicodeError, StopIteration, IndexError) as exc:
                row.update({'replay_status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
                if events is not None:
                    row['events_observed'] = len(events)
                    row['raw_event_counts'] = dict(Counter(event.get('eventName', '<missing>') for event in events))
            else:
                row.update(result)
                for menu in menus:
                    menu_out.write(json.dumps(menu, ensure_ascii=False) + '\n')
                endpoint_out.write(json.dumps(endpoint, ensure_ascii=False) + '\n')
                menu_statuses.update(menu['status'] for menu in menus)
            rows.append(row)
    with (args.output / 'sessions.jsonl').open('w') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    replayed = [row for row in rows if row['replay_status'] == 'replayed']
    by_type = {}
    for kind in ['argumentative', 'creative']:
        group = [row for row in rows if row['writing_type'] == kind]
        by_type[kind] = {
            'metadata_sessions': len(group),
            'metadata_workers': len({row['worker_id'] for row in group}),
            'replay_statuses': dict(Counter(row['replay_status'] for row in group)),
        }
    summary = {
        'source_ref': '8ba055111af24cb4c5fd1eabe759f80948da5fb9',
        'membership': 'all sessions in metadata_argumentative.csv and metadata_creative.csv',
        'started_at_utc': started,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'metadata_sessions': len(members),
        'metadata_workers': len({row['worker_id'] for row in members}),
        'available_at_run_start': len(available),
        'acquisition_complete_at_run_start': len(available) == len(members),
        'replay_statuses': dict(Counter(row['replay_status'] for row in rows)),
        'by_writing_type': by_type,
        'events_in_replayed_sessions': sum(row['events'] for row in replayed),
        'menu_statuses': dict(menu_statuses),
        'five_option_menus': sum(row['five_option_menus'] for row in replayed),
        'selected_menus': sum(row['selected'] for row in replayed),
        'exact_insertion_matches': sum(row['exact_insertion_matches'] for row in replayed),
        'insertion_mismatches': sum(row['insertion_mismatches'] for row in replayed),
        'selected_without_matched_insertion': sum(row['selected_without_matched_insertion'] for row in replayed),
        'choice_alignment_counts_in_available_logs': dict(sum((Counter(row.get('choice_alignment_counts', {})) for row in rows), Counter())),
        'choices_with_unique_api_text_in_replayed_sessions': sum(row['choices_with_unique_api_text'] for row in replayed),
        'sessions_with_metadata_event_count_mismatch': sum(not row['event_count_matches_metadata'] for row in replayed),
        'sessions_with_metadata_selection_count_mismatch': sum(not row['selected_count_matches_metadata'] for row in replayed),
        'menus_with_metadata_settings_mismatch': sum(len(row['menu_settings_mismatches']) for row in replayed),
        'noninitial_document_snapshots': sum(row['noninitial_document_snapshots'] for row in replayed),
        'noninitial_snapshot_mismatches': sum(row['noninitial_snapshot_mismatches'] for row in replayed),
        'sessions_with_embed_insertions': sum(bool(row['embed_insertions']) for row in replayed),
        'embed_insertions': sum(len(row['embed_insertions']) for row in replayed),
        'final_embeds': sum(row['final_embed_count'] for row in replayed),
        'text_export_excludes_embeds': True,
        'independent_final_text_available': False,
        'final_matches_independent_text': None,
        'retention_unit': 'UTF-16 code units of exactly matched selected suggestion',
        'scope': 'schema and replay only; partial arrivals do not estimate corpus effects',
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
