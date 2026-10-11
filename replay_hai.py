"""Recover displayed menus and exact accepted-text lineage from HAI event logs.

Input membership comes only from the two published processed essay indexes.
Raw replay and finite-corpus summaries are distinct from writer-level inference.
"""

import argparse
from collections import Counter
import json
from pathlib import Path


def units(text):
    value = text.encode('utf-16-le')
    return [value[i:i + 2] for i in range(0, len(value), 2)]


def as_text(value):
    # Image objects occupy one Quill document unit but contribute no text.
    try:
        return b''.join(value).decode('utf-16-le')
    except TypeError:
        return b''.join(unit for unit in value if isinstance(unit, bytes)).decode('utf-16-le')


def essay_part(text):
    return text.split('\n---\n')[-1].strip()


def apply_delta(doc, owners, ops, owner, candidate):
    """Quill offsets count UTF-16 units; attribution requires an exact string match."""
    inserted = ''.join(op['insert'] for op in ops if isinstance(op.get('insert'), str))
    target = candidate if candidate else None
    start = inserted.find(target) if target else -1
    matched = start >= 0
    left = len(units(inserted[:start])) if matched else 0
    right = left + len(units(target)) if matched else 0
    result, result_owners = [], []
    pos = insert_pos = attributed = 0
    for op in ops:
        if 'retain' in op:
            n = int(op['retain'])
            if pos + n > len(doc):
                raise ValueError(f'retain beyond current document: {pos}+{n}>{len(doc)}')
            result.extend(doc[pos:pos + n])
            result_owners.extend(owners[pos:pos + n])
            pos += n
        elif 'delete' in op:
            n = int(op['delete'])
            if pos + n > len(doc):
                raise ValueError(f'delete beyond current document: {pos}+{n}>{len(doc)}')
            pos += n
        elif 'insert' in op:
            if not isinstance(op['insert'], str):
                if isinstance(op['insert'], dict) and set(op['insert']) == {'image'}:
                    result.append(op['insert'])
                    result_owners.append(None)
                    continue
                raise ValueError('unsupported non-text Quill insertion')
            value = units(op['insert'])
            result.extend(value)
            for j in range(len(value)):
                tag = owner if matched and left <= insert_pos + j < right else None
                result_owners.append(tag)
                attributed += tag is not None
            insert_pos += len(value)
        else:
            raise ValueError(f'unknown delta operation: {list(op)}')
    result.extend(doc[pos:])
    result_owners.extend(owners[pos:])
    return result, result_owners, attributed, matched


def replay_log(events, path, condition, topic=None):
    """Replay a raw log without requiring a separately published final essay."""
    path = path.removeprefix('./')
    session = Path(path).stem
    assigned_prompt = next(e['currentDoc'] for e in events
                           if e['eventName'] == 'system-initialize' and e.get('currentDoc'))
    if topic is None:
        topic = assigned_prompt.strip().splitlines()[0]
    # Quill's empty document contains a terminal newline. It is not an insert
    # event; the observed system-initialize snapshots confirm this initial unit.
    initial = events[0].get('currentDoc') or '\n'
    doc, owners = units(initial), [None] * len(units(initial))
    menus = []
    active = pending = last_menu = None
    event_counts = Counter()
    snapshots = snapshot_mismatches = 0
    document_checks = []
    embed_insertions = []
    before_last_doc = doc
    before_last = as_text(doc)
    for ei, event in enumerate(events):
        before_last_doc = doc
        before_last = as_text(doc)
        name = event['eventName']
        event_counts[name] += 1
        if name == 'system-initialize' and event.get('currentDoc'):
            snapshots += 1
            snapshot_mismatches += event['currentDoc'] != before_last
        if name in ['suggestion-open', 'suggestion-reopen']:
            if active is not None:
                active['status'] = 'replaced_by_menu'
            suggestions = event.get('currentSuggestions', [])
            if name == 'suggestion-reopen' and last_menu is not None:
                candidates = [dict(s) for s in last_menu['candidates']]
                source_open = last_menu['source_open_index']
            else:
                candidates = [{
                    'id': int(s['index']), 'position': j,
                    'text': s.get('trimmed', s.get('original', '')),
                    'source': s.get('source'), 'probability': s.get('probability'),
                } for j, s in enumerate(suggestions)]
                source_open = ei
            if not candidates:
                active = None
                continue
            cursor_value = event.get('currentCursor')
            cursor = int(cursor_value) if cursor_value != '' and cursor_value is not None else len(doc)
            prefix = essay_part(as_text(doc[:cursor]))
            active = {
                'menu_id': f'{session}:{ei}', 'session': session,
                'path': path, 'condition': condition,
                'topic': topic,
                'event_index': ei, 'timestamp': event['eventTimestamp'],
                'source_open_index': source_open, 'display_kind': name,
                'prefix': ' '.join(prefix.split()[-200:]),
                'candidates': candidates, 'status': 'unresolved_end',
                'selected_id': None, 'selected_position': None,
                'inserted_units': 0, 'retained_units': 0,
                'insertion_match': None,
            }
            if len({s['id'] for s in candidates}) != len(candidates):
                raise ValueError(f'duplicate candidate IDs at event {ei}')
            menus.append(active)
            last_menu = active
        elif name == 'suggestion-select':
            if active is None:
                raise ValueError(f'selection without displayed menu at event {ei}')
            chosen = int(event['currentSuggestionIndex'])
            matches = [s for s in active['candidates'] if s['id'] == chosen]
            if len(matches) != 1:
                raise ValueError(f'candidate ID {chosen} missing from menu at event {ei}')
            active['selected_id'] = chosen
            active['selected_position'] = matches[0]['position']
            active['status'] = 'selected'
            pending = (active, matches[0]['text'])
            active = None
        elif name == 'suggestion-close' and active is not None:
            active['status'] = ('closed_by_user' if event['eventSource'] == 'user'
                                else 'closed_by_system')
            active = None
        delta = event.get('textDelta')
        if isinstance(delta, dict) and delta.get('ops'):
            for op in delta['ops']:
                if isinstance(op.get('insert'), dict):
                    embed_insertions.append({
                        'event_index': ei, 'event_source': event['eventSource'],
                        'types': list(op['insert']),
                    })
            take_pending = pending is not None and event['eventSource'] == 'api'
            owner = pending[0]['menu_id'] if take_pending else None
            candidate = pending[1] if take_pending else None
            doc, owners, count, matched = apply_delta(
                doc, owners, delta['ops'], owner, candidate)
            if take_pending:
                pending[0]['inserted_units'] += count
                pending[0]['insertion_match'] = matched
                pending = None
        if event.get('currentDoc'):
            document_checks.append({
                'event_index': ei, 'event_name': name,
                'initialization_anchor': ei == 0,
                'matches_before_event': event['currentDoc'] == before_last,
                'matches_after_event': event['currentDoc'] == as_text(doc),
            })
    retained = Counter(tag for tag in owners if tag is not None)
    for menu in menus:
        menu['retained_units'] = retained[menu['menu_id']]
        n = menu['inserted_units']
        menu['retention_fraction'] = menu['retained_units'] / n if n else None
    final = as_text(doc)
    summary = {
        'path': path, 'session': session, 'condition': condition,
        'topic': topic,
        'events': len(events), 'event_counts': dict(event_counts),
        'menus': len(menus), 'five_option_menus': sum(len(x['candidates']) == 5 for x in menus),
        'selected': sum(x['status'] == 'selected' for x in menus),
        'exact_insertion_matches': sum(x['insertion_match'] is True for x in menus),
        'snapshot_comparisons': snapshots, 'snapshot_mismatches': snapshot_mismatches,
        'final_units': len(doc),
    }
    endpoint = {
        'final_document': final, 'before_last_event_document': before_last,
        'document_snapshot_checks': document_checks,
        'final_document_units': len(doc),
        'final_text_utf16_units': sum(isinstance(unit, bytes) for unit in doc),
        'before_last_event_document_units': len(before_last_doc),
        'last_event_changes_document': doc != before_last_doc,
        'embed_insertions': embed_insertions,
        'final_embed_positions': [{'document_index': i, 'types': list(unit)}
                                  for i, unit in enumerate(doc) if isinstance(unit, dict)],
    }
    return menus, summary, endpoint


def replay(events, record, condition):
    """Compare HAI event replay with the released processed essay."""
    menus, summary, endpoint = replay_log(events, record['path'], condition)
    expected = record['essay'].strip()
    processed_topic = record['prompt'].strip().splitlines()[0]
    summary.update({
        'processed_topic': processed_topic,
        'topic_changed_in_processed': summary['topic'] != processed_topic,
        'final_matches_processed': essay_part(endpoint['final_document']) == expected,
        'excluding_last_matches_processed': essay_part(endpoint['before_last_event_document']) == expected,
        'processed_essay_characters': len(expected),
    })
    return menus, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, duplicates = {}, []
    indexed_rows = Counter()
    for condition in ['gpt3', 'instructgpt']:
        source = args.data / 'processed_data' / f'essays_{condition}.jsonl'
        for line_number, line in enumerate(source.read_text().splitlines(), 1):
            row = json.loads(line)
            path = row['path'].removeprefix('./')
            indexed_rows[condition] += 1
            if path in records:
                old_condition, old_row = records[path]
                if old_condition != condition or old_row != row:
                    raise ValueError(f'conflicting processed records for {path}')
                duplicates.append({'path': path, 'condition': condition, 'line': line_number})
            else:
                records[path] = (condition, row)
    counts = Counter()
    missing, failures, sessions = [], [], []
    with (args.output / 'menus.jsonl').open('w') as out:
        for path, (condition, row) in records.items():
            file = args.data / path
            if not file.is_file():
                missing.append(path)
                continue
            try:
                events = [json.loads(s) for s in file.read_text().splitlines() if s.strip()]
                menus, summary = replay(events, row, condition)
            except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                failures.append({'path': path, 'error': str(exc)})
                continue
            sessions.append(summary)
            counts.update(x['status'] for x in menus)
            for menu in menus:
                out.write(json.dumps(menu, ensure_ascii=False) + '\n')
    with (args.output / 'sessions.jsonl').open('w') as out:
        for row in sessions:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = {
        'membership': 'published processed indexes; identical repeated row counted once',
        'indexed_rows': dict(indexed_rows), 'unique_paths': len(records),
        'duplicate_rows': duplicates, 'available_logs': len(records) - len(missing),
        'missing_logs': missing, 'replay_failures': failures,
        'replayed_sessions': len(sessions), 'by_condition': dict(Counter(x['condition'] for x in sessions)),
        'events': sum(x['events'] for x in sessions),
        'menu_statuses': dict(counts),
        'five_option_menus': sum(x['five_option_menus'] for x in sessions),
        'selected_menus': sum(x['selected'] for x in sessions),
        'exact_insertion_matches': sum(x['exact_insertion_matches'] for x in sessions),
        'snapshot_mismatches': sum(x['snapshot_mismatches'] for x in sessions),
        'final_matches_processed': sum(x['final_matches_processed'] for x in sessions),
        'excluding_last_matches_processed': sum(x['excluding_last_matches_processed'] for x in sessions),
        'topic_changed_in_processed': sum(x['topic_changed_in_processed'] for x in sessions),
        'writer_ids_available': False,
        'retention_unit': 'UTF-16 code units of exactly matched selected suggestion',
        'scope': 'replay validation, not evidence of population-level selection effects',
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['missing_logs','replay_failures']}, ensure_ascii=False, indent=2))
    print('missing_logs', len(missing), 'replay_failures', len(failures))


if __name__ == '__main__':
    main()
