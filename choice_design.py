"""Five-option menu features and equal topic/session/menu weights."""

import numpy as np


FEATURES = ['prefix_similarity', 'log1p_words', 'position_1', 'position_2',
            'position_3', 'position_4', 'centrality']


def weights(meta):
    n_topics = meta['topic'].nunique()
    log_counts = meta.groupby('topic')['session'].nunique()
    menu_counts = meta.groupby(['topic', 'session']).size()
    return np.array([1 / n_topics / log_counts[r.topic] / menu_counts[(r.topic, r.session)]
                     for r in meta.itertuples()])


def pack(frame):
    frame = frame.sort_values(['menu_id', 'position']).copy()
    grouped = frame.groupby('menu_id', sort=True)
    if not (grouped.size() == 5).all() or not (grouped['chosen'].sum() == 1).all():
        raise ValueError('Each accepted five-option menu must have exactly one chosen candidate')
    pos = frame['position'].to_numpy()
    raw = np.column_stack([
        frame['prefix_similarity'].to_numpy(), np.log1p(frame['words'].to_numpy()),
        *[(pos == p).astype(float) for p in range(1, 5)],
        frame['centrality'].to_numpy(),
    ]).reshape(-1, 5, len(FEATURES))
    raw -= raw.mean(axis=1, keepdims=True)
    chosen = frame['chosen'].to_numpy().reshape(-1, 5).argmax(axis=1)
    meta = grouped[['session', 'topic']].first().reset_index()
    return raw, chosen, meta
