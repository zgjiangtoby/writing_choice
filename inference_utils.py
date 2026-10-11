"""Explicit normal-approximation behavior when paired resampling is degenerate."""

import math

from scipy.stats import norm


NUMERICAL_ZERO = 1e-12


def normal_tail_probability(delta, standard_error):
    delta, standard_error = float(delta), float(standard_error)
    assert math.isfinite(delta) and math.isfinite(standard_error) and standard_error >= 0
    if standard_error <= NUMERICAL_ZERO:
        if abs(delta) <= NUMERICAL_ZERO:
            return 1., "numerically_zero_contrast"
        return None, "nonzero_contrast_with_degenerate_resampling"
    return float(2 * norm.sf(abs(delta) / standard_error)), "normal_approximation"


def adjusted_family(rows, input_key, output_key, holm):
    # An unestimable row stays in the original family, but receives no claimed p-value.
    inputs = [1. if row[input_key] is None else float(row[input_key]) for row in rows]
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in inputs)
    adjusted = holm(inputs)
    for row, value, adjustment in zip(rows, inputs, adjusted):
        row[output_key + "_family_input"] = value
        row[output_key] = None if row[input_key] is None else float(adjustment)
    return all(row[input_key] is not None for row in rows)
