"""
Thin wrapper that loads the Week 2 rule-based scoring engine
(scripts/03_rule_based_baseline.py) so the FastAPI backend can fall back to it
when no trained model is available, or when explicitly requested via
?method=rule_based.

We load by file path (rather than a normal package import) because the
script filename starts with a digit, which isn't a valid Python module name.
"""

import importlib.util
import os

_HERE = os.path.dirname(__file__)
_SCRIPT_PATH = os.path.join(_HERE, "..", "..", "scripts", "03_rule_based_baseline.py")

_spec = importlib.util.spec_from_file_location("rule_based_baseline", _SCRIPT_PATH)
_rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rb)

# Re-export the public API
compute_impact_score = _rb.compute_impact_score
severity_band = _rb.severity_band
recommend_resources = _rb.recommend_resources
score_event = _rb.score_event
time_multiplier = _rb.time_multiplier
