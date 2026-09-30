"""Experiment-only code. Nothing here is imported by scripts/run_rural.py or the map build.

Every function in this package reads the existing str_suitability pipeline as a library
(market_features, label_performance, build_grid, road distance, ...). None of those
functions are modified. This package only adds new, separate feature and validation code
so the current thesis implementation on the 2ndver branch stays untouched.
"""
