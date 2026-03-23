# Diagnostics Scripts

This folder contains exploratory and workflow validation scripts that were historically named with `test_*.py`.

These scripts are **diagnostic/integration utilities**, not strict unit tests.

## Run diagnostics manually

From project root:

- `python diagnostics/test_initialization.py`
- `python diagnostics/test_ml_prediction.py`
- `python diagnostics/test_mp_end_to_end.py`

## Note

`pytest` discovery is now scoped to the `tests/` folder via `pytest.ini`, so these diagnostics are not collected as unit tests.
