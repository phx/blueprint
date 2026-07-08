# Installation Guide

This guide covers the local setup for The Divine Blueprint validation harness.

## Prerequisites

- Python 3.9 or newer
- `pip`
- A POSIX-like shell for `verify.sh` on macOS/Linux

## Setup

```bash
git clone https://github.com/phx/blueprint.git
cd blueprint
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows users can activate the environment with:

```powershell
.venv\Scripts\activate
```

## Verify

Run the complete active-core validation suite:

```bash
cd code
python validate_all.py
```

Or run the same suite directly:

```bash
cd code
python -m pytest --cov=core --cov-report=term-missing
```

Expected result for the current release:

```text
238 passed, 0 skipped
TOTAL 1818 statements, 0 missed, 100% active-core coverage
```

## Focused Checks

```bash
cd code
./verify.sh --paper
./verify.sh --self-replication
./verify.sh --quick
```

The validation suite checks internal consistency among the paper, formula
registry, validation matrix, code, and tests. It does not prove external
empirical truth.
