# BCI Analysis

This repository contains tools and notes for analyzing brain-computer interface (BCI) data.

## Conventions

See [CODE_PRACTICES.md](CODE_PRACTICES.md) for the code and analysis conventions used here
(utilities-vs-notebook split, the generated-notebook workflow, verification requirements,
confound control, and dataset-specific gotchas).

## Purpose

The project is intended to support exploratory analysis, preprocessing, and interpretation of BCI datasets as the workflow evolves.

## Local Install

From the repository root, install the local package into the active notebook environment:

```bash
pip install -e .
```

Then import analysis helpers with:

```python
from bci_analysis.behavior import *
```
