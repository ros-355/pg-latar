 PG-LATAR MCC simulation benchmark

Code and synthetic benchmark outputs supporting **Load Conditioned Thermal Anomaly Detection in Motor Control Centres under False Alarm Calibration and Model Mismatch**.

**Authors:** Roshan Kumar Sharma, Ujjwal Prasad Dahal and Naresh Kumar Ray  
**Affiliation:** Department of Electrical Engineering, IOE Purwanchal Campus, Dharan, Nepal  
**Corresponding author:** Roshan Kumar Sharma, pur078bel042@ioepc.edu.np

This is a simulation-only study. No factory telemetry, thermal-camera images, maintenance records or verified field faults are included. Results describe synthetic generator populations and do not establish operational fault diagnosis or protection performance.

## Repository contents

| Path | Contents |
| --- | --- |
| `experiments/run_experiments.py` | Seeded generator, 14 scores, calibration, evaluation, uncertainty analyses and figures |
| `tests/test_protocol.py` | Deterministic protocol checks |
| `results/*.csv` | Archived aggregate benchmark data and statistical summaries |
| `results/run_manifest.json` | Generator settings, seed partitions and original run arguments |
| `results/*.pdf` | Archived vector figures |
| `manuscript/` | Revised manuscript in PDF, Word and LaTeX with vector figures |
| `docs/DATA_DICTIONARY.md` | File descriptions, column inventory and metric definitions |
| `docs/REPRODUCIBILITY.md` | Setup, tests, full experiment and manuscript commands |
| `docs/DATA_AVAILABILITY.md` | Guidance for linking this repository in the paper |
| `provenance/` | Supplied original environment snapshot, run log and exit status |
| `CITATION.cff` | Author and software citation metadata |
| `SHA256SUMS.json` | Integrity checks for the packaged files |

## Quick start

Use Python 3.11 in a virtual environment. Install `requirements.txt`, then run:

```sh
python tests/test_protocol.py
```

See [reproduction instructions](docs/REPRODUCIBILITY.md) for the complete full-run command. A smoke test is a plumbing check only and must not replace the archived full-run results in the paper.

## Data scope and provenance

The CSVs contain aggregate outputs from the supplied full run recorded on 23 October 2026. The repository was assembled on 29 October 2026. Raw minute-by-minute trajectories and individual sequence scores were not archived in the supplied package. The generator, seed rules and configuration are provided to regenerate synthetic sequences. Exact results can depend on library versions and floating-point behaviour.

The original numerical experiment was not rerun while preparing this repository. Python source syntax, JSON files, CSV structure and archive checksums were checked. Protocol-test execution requires the dependencies listed in `requirements.txt`.

## Interpretation

At the nominal 0.5/day calibration target, PG-LATAR detects 400/400 injected events at 0.437 false episodes per clean system-day. The multi-step comparator detects 400/400 at 0.448/day with shorter delay. Under parameter shift, the peer term increases false-alarm burden relative to the no-peer ablation. The main events are relatively strong; inspect the severity and chance-hit controls before drawing conclusions.

False alarms are system-level episodes, not phase-level samples. A pre-existing active alarm may count as a hit during an event window. Calibration targets and held-out realized rates differ. Post-hoc interpolation is descriptive only. Historical hyperparameter selection is unverified. No general superiority or field-validation claim is made.

## Citation and rights

Use `CITATION.cff` or cite the manuscript. No publication DOI, public repository URL, release tag or journal acceptance is claimed in this archive. The authors have not yet selected an open-source or data licence; see [RIGHTS.md](RIGHTS.md).

## Declarations

This research received no funding. The authors declare no conflicts of interest.
