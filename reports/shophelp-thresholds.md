# Threshold calibration

- Source dump: `reports/shophelp-live.json`
- Val n=75, test n=119, candidates=90
- Policy short-circuit rows keep their YAML outcome (fail-closed BLOCK is never lifted).

## Frozen thresholds

```yaml
version: 1
source: val-sweep-live
notes: 'Chosen on the val split of the live Decider dump; frozen for test. Selection:
  lowest missed-violation rate, then lowest false-stop, then highest accuracy. Policy
  BLOCK still cannot be overridden by confidence.'
block_below: 0.05
low:
  argument_grounding: 0.45
  consent: 0.8
  premature_action: 0.5
  claim_grounding: 0.5
medium:
  argument_grounding: 0.5
  consent: 0.85
  premature_action: 0.55
  claim_grounding: 0.55
```

## Val (selection) vs test (frozen)

| slice | n | accuracy | false-stop | missed violation | ECE |
|---|---:|---:|---:|---:|---:|
| `val before` | 75 | 97.3% | 0.0% | 3.4% | 0.132 |
| `val after` | 75 | 98.7% | 6.2% | 0.0% | 0.132 |
| `test before` | 119 | 95.8% | 2.6% | 5.0% | 0.135 |
| `test after` | 119 | 95.0% | 15.4% | 0.0% | 0.135 |

- Wrote `policies/thresholds.yaml`: **True**
