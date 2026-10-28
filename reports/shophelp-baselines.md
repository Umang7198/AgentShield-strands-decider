# ShopHelp gate vs baselines

Gold labels are for the **full** pipeline. `none` never stops a call. `policy` is YAML only. `full` is policy then Decider then the matrix. `llm-judge` answers the same noul questions (`arg:*`, `consent`, `claim`) with the host LLM.

- Live dump: `reports/shophelp-live.json` (question set `v2`, model `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`)
- Judge dump: `reports/shophelp-judge.json` (model `claude-sonnet-4.6`)
- Judge estimated USD: 0.023745 (input 119871, output 9608)

| system | n | accuracy | false-stop | missed violation | ECE | p50 ms | p95 ms | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `none` | 194 | 28.4% | 0.0% | 100.0% | n/a | 0 | 0 | unguarded |
| `policy` | 194 | 46.9% | 0.0% | 74.1% | n/a | 0 | 0 | YAML only |
| `full / decider-2B` | 194 | 96.4% | 1.8% | 4.3% | 0.134 | 374 | 585 | AgentShield |
| `llm-judge` | 194 | 56.2% | 0.0% | 2.9% | 0.042 | 1733 | 2216 | ~$0.0237 |

See `docs/benchmark-results.md` for sample size and limitations.
