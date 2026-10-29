# Comprehensive relationship study

Three saved drafts use the same frozen 8,329-instrument Parquet snapshot, including reference ETFs. Only eligible US stocks are candidates. Each fold selects up to 1,000 names by training-period liquidity; this is not an exhaustive test of the entire US market or every possible basket. Current catalogue and incomplete download biases remain.

Open Research Studio → Experiments. Queue A, B and C using their Run buttons; the shared engine executes one at a time. These three studies were saved as drafts, not automatically launched. Collection remains paused as requested through the app.

| Study | Observations | Relationship controls | Question |
|---|---|---|---|
| A · Daily raw returns | Daily | None | Which apparent relationships exist before controlling common exposures? |
| B · Daily market-sector controls | Daily | SPY + sector ETFs + IWM/TLT | Which associations survive the declared common-influence controls? |
| C · Weekly market-sector controls | Weekly | Same controls | Do findings survive a different observation timescale? |

Comparisons change the candidate-neighbor model; Johansen basket coefficients are still fitted on training log prices. Long-only groups use raw return correlations, so A and B need not produce different long-only allocations. Differences between A/B basket results also reflect different candidate sets, not a clean causal effect of factor removal. Weekly observations have fewer samples and are not automatically better.

Shared specification: 2016-10-28–2026-10-27, three-year rolling training, twelve-month validation/test, annual advances, twenty-session boundary gaps, six non-overlapping test windows. Up to 100,000 candidate attempts per fold, sizes 3/5/8, seed42, 98% coverage, $10m minimum training daily turnover, no unknown-volume exception. Up to 600,000 attempts per experiment / 1.8m total; actual counts can be lower when unique candidates or eligible assets are exhausted.

Validation chooses the predeclared diversified portfolio or a basket rule from entry z-scores 1.5/2/2.5 and holding limits 10/20 sessions. Later test observations do not choose that fold's rule. Twelve long-only holdings maximum, five simultaneous disjoint baskets, gross/net ceilings1, single-name cap10%, volatility target12%, drawdown stop35%. $100k starting capital; commission1bps, full spread10bps, slippage5bps, impact coefficient0.1, borrow6% annual, cash0%, funding6%. Known-sector cap35% only applies where dated sector metadata exist. These are assumptions, not observed historical execution prices or borrowing.

Resources: 128 thread ceiling for numerical work; basket pool now allows up to61 single-threaded child processes on Windows, limited further by CPU configuration, RAM and available batches. RAM ceiling64GiB, OS reserve24GiB, mandatory CPU sensor, stop85°C, six-hour duration ceiling per experiment. 5,000 bootstrap repetitions, twenty-session blocks. CPU use varies across loading, fitting, accounting and reporting. The GPU is not used.

## Read all results, including failures

Compare per-window performance and drawdowns, conditional associations, model/group changes, selected strategy family, turnover/costs, exposures and doubled-cost results. Check whether apparent risk reduction comes from more cash or lower exposure. Broad consistency across windows is more informative than one winning year; do not select the best of three historical returns and call that an independent test. If no baskets pass training/validation filters, that is a valid outcome.

Unknown short availability is permitted solely for these exploratory basket studies. No borrow inventory is supplied, so basket performance is not verified as executable. The existing Learning test1 draft is the long-only alternative without that assumption. Multiple testing, survivorship bias, provider anomalies, missing delistings and repeated historical inspection remain limitations. A genuinely new forward paper record is required for fresh evidence.

## Capacity verification

A guarded real-data check completed 10,000 basket attempts across one fold using61 workers in23.81s worker time (28.15s observed). Sampled maximum worker-tree RSS10.47GiB, CPU temperature83.3°C. The CPU-percentage samples did not capture a high utilization peak; no sustained utilization or speedup claim is made. This short check does not establish the temperature of a longer soak. The85°C stop guard stays active. See CPU_RESEARCH_VALIDATION.json.

The settings and draft IDs are in COMPREHENSIVE_STUDY.json. Completed historical experiment records and earlier source fingerprints are retained; the expanded worker count changes the code fingerprint and prevents incompatible checkpoint reuse.
