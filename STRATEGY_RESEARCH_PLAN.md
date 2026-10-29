# Research protocol: monthly stock momentum with a diversification test

Status: research specification, 29 October 2026. Not implemented in the current strategy selector; no momentum backtest or profit claim has been produced. The current app supports diversified fixed allocations and basket mean reversion, not monthly momentum ranking. This proposal is a learning experiment, not an instruction to invest.

## Question and rationale

Do stocks with stronger medium-term past returns continue to outperform a comparable liquid-stock universe after costs? Separately, can limits on correlated holdings improve the tradeoff between return and drawdown?

Cross-sectional momentum has an established research literature. Kenneth French's monthly momentum methodology ranks prior months2–12 and forms winner-minus-loser portfolios. Our proposed long-only portfolio is an adaptation, not a replication of that factor. Slow information response and investor behaviour are possible explanations, not mechanisms proven by our data. Momentum is widely known and can suffer severe reversals; a famous anomaly is not a private edge.

## What the existing runs tell us

All three comprehensive studies completed. Their validation-selected portfolios earned approximately4.88%,5.13%,5.57% annualised, versus8.59% for always-diversified long-only and15.13% for SPY. Their equity exposures and risks differ, so these are not matched-risk alpha estimates. Existing conditional/basket findings contain unresolved ticker identities and cannot establish executable arbitrage. STRATEGY_RESEARCH_EVIDENCE.json records the actual results, source IDs and input hashes. Equal-weight returns must not be used as an excuse to select the historical winning constituents with hindsight.

## Data gates before a credible stock test

1. Resolve security identity across ticker changes and reuse. YY/JOYY is a confirmed rename; BBT/TFC requires date-specific identity review, and current BBT metadata identifies a different company. Related share classes can be valid distinct instruments but belong to the same issuer exposure. Flag near-duplicate histories for review; do not indiscriminately merge every correlated pair.
2. Require verified dated trading identity for every included listing. Quarantine unresolved tickers until reviewed. Store mapping provenance and known-at dates; do not map today's ticker backwards by assumption.
3. Review splits, dividend adjustments, stale/zero-return prices, large jumps, calendar and volume. Current close×volume is only a liquidity proxy where adjustment compatibility is unverified. A historical dollar-price filter requires appropriate unadjusted prices, not today's back-adjusted levels.
4. Require adequate previous observations, and never require a stock to survive until the end of the backtest. Missing held prices/delistings require verified outcomes; do not silently substitute survivors. The current present-day catalogue omits some disappeared securities. Identity cleanup alone does not remove survivorship bias: results remain exploratory without historical universe and delisting coverage.
5. Freeze a new cleaned manifest and retain original inputs and all prior results. Do not overwrite older experiments. No additional provider plan purchase is authorised or assumed.

## Fixed primary rules

- Scope: ordinary exchange-listed USD stock listings with verified identity; exclude OTC, funds, preferreds, warrants, rights and ambiguous issuer duplicates from the primary study. Report any missing historical exchange or instrument-type classification.
- At each completed month end, shortlist up to500 eligible stocks using only prior observed liquidity, approximated by trailing63-session median close×volume where valid. Minimum proxy threshold$10m/day and98% observed-price coverage in the required lookback. No future-history selection. Require the exact month-end observations used by the signal; do not fill them from later dates.
- Signal for the next holding month: adjusted-price return from twelve month-ends earlier through the previous month-end, omitting the most recent month. Store the precise endpoint dates for audit. Example: at end-December, compare end-November with end-December of the previous year; do not use January data.
- Rank strongest first. Select25 eligible stocks; equal target weight4% each before risk scaling/caps. If fewer than25 eligible stocks are available, retain the unused allocation in cash rather than increasing remaining weights. No shorting or leverage.
- Observe the completed month-end signal and execute at the next observed daily close. Re-evaluate monthly. This delayed daily-close execution is a modelling approximation, not proof of historical fills. Existing fixed fold-end liquidation and gap rules must be applied identically to comparisons.
- Use prior126-session daily return covariance to estimate current portfolio volatility; scale exposure down toward12% annualised target, never above100% gross. Recalculate at each monthly rebalance using only past data. Target risk is not guaranteed realised risk.
- Costs: commission1bp, full spread10bp, slippage5bp plus declared volume-impact approximation. Cash0% in the primary comparison to avoid inventing a historical interest series. Record turnover and fees. Use no short borrow or margin financing because there is no shorting/leverage. Separately report that cash0% is a simplification.

## Three comparisons fixed before seeing their results

A. Benchmark: equal-weight the same eligible liquid universe, applying the same lagged risk target, execution delay, costs and evaluation dates. Also report unscaled SPY and a lagged-volatility-controlled SPY reference, identifying their different risks.
B. Momentum: the25 strongest stocks under the primary rules, with the same lagged risk target.
C. Momentum with relationship limits: same ranking, but permit at most20% of total target portfolio value in any correlation group. Build groups on the preceding252 observed daily return sessions using raw Pearson return correlations, deterministic average-linkage distance1−correlation, cut0.2 (equivalent to0.8 at pairwise linkage threshold). Choose down the ranking until25 holdings or the eligible universe is exhausted. Group caps apply to actual proposed target dollars, not just a count of names. Leave unallocated risk budget in cash; do not choose the clustering threshold from test performance. Publish group assignments and rejected candidates.

B versus A tests the ranking signal. C versus B tests whether the relationship constraint adds value. Changing signal, weights, universe, volatility and costs all at once would confound this interpretation. The benchmark must use the same point-in-time eligible universe; its500-stock breadth is an intentional comparison difference, not an independently certified market index.

## Evaluation and failure criteria

Use the existing dated walk-forward framework to build/validate the implementation and analyse stability; freeze these primary rules rather than choosing winners from a large validation grid. All2016–2026 observations already inspected are development evidence. Do not advertise2025/2026 as a fresh holdout. Any later truly unobserved completed period, after the strategy and data gates are frozen, can supply forward paper evidence.

Report annualised/total return, volatility, drawdown, Sharpe using declared cash assumption, exposure, turnover, costs, monthly/yearly outcomes, eligible counts, issuer/group concentration and invalid-data flags. Include historical beta/factor attribution using appropriately sourced factor data; this attribution is not present in the current app and must be added before calling an excess return alpha. Compare risks, not only endpoint wealth. Block-bootstrap differences using aligned return dates; label intervals conditional and not a solution to survivorship or research selection.

Predeclare limited sensitivity checks:6-month versus12-month ranking (both omit latest month); doubled trading costs; one extra session execution delay; and leave-one-year-out attribution of historical profits. Report all checks, not only the best. These are robustness checks, not independent confirmations. Do not optimise hundreds of lookbacks or select a revised rule by the later test results.

Reasons to reject or suspend the idea: momentum does not improve matched-risk net results over the benchmark; any apparent improvement disappears under modest costs/delays; profit is concentrated in one year or security; relationship caps merely add cash without a better matched-risk tradeoff; required identities or delisting outcomes remain unresolved. Even a promising historical result requires prospective paper testing.

## Platform changes required

A clean dated identity layer; dynamic month-end ranking and target-weight schedules; audited lagged risk estimates; full-universe comparable baseline; the separately specified group-cap variant; and return-difference/factor-attribution reporting. Current portfolio accounting, Parquet snapshots, guarded CPU workers, dated windows, exports and paper records can be reused. Existing strategy menu settings cannot substitute for these missing components. No new momentum run has been queued.

## Primary research sources

- Kenneth French, US monthly momentum-factor construction: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor.html
- Daniel and Moskowitz, Momentum Crashes: https://www.nber.org/papers/w20439
- Frazzini, Israel and Moskowitz, Trading Costs of Asset Pricing Anomalies: https://www.aqr.com/insights/research/working-paper/trading-costs-of-asset-pricing-anomalies

The research describes historical strategy families and institutional implementations. It does not establish the performance of this proposed long-only adaptation on this cache or predict future returns.
