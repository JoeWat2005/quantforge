# UK Trading 212: investment strategy research

29 October 2026. The user confirmed a UK account and interest in both holding and intraday trading. Capital, maximum tolerable loss, experience and exact account type remain unspecified. No real trades, personal position sizes, account changes or deposits have been made.

## Practical distinction

An unleveraged diversified equity UCITS ETF held over a long horizon is an investment baseline. A strategy that changes positions needs to demonstrate an advantage over that baseline after risk and costs. Buying via Invest means actual securities; CFDs are derivative exposure with broker-specific financing and margin rules. Intraday trading is a separate research problem requiring intraday quotes and realistic fills. The daily adjusted-price cache cannot establish an intraday, CFD or options edge.

The suggested monthly stock-momentum research protocol remains in STRATEGY_RESEARCH_PLAN.md; dynamic momentum is not implemented in the current app and no momentum performance is claimed. This note investigates a simpler published monthly ETF trend rule as an additional candidate. It does not supersede the previous protocol or claim a successful active strategy.

## Concrete trend candidate and observed result

At completed month-end, compare the adjusted ETF close with the mean of the latest ten completed month-end adjusted closes, including that month. Above average: target the paper sleeve fully in the ETF; at or below: target cash. Execute only at the next observed daily close. No shorting/leverage, cash0%,11bps per side assumed trading cost. Compare with buying and holding the same instrument over exactly the same evaluation dates. This is an adaptation of Faber's monthly trend framework; its next-close execution differs from the publication's same-close assumptions.

Exploratory local check: frozen cached SPY adjusted USD closes,2017-10-02–2026-10-27,2,280 observed daily sessions. Primary10-month rule:7.63% annualised,−27.92% maximum drawdown,22 buy/sell orders. Buy-and-hold:14.82% annualised,−33.72% drawdown,2 orders including final liquidation. The primary rule did not improve return or Sharpe in this historical sample. It reduced volatility and drawdown but missed substantial upside. This does not establish a useful active edge.

All declared sensitivity variants are exported in TREND_RESEARCH_CHECK.csv/.json, including8/12 months, doubled costs and an additional execution delay. The12-month sensitivity looks better on some risk measures, but selecting it after inspecting these outcomes would be further model selection, not independent validation. Do not promote it as a proven alternative.

Checks: buy-and-hold net return agrees with the independently computed price ratio and two transaction charges; recorded historical signals precede execution; trace length agrees with the common observed calendar; NAV remains nonnegative. Original frozen dataset hash and per-order logs are in theJSON. These checks verify accounting/chronology, not market-data reliability or future profitability.

## UK execution gaps

SPYUSD is a historical proxy, not a Trading212UK executable strategy result. A UK implementation needs the actual eligible UCITS ETF's GBP total-return history, listing identity, costs, fund inception date and GBP cash return assumptions. Trading212's public Invest catalogue lists VUAG, Vanguard S&P500 accumulating, GBP London listing, ISINIE00BFMXXD54; this is an example of a test instrument, not a live buy recommendation. The Vanguard factsheet confirms a2019 inception, so earlier SPY history must be labelled a proxy rather than invented VUAG history. GBP quotation does not remove underlying currency exposure.

Trading212's published UK Invest terms show free trading commission/custody and0.15% FX conversions; spreads, fund costs and applicable taxes/levies still matter. The11bps assumed research cost is not a measurement of Trading212 spreads and does not include GBP conversion. Currency/account handling must be modelled from actual order flows. Do not apply today's cash rate retrospectively.

CFD overnight rates vary daily by instrument and direction. Financing is charged on notional exposure according to the broker's formula, not simply the small cash margin. Both sides of a long/short hedge can incur overnight interest. Historical broker spreads, financing, margin close-outs, contract multipliers and gaps are absent from the current app. A stock backtest cannot be relabelled a CFD backtest.

Trading212 staff's May2026 statement says options remain planned without an ETA. This is not verification that listed options are available to this user today. An options hedge needs a broker offering the relevant contracts and reliable option-chain bid/ask histories, expiry/strike choices, premiums, exercise/assignment, multipliers and currency/basis-risk treatment. These inputs are absent. Do not manufacture them from stock closing prices.

## Hedging distinction

Reducing the equity allocation reduces exposure, without buying a derivative, but provides no guaranteed loss floor. A monthly trend exit is not crash insurance: sharp losses can happen before a signal or execution, and whipsaws can repeatedly sell low and buy back higher. A correctly matched protective put can limit the underlying position's downside for its term, but costs a premium and expires. A collar offsets some put cost by selling a call and limiting upside, with additional assignment/execution requirements. An index put on a stock basket is an imperfect hedge with basis and currency risk. None has been simulated here.

## Research decision

Do not add CFD leverage or option complexity to rescue the failed basket selector or the weak primary trend result. Keep diversified buy-and-hold as the investment benchmark, with personal allocation dependent on horizon and loss capacity. Continue active research as a separately recorded paper experiment: clean dated security identities first, implement the predeclared momentum ranking, add correct UK costs/currency, compare matched risks, and require fresh forward evidence. Already inspected2017–2026 results are development evidence.

## Sources

- Trading212 Invest terms: https://www.trading212.com/terms/invest
- Trading212 overnight interest and hedged-position financing: https://helpcentre.trading212.com/hc/en-us/articles/360007113997-What-is-overnight-interest
- Trading212 public instrument catalogue: https://www.trading212.com/trading-instruments/invest
- Vanguard official VUAG/VUAA factsheet: https://fund-docs.vanguard.com/SandP_500_UCITS_ETF_USD_Accumulating_9694_EU_INT_UK_EN.pdf
- Trading212 staff options-plans statement: https://community.trading212.com/t/options-a-feature-that-should-be-at-the-top-of-everyones-new-features-list/90447
- Faber's account of the10-month framework and limitations: https://mebfaber.com/2017/12/13/episode-86-quantitative-approach-tactical-asset-allocation/
- Options Industry Council protective-put explanation: https://prd-web.optionseducation.org/strategies/all-strategies/protective-put-married-put
