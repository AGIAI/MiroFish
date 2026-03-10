# Alpha Sources & Data Edges for MiroFish Quant

## Overview

Differentiated alpha sources that most retail and institutional investors don't have — or don't combine. These are designed to feed MiroFish's multi-agent LLM simulation engine with information edges that generate true, uncorrelated alpha.

---

## 1. Government & Regulatory Data — Before the Market Digests It

Most traders react to CPI/NFP *headlines*. Almost nobody parses the **underlying microdata** in real-time.

| Source | Edge | Why It's Underutilized |
|--------|------|----------------------|
| **FRED API + BLS microdata** | Parse sub-components of CPI (shelter, used cars, medical) before analysts write summaries | Analysts take 30-60 min to digest; your agents can do it in seconds |
| **Federal Register API** | New regulations filed daily — most affect specific sectors before anyone notices | Nobody reads the Federal Register except lawyers |
| **USPTO patent filings** | Companies file patents months before product announcements | Patent text is dense; LLMs can extract meaning instantly |
| **USDA crop reports (raw data)** | Commodity traders wait for Bloomberg summaries; you parse the PDF directly | Direct parsing = minutes faster than news wire |
| **Congressional trading disclosures (STOCK Act)** | Senators/reps must disclose trades within 45 days — many are suspiciously well-timed | Several APIs exist (QuiverQuant, Capitol Trades) but few integrate programmatically |
| **FDA MAUDE database** | Medical device adverse event reports — early warning for healthcare stocks | Completely ignored by non-specialist traders |
| **DOE petroleum status reports** | Weekly crude/gasoline inventory data, raw tables | Most wait for EIA summary; raw data hits first |

### Implementation

Add parsers in `news_feed_service.py` that pull these on schedule, feed into the knowledge graph, and let agents reason about sector implications *before* the narrative hits Twitter.

### API Endpoints

```
FRED API:          https://api.stlouisfed.org/fred/
Federal Register:  https://www.federalregister.gov/developers/documentation/api/v1
USPTO:             https://patentsview.org/apis
USDA:              https://quickstats.nass.usda.gov/api
Capitol Trades:    https://www.capitoltrades.com/ (scrape or API)
FDA MAUDE:         https://open.fda.gov/apis/device/event/
DOE/EIA:           https://api.eia.gov/
```

---

## 2. On-Chain Intelligence — The Transparent Ledger Nobody Reads

Crypto is unique: **all transactions are public**. This is the most underutilized dataset in finance.

| Signal | Source | Edge |
|--------|--------|------|
| **Exchange inflow/outflow** | Glassnode, CryptoQuant, Nansen | Large deposits to exchanges = selling pressure incoming. Most retail doesn't track this |
| **Whale wallet clustering** | Arkham Intelligence, Nansen | Track the wallets of known funds (Jump, Alameda successors, etc.) — see their moves before they announce |
| **Smart money DeFi positions** | On-chain DEX/lending data | See when whales open leveraged longs/shorts on Aave/GMX before funding rates react |
| **Stablecoin flows** | USDT/USDC mint/burn events | Large USDT mints historically precede BTC rallies by 24-72h |
| **MEV bot activity** | Flashbots, MEV-Boost data | Unusual MEV activity signals informed trading (someone knows something) |
| **Token unlock schedules** | Token Unlocks, Messari | Predictable selling pressure that most retail ignores |
| **Governance proposals** | Snapshot, Tally | Protocol changes (fee switches, token burns) before they're priced in |
| **Bridge flows** | LayerZero, Wormhole data | Capital movement between chains signals narrative rotation |

### Implementation

New `onchain_data_service.py` with Nansen/Arkham/Dune adapters. Feed wallet movements as episodes into Zep. Your agents can then reason: *"Three known whale wallets just moved 50K ETH to Binance — this archetype historically sells within 48h."*

### API Endpoints

```
Glassnode:         https://api.glassnode.com/
CryptoQuant:       https://cryptoquant.com/docs
Nansen:            https://api.nansen.ai/
Arkham:            https://platform.arkhamintelligence.com/
Dune Analytics:    https://dune.com/docs/api/
Token Unlocks:     https://token.unlocks.app/api
Snapshot:          https://hub.snapshot.org/graphql
```

---

## 3. Satellite & Physical World Data

This is what Renaissance Technologies and Two Sigma use. Now accessible via APIs.

| Data | Source | Trading Application |
|------|--------|-------------------|
| **Satellite imagery of parking lots** | Orbital Insight, SpaceKnow | Count cars at Walmart/Target before earnings — proxy for revenue |
| **Satellite imagery of oil storage** | Kayrros, Ursa Space | Measure crude oil floating storage before EIA reports |
| **Ship tracking (AIS)** | MarineTraffic, VesselsValue | Track oil tankers, LNG carriers, container ships — supply chain disruptions visible weeks early |
| **Flight tracking** | ADS-B Exchange, FlightRadar24 | Corporate jet movements of CEOs before M&A announcements |
| **Credit card transaction data** | Second Measure, Earnest Research | Real-time consumer spending before earnings |
| **Web traffic / app downloads** | SimilarWeb, Sensor Tower | Company growth metrics before quarterly reports |
| **Job posting data** | Revelio Labs, Thinknum | Companies hiring aggressively = expansion; mass layoffs = trouble |
| **Electricity consumption** | Grid data APIs | Factory output proxy for industrial production |

### Implementation

These are expensive individually but many offer free tiers or academic access. Even one or two give you an edge. Add as data adapters that feed the knowledge graph.

### Free / Low-Cost Alternatives

```
Ship tracking:     https://www.marinetraffic.com/en/ais-api-services (free tier)
Flight tracking:   https://opensky-network.org/apidoc/ (free, academic)
Job postings:      https://thinknum.com/ or scrape Indeed/LinkedIn
Web traffic:        https://www.similarweb.com/corp/developer/ (free tier)
App downloads:     https://sensortower.com/ (limited free)
```

---

## 4. Social Sentiment — Beyond Basic NLP

Everyone does Twitter sentiment. Almost nobody does it *well*.

| Approach | Edge Over Standard Sentiment |
|----------|------------------------------|
| **Sentiment velocity** (rate of change, not level) | A stock going from -0.3 to +0.1 sentiment is more bullish than one sitting at +0.8 |
| **Influencer network mapping** | Track who influences whom — a signal from a market-moving account matters more than 10K retail tweets |
| **Contrarian sentiment extremes** | When *everyone* is bullish, the trade is short. Track sentiment distributions, not averages |
| **Cross-platform divergence** | Reddit bullish + Twitter bearish + 4chan accumulating = information asymmetry |
| **Insider language detection** | LLM can detect when someone "knows something" from linguistic patterns (hedging, specificity, urgency) |
| **Telegram/Discord alpha groups** | Many crypto moves originate in private groups before hitting public — monitor public spillover patterns |
| **Earnings call tone analysis** | CEO vocal patterns, hedging language, Q&A evasiveness — LLMs excel at this vs traditional NLP |
| **Chinese social media** (Weibo, WeChat sentiment) | Crypto markets are heavily influenced by Chinese capital — almost no Western quants monitor this |

### Why MiroFish Has a Structural Advantage Here

Your LLM agents can do nuanced sentiment analysis that simple NLP models cannot. They understand sarcasm, context, coded language, and narrative arcs. A traditional sentiment model sees "BTC is dead" as bearish. Your agents understand that when *everyone* says "BTC is dead," it's historically been the bottom.

### API Endpoints

```
Twitter/X:         https://developer.twitter.com/en/docs (API v2)
Reddit:            https://www.reddit.com/dev/api/
CryptoPanic:       https://cryptopanic.com/developers/api/
LunarCrush:        https://lunarcrush.com/developers/api
Stocktwits:        https://api.stocktwits.com/developers
Weibo:             https://open.weibo.com/wiki/API (requires Chinese entity)
```

---

## 5. Cross-Asset Correlation Breaks — The Hidden Alpha

Most traders watch one asset. Almost nobody systematically monitors **correlation regime changes**.

| Signal | What It Means |
|--------|--------------|
| BTC-SPX correlation drops suddenly | Crypto decoupling — trade the divergence |
| Gold-USD inverse breaks | Macro regime shift incoming |
| VIX-SPX correlation inverts | Options market pricing something the equity market isn't |
| BTC-ETH correlation drops | Altcoin rotation starting — sector-specific catalysts |
| Commodity-equity correlation spikes | Inflation regime — different playbook needed |
| Cross-exchange price divergence | Arbitrage opportunity or liquidity crisis developing |
| Bond-equity correlation flip | Risk parity funds forced to rebalance — mechanical flows |
| DXY-commodity inverse breaks | Dollar regime change — commodity supercycle signal |

### Implementation

Add a `correlation_monitor.py` service that tracks rolling correlations across 20+ asset pairs and fires alerts when correlations break historical norms (>2 standard deviations from rolling mean). Feed these breaks as high-priority episodes into the knowledge graph. Your agents can then reason about *why* the break is happening.

### Key Pairs to Monitor

```python
CORRELATION_PAIRS = [
    ("BTC", "SPX"),    ("BTC", "ETH"),     ("BTC", "GOLD"),
    ("BTC", "DXY"),    ("ETH", "SOL"),     ("SPX", "VIX"),
    ("GOLD", "DXY"),   ("OIL", "DXY"),     ("US10Y", "SPX"),
    ("US10Y", "GOLD"), ("DXY", "EUR"),     ("SPX", "RUSSELL"),
    ("COPPER", "SPX"), ("BTC", "MSTR"),    ("ETH", "BTC"),
    ("SOL", "BTC"),    ("NVDA", "SMH"),    ("OIL", "XLE"),
    ("US2Y", "US10Y"), ("HYG", "SPX"),     # yield curve, credit spread
]
```

---

## 6. Options Flow — The Smart Money Tells You What They Think

Options data is the single most underutilized edge available to non-institutional traders.

| Signal | Source | Edge |
|--------|--------|------|
| **Unusual options activity** | CBOE, Unusual Whales, FlowAlgo | Large, directional bets by institutions — they know something |
| **Put/call ratio extremes** | Options exchanges | Contrarian indicator — extreme fear = buy, extreme greed = sell |
| **Implied volatility surface** | Options chains | When IV is cheap relative to realized vol, big move coming |
| **Gamma exposure (GEX)** | Calculated from OI data | Market maker hedging flows create mechanical price levels |
| **Dark pool prints** | FINRA ATS data | Large block trades that institutions try to hide |
| **0DTE options flow** | Intraday options data | Same-day options reveal very short-term directional conviction |
| **Skew** | IV difference between OTM puts and calls | Measures tail risk pricing — high skew = institutions hedging hard |
| **Term structure** | IV across expirations | Backwardation = event risk priced in; contango = complacency |

### GEX (Gamma Exposure) — Deep Dive

GEX is particularly powerful: When dealer gamma is negative, price moves are amplified. When positive, moves are dampened. This creates predictable mechanical levels that most retail traders don't understand.

```
Positive GEX: Dealers are long gamma → they buy dips, sell rips → price is "pinned"
Negative GEX: Dealers are short gamma → they sell dips, buy rips → price moves are AMPLIFIED
GEX flip point: The price level where dealer hedging switches from stabilizing to destabilizing
```

**Trading rules:**
- Above GEX flip: mean-reversion strategies work (sell resistance, buy support)
- Below GEX flip: momentum strategies work (breakouts follow through)
- At GEX flip: maximum uncertainty — reduce position size

### Implementation

Add options flow parsing and GEX calculation. Feed significant unusual activity as events into the simulation. Your agents can react to "500K in SPY puts just swept at the ask" like a real institutional trader would.

### API Endpoints

```
Unusual Whales:    https://unusualwhales.com/api (paid)
CBOE:              https://www.cboe.com/market_data/
FlowAlgo:          https://flowalgo.com/ (paid)
FINRA ATS:         https://otctransparency.finra.org/otctransparency/
Tradier:           https://documentation.tradier.com/ (free options chains)
Polygon.io:        https://polygon.io/docs/options (free tier available)
```

---

## 7. Liquidation Cascade Prediction — Crypto's Unique Edge

In crypto, leveraged positions are transparent. You can literally see where the liquidations are stacked.

| Data | Source | How to Use It |
|------|--------|---------------|
| **Liquidation heatmaps** | Coinglass, Hyblock Capital | See where leveraged longs/shorts will get liquidated — price is magnetically attracted to these levels |
| **Funding rate extremes** | Exchange APIs | When funding is extremely positive, longs are overcrowded — squeeze risk reverses |
| **Open interest changes** | Coinglass, exchange APIs | Rising OI + rising price = new money entering. Rising OI + falling price = shorts opening aggressively |
| **Long/short ratio** | Exchange APIs | Retail positioning — fade the crowd |
| **Insurance fund changes** | Exchange APIs | Large drawdowns = cascading liquidations just happened — potential bottom |
| **Liquidation volume** | Exchange APIs | Spikes in liquidation volume often mark local tops/bottoms |
| **Basis (futures premium)** | Exchange APIs | High basis = excessive leverage. Negative basis = extreme fear |

### Why This Is Pure Alpha

Liquidation levels act as magnets — market makers hunt these levels because the forced selling/buying creates guaranteed profit. Your agents should know exactly where these levels are and factor them into predictions.

**Example signal flow:**
```
1. Detect: $500M in long liquidations clustered at $65,000
2. Detect: Funding rate at +0.05% (very high)
3. Detect: Open interest at all-time high
4. Agent reasoning: "Overcrowded long, liquidation magnet below, funding unsustainable"
5. Signal: SHORT with high conviction, target $65,000 liquidation cascade
6. Risk: Stop above recent high (invalidates thesis if new buyers absorb)
```

### API Endpoints

```
Coinglass:         https://coinglass.com/api (free tier)
Hyblock Capital:   https://hyblockcapital.com/ (paid)
Binance Futures:   https://binance-docs.github.io/apidocs/futures/en/
Bybit:             https://bybit-exchange.github.io/docs/
OKX:               https://www.okx.com/docs-v5/
```

---

## 8. MiroFish-Specific Novel Approaches

These leverage what **only** MiroFish can do — no other platform combines multi-agent LLM simulation with knowledge graph memory for trading.

| Approach | Description | Why Nobody Else Has This |
|----------|-------------|--------------------------|
| **Adversarial agent simulation** | Run bull-case and bear-case agent swarms simultaneously, measure which side "wins" | Traditional models don't pit competing narratives against each other |
| **Narrative half-life measurement** | Track how long each narrative drives price action before fading | LLM agents can detect narrative fatigue in real-time |
| **Information propagation modeling** | Simulate how a piece of news spreads through different trader archetypes | Predicts second-order effects (retail FOMO buying 24h after whale accumulation) |
| **Counterfactual simulation** | "What if CPI comes in hot?" — run the simulation both ways pre-event | Creates pre-positioned trades for both outcomes |
| **Agent disagreement as volatility proxy** | When agents strongly disagree, expect high volatility regardless of direction | Useful for options trading (buy straddles when agents disagree) |
| **Historical analogy matching** | Knowledge graph finds past situations with similar entity/relationship patterns | "This setup looks like March 2023 banking crisis with 78% similarity" |
| **Reflexivity modeling** | Agents that know other agents are watching the same signals — models Soros-style reflexivity | No other system simulates self-referential market dynamics |
| **Regime-adaptive archetype weighting** | Automatically increase influence of archetypes that are "hot" in current regime | Momentum agents get more weight in trending markets; contrarians in ranging |
| **Narrative collision detection** | Two competing narratives accelerating simultaneously — predict which wins | "Rate cut hope" vs "inflation persistence" — simulate the resolution |
| **Crowding decay curves** | Model how a crowded trade unwinds — slowly (orderly) or fast (cascade) | Predicts whether a reversal will be gradual or violent |

### Adversarial Simulation — Deep Dive

This is MiroFish's most unique capability. Here's how it works:

```
Round 1: Run 75 bull-biased agents + 75 bear-biased agents
Round 2: Each agent sees the simulated price action from Round 1
Round 3: Agents update their conviction based on new information
...
Round N: Measure final positioning

Key metrics:
- Which side gained converts? (agents switching sides)
- How strong is the winning side's conviction?
- Did any agents capitulate? (strong signal)
- What was the simulated price path? (reveals support/resistance)
```

### Counterfactual Simulation — Deep Dive

Before major events (FOMC, CPI, earnings), run parallel simulations:

```
Simulation A: "CPI comes in at 2.8% (below expectations)"
  → Agents react → Price path → Signal

Simulation B: "CPI comes in at 3.4% (above expectations)"
  → Agents react → Price path → Signal

Pre-position:
  - If both simulations agree on direction → high conviction trade
  - If simulations disagree → straddle (buy volatility)
  - Position size based on expected magnitude difference
```

---

## 9. The Meta-Edge: Combining Everything

The real alpha isn't in any single data source. It's in the **combination** that nobody else has:

```
On-chain whale movement (72h lead time)
  + Unusual options flow (24h lead time)
  + Liquidation heatmap (mechanical price levels)
  + Narrative velocity (qualitative context from LLM)
  + Regime detection (which strategy works right now)
  + Correlation break alerts (macro regime shifts)
  + Government data parsing (before analyst digestion)
  + 150 LLM agents reasoning about ALL of the above simultaneously
  + Knowledge graph remembering what happened last time this exact setup occurred
  = Signal that no single model, analyst, or fund can replicate
```

### Why This Beats Traditional Approaches

| Traditional Quant | MiroFish Quant |
|-------------------|----------------|
| Single model, single perspective | 150+ agents with diverse strategies |
| Static rules | Adaptive reasoning per market condition |
| No narrative understanding | LLM-powered narrative analysis |
| Backward-looking indicators only | Forward-looking agent simulations |
| No institutional memory | Knowledge graph retains all market history |
| Black box | Explainable (interview any agent for rationale) |
| Siloed data sources | All data sources feed into unified knowledge graph |
| Can't do counterfactuals | Runs "what if" simulations before events |
| Fixed strategy allocation | Regime-adaptive archetype weighting |

---

## 10. Implementation Priority — Highest ROI First

### Tier 1: Free Data, Proven Edge (Start Here)

| Source | Cost | Setup Time | Expected Edge |
|--------|------|------------|---------------|
| Liquidation heatmaps + funding rates (Coinglass) | Free | 1 day | High — mechanical price levels |
| On-chain exchange flows (free RPCs + Etherscan) | Free | 2 days | High — whale movements visible |
| FRED API (macro data + CPI components) | Free | 1 day | Medium — faster than analyst digestion |
| Reddit/Twitter sentiment velocity | Free | 2 days | Medium — contrarian extremes |
| Correlation monitoring (calculated from price data) | Free | 1 day | Medium — regime shift detection |

### Tier 2: Low Cost, Strong Edge

| Source | Cost | Setup Time | Expected Edge |
|--------|------|------------|---------------|
| Options flow (Tradier free tier + calculations) | Free-$50/mo | 3 days | Very High — GEX levels |
| Unusual Whales API | $30/mo | 1 day | High — institutional flow |
| Congressional trades (QuiverQuant) | Free | 1 day | Medium — political insider edge |
| Ship tracking (OpenSky free tier) | Free | 2 days | Medium — supply chain signals |
| Token unlock schedules | Free | 1 day | Medium — predictable sell pressure |

### Tier 3: Premium Data, Maximum Edge

| Source | Cost | Setup Time | Expected Edge |
|--------|------|------------|---------------|
| Nansen (on-chain intelligence) | $150/mo | 2 days | Very High — smart money tracking |
| Arkham Intelligence | $50/mo | 2 days | Very High — wallet clustering |
| Satellite imagery (SpaceKnow) | $500+/mo | 1 week | High — physical world data |
| Credit card data (Second Measure) | Enterprise | 1 week | Very High — real-time revenue proxy |
| Bloomberg Terminal API | $2K/mo | 3 days | High — fastest news + data |

---

## 11. Data Flow Architecture

```
                    ┌─────────────────────────────────────┐
                    │         DATA INGESTION LAYER         │
                    │                                       │
                    │  market_data_service.py               │
                    │  news_feed_service.py                 │
                    │  onchain_data_service.py              │
                    │  options_flow_service.py              │
                    │  correlation_monitor.py               │
                    │  liquidation_tracker.py               │
                    │  government_data_service.py           │
                    └──────────────┬────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │        KNOWLEDGE GRAPH (ZEP)         │
                    │                                       │
                    │  Financial Ontology                   │
                    │  Entity Memory (whale wallets,        │
                    │    narratives, correlations)          │
                    │  Historical Episodes                  │
                    │  Signal Outcome Memory               │
                    └──────────────┬────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │      REGIME DETECTION LAYER          │
                    │                                       │
                    │  regime_detector.py                   │
                    │  → Adjusts agent weights              │
                    │  → Selects appropriate strategies     │
                    └──────────────┬────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │     MULTI-AGENT SIMULATION           │
                    │                                       │
                    │  150+ LLM-powered trader agents       │
                    │  8 archetypes with quant parameters   │
                    │  Simulated order book                 │
                    │  Adversarial bull/bear swarms         │
                    │  Counterfactual branches              │
                    └──────────────┬────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │       SIGNAL EXTRACTION              │
                    │                                       │
                    │  signal_extractor.py                  │
                    │  signal_calibrator.py (Bayesian)      │
                    │  Ensemble across N simulation runs    │
                    └──────────────┬────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │       RISK MANAGEMENT                │
                    │                                       │
                    │  risk_manager.py                      │
                    │  Position sizing (Kelly)              │
                    │  Correlation-adjusted exposure        │
                    │  Circuit breakers                     │
                    └──────────────┬────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │       OUTPUT LAYER                    │
                    │                                       │
                    │  REST API  → Freqtrade, CCXT bots    │
                    │  WebSocket → Real-time streaming      │
                    │  Webhook   → 3Commas, Cornix          │
                    │  Dashboard → Human monitoring         │
                    └─────────────────────────────────────┘
```

---

## 12. New Files to Create

```
backend/app/services/
├── market_data_service.py        # Price, volume, OHLCV feeds
├── news_feed_service.py          # News, sentiment, government data
├── onchain_data_service.py       # Blockchain data adapters
├── options_flow_service.py       # Options flow + GEX calculation
├── correlation_monitor.py        # Cross-asset correlation tracking
├── liquidation_tracker.py        # Liquidation heatmaps + funding
├── government_data_service.py    # FRED, Federal Register, SEC, USDA
├── order_book_simulator.py       # Simulated market microstructure
├── signal_extractor.py           # Simulation → trading signal
├── signal_calibrator.py          # Bayesian calibration + ensemble
├── backtester.py                 # Walk-forward backtesting
├── regime_detector.py            # Market regime classification
└── risk_manager.py               # Portfolio risk management

backend/app/api/
├── signals.py                    # Signal REST API endpoints
└── market.py                     # Market state API endpoints

backend/scripts/
├── run_market_simulation.py      # Market simulation preset
└── run_backtest.py               # Backtesting preset

frontend/src/views/
├── SignalDashboard.vue           # Real-time signal display
├── BacktestView.vue              # Backtesting interface
├── PortfolioView.vue             # Position tracking
└── MarketStateView.vue           # Regime + correlation display
```
