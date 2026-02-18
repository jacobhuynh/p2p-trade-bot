"""
generate_mock_db.py

Generates mock Kalshi NBA data that is 1:1 with the real database schema.

Real schema (from SCHEMAS.md):
  Markets:  ticker, event_ticker, market_type, title, yes_sub_title, no_sub_title,
            status, yes_bid, yes_ask, no_bid, no_ask, last_price, volume, volume_24h,
            open_interest, result, created_time, open_time, close_time, _fetched_at
  Trades:   trade_id, ticker, count, yes_price, no_price, taker_side,
            created_time, _fetched_at

Real ticker formats (observed from live Kalshi data):
  Game winner:   KXNBAGAME-{YYMONDD}{HOME}{AWAY}-{SIDE}     e.g. KXNBAGAME-26FEB19BKNCLE-BKN
  Season wins:   KXNBAWINS-{TEAM}-{SEASON}-T{THRESHOLD}     e.g. KXNBAWINS-NOP-25-T30
  Player prop:   KXNBASGPROP-{YYMONDD}{PLAYER}-{STAT}{THRESHOLD}
                 e.g. KXNBASGPROP-26FEB19LEBRON-PTS25

Context lives in title/yes_sub_title, NOT encoded columns.
"""

import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

NBA_SEASONS = [2023, 2024, 2025]
GAMES_PER_SEASON = 82  # per team, but we'll generate matchups per week

# (last_name_slug, display_name, team)
PLAYERS = [
    ("LEBRON",    "LeBron James",      "LAL"),
    ("CURRY",     "Stephen Curry",     "GSW"),
    ("GIANNIS",   "Giannis Antetokounmpo", "MIL"),
    ("JOKIC",     "Nikola Jokic",      "DEN"),
    ("EMBIID",    "Joel Embiid",       "PHI"),
    ("TATUM",     "Jayson Tatum",      "BOS"),
    ("LUKA",      "Luka Doncic",       "DAL"),
    ("BOOKER",    "Devin Booker",      "PHX"),
    ("DURANT",    "Kevin Durant",      "PHX"),
    ("LILLARD",   "Damian Lillard",    "MIL"),
]
PLAYER_BY_TEAM = {team: (slug, name) for slug, name, team in PLAYERS}

TEAMS = [
    "LAL", "GSW", "MIL", "DEN", "PHI", "BOS", "DAL", "PHX",
    "MIA", "BKN", "NYK", "CHI", "CLE", "ATL", "TOR", "IND",
    "MEM", "NOP", "MIN", "OKC", "SAC", "POR", "UTA", "SAS",
    "ORL", "WAS", "DET", "CHA", "HOU", "LAC",
]

# (series_prefix, stat_slug, threshold, title_template, yes_sub_template, true_prob_range)
PROP_TYPES = [
    # Game winner
    ("KXNBAGAME", None, None,
     "{home} at {away} Winner?",
     "{home} wins",
     (0.35, 0.68)),

    # Season win totals
    ("KXNBAWINS", None, 45,
     "Will {home} win at least 45 games this season?",
     "{home} wins 45+",
     (0.30, 0.60)),

    ("KXNBAWINS", None, 30,
     "Will {home} win at least 30 games this season?",
     "{home} wins 30+",
     (0.45, 0.75)),

    # Player points props
    ("KXNBASGPROP", "PTS", 20,
     "{player} Over 19.5 points",
     "{player} scores 20+ points",
     (0.45, 0.65)),

    ("KXNBASGPROP", "PTS", 25,
     "{player} Over 24.5 points",
     "{player} scores 25+ points",
     (0.30, 0.55)),

    ("KXNBASGPROP", "PTS", 30,
     "{player} Over 29.5 points",
     "{player} scores 30+ points",
     (0.18, 0.38)),

    # Player assists props
    ("KXNBASGPROP", "AST", 6,
     "{player} Over 5.5 assists",
     "{player} records 6+ assists",
     (0.40, 0.60)),

    ("KXNBASGPROP", "AST", 10,
     "{player} Over 9.5 assists",
     "{player} records 10+ assists",
     (0.20, 0.40)),

    # Player rebounds props
    ("KXNBASGPROP", "REB", 8,
     "{player} Over 7.5 rebounds",
     "{player} grabs 8+ rebounds",
     (0.40, 0.60)),
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def season_start(season: int) -> datetime:
    return datetime(season, 10, 18)  # NBA tipoff ~mid October

def game_date(season: int, week: int) -> datetime:
    return season_start(season) + timedelta(weeks=week - 1)

def date_slug(dt: datetime) -> str:
    """Format datetime as YYMONDD, e.g. 26FEB19"""
    return dt.strftime("%y%b%d").upper()

def spread_from_prob(true_prob: float, noise: float = 0.03) -> tuple:
    market_prob = max(0.03, min(0.97, true_prob + random.gauss(0, noise)))
    yes_mid = round(market_prob * 100)
    spread = random.randint(1, 3)
    yes_bid = max(1,  yes_mid - spread)
    yes_ask = min(99, yes_mid + spread)
    no_bid  = 100 - yes_ask
    no_ask  = 100 - yes_bid
    last    = random.choice([yes_bid, yes_mid, yes_ask])
    return yes_bid, yes_ask, no_bid, no_ask, last, round(market_prob, 4)

def rand_volume(series: str) -> int:
    ranges = {
        "KXNBAGAME":   (50000, 5000000),
        "KXNBAWINS":   (10000, 500000),
        "KXNBASGPROP": (1000,  100000),
    }
    lo, hi = ranges.get(series, (1000, 50000))
    return random.randint(lo, hi)


# ─────────────────────────────────────────────
# GENERATE MARKETS
# ─────────────────────────────────────────────

def generate_markets() -> pd.DataFrame:
    print("  Generating NBA markets...")
    markets = []

    for season in NBA_SEASONS:
        for week in range(1, 28):  # ~27 weeks in an NBA season
            gdate    = game_date(season, week)
            dslug    = date_slug(gdate)
            open_t   = gdate - timedelta(days=1)
            close_t  = gdate + timedelta(hours=3)
            finalized = gdate < datetime(2025, 1, 1)
            status    = "finalized" if finalized else "open"

            teams = TEAMS[:]
            random.shuffle(teams)
            matchups = [(teams[i], teams[i+1]) for i in range(0, 14, 2)]

            for home, away in matchups:
                for (series, stat, threshold, title_tmpl, yes_sub_tmpl, prob_range) in PROP_TYPES:
                    is_prop = series == "KXNBASGPROP"
                    is_wins = series == "KXNBAWINS"

                    if is_prop:
                        player_data = PLAYER_BY_TEAM.get(home) or PLAYER_BY_TEAM.get(away)
                        if not player_data:
                            continue
                        player_slug, player_name = player_data
                        ticker  = f"{series}-{dslug}{player_slug}-{stat}{threshold}"
                        title   = title_tmpl.format(player=player_name)
                        yes_sub = yes_sub_tmpl.format(player=player_name)
                        no_sub  = f"No: {yes_sub}"
                        event_t = f"{series}-{dslug}{player_slug}"

                    elif is_wins:
                        season_short = str(season)[2:]
                        ticker  = f"{series}-{home}-{season_short}-T{threshold}"
                        title   = title_tmpl.format(home=home)
                        yes_sub = yes_sub_tmpl.format(home=home)
                        no_sub  = f"No: {yes_sub}"
                        event_t = f"{series}-{home}-{season_short}"

                    else:
                        # Game winner — generate both sides (home and away)
                        for side, side_team, side_prob_range in [
                            ("home", home, prob_range),
                            ("away", away, (1 - prob_range[1], 1 - prob_range[0])),
                        ]:
                            ticker  = f"{series}-{dslug}{home}{away}-{side_team}"
                            title   = title_tmpl.format(home=home, away=away)
                            yes_sub = f"{side_team} wins"
                            no_sub  = f"{side_team} loses"
                            event_t = f"{series}-{dslug}{home}{away}"

                            tp = random.uniform(*side_prob_range)
                            yb, ya, nb, na, last, market_prob = spread_from_prob(tp)
                            vol    = rand_volume(series)
                            result = ("yes" if random.random() < tp else "no") if finalized else ""

                            markets.append({
                                "ticker":        ticker,
                                "event_ticker":  event_t,
                                "market_type":   "binary",
                                "title":         title,
                                "yes_sub_title": yes_sub,
                                "no_sub_title":  no_sub,
                                "status":        status,
                                "yes_bid":       yb   if not finalized else None,
                                "yes_ask":       ya   if not finalized else None,
                                "no_bid":        nb   if not finalized else None,
                                "no_ask":        na   if not finalized else None,
                                "last_price":    last,
                                "volume":        vol,
                                "volume_24h":    int(vol * random.uniform(0.01, 0.1)) if not finalized else 0,
                                "open_interest": int(vol * random.uniform(0.05, 0.25)),
                                "result":        result,
                                "created_time":  open_t - timedelta(days=random.randint(1, 3)),
                                "open_time":     open_t,
                                "close_time":    close_t,
                                "_fetched_at":   datetime.now(),
                            })
                        continue  # skip the shared append below for game winner

                    tp = random.uniform(*prob_range)
                    yb, ya, nb, na, last, market_prob = spread_from_prob(tp)
                    vol    = rand_volume(series)
                    result = ("yes" if random.random() < tp else "no") if finalized else ""

                    markets.append({
                        "ticker":        ticker,
                        "event_ticker":  event_t,
                        "market_type":   "binary",
                        "title":         title,
                        "yes_sub_title": yes_sub,
                        "no_sub_title":  no_sub,
                        "status":        status,
                        "yes_bid":       yb   if not finalized else None,
                        "yes_ask":       ya   if not finalized else None,
                        "no_bid":        nb   if not finalized else None,
                        "no_ask":        na   if not finalized else None,
                        "last_price":    last,
                        "volume":        vol,
                        "volume_24h":    int(vol * random.uniform(0.01, 0.1)) if not finalized else 0,
                        "open_interest": int(vol * random.uniform(0.05, 0.25)),
                        "result":        result,
                        "created_time":  open_t - timedelta(days=random.randint(1, 3)),
                        "open_time":     open_t,
                        "close_time":    close_t,
                        "_fetched_at":   datetime.now(),
                    })

    df = pd.DataFrame(markets).drop_duplicates(subset=["ticker"])
    print(f"  ✓ {len(df):,} markets")
    return df


# ─────────────────────────────────────────────
# GENERATE TRADES
# ─────────────────────────────────────────────

def generate_trades(markets_df: pd.DataFrame, target: int = 400_000) -> pd.DataFrame:
    print(f"  Generating ~{target:,} trades...")
    finalized = markets_df[markets_df["status"] == "finalized"].copy()

    total_vol = finalized["volume"].sum()
    trades = []

    for _, mkt in finalized.iterrows():
        n = max(1, int(mkt["volume"] / total_vol * target))
        result  = mkt["result"]
        open_t  = pd.Timestamp(mkt["open_time"])
        close_t = pd.Timestamp(mkt["close_time"])
        last    = int(mkt["last_price"])

        true_prob = (last / 100.0 + (0.95 if result == "yes" else 0.05)) / 2

        for _ in range(n):
            t = random.random()
            drift_target = 0.95 if result == "yes" else 0.05
            price_center = true_prob * (1 - t**2) + drift_target * t**2
            price_center += random.gauss(0, 0.04 * (1 - t))
            yp = max(1, min(99, round(price_center * 100)))

            trades.append({
                "trade_id":     str(uuid.uuid4()),
                "ticker":       mkt["ticker"],
                "count":        random.choices(
                                    [1, 2, 5, 10, 25, 50, 100, 250],
                                    weights=[30, 20, 15, 12, 8, 6, 5, 4]
                                )[0],
                "yes_price":    yp,
                "no_price":     100 - yp,
                "taker_side":   random.choices(["yes", "no"], weights=[yp, 100-yp])[0],
                "created_time": open_t + (close_t - open_t) * t,
                "_fetched_at":  datetime.now(),
            })

    df = pd.DataFrame(trades)
    print(f"  ✓ {len(df):,} trades")
    return df


# ─────────────────────────────────────────────
# SAVE TO PARQUET
# ─────────────────────────────────────────────

def save(markets_df, trades_df):
    base = Path("data/kalshi")
    (base / "markets").mkdir(parents=True, exist_ok=True)
    (base / "trades").mkdir(parents=True, exist_ok=True)

    print("  Saving Parquet files...")
    for season in NBA_SEASONS:
        mask = markets_df["open_time"].apply(
            lambda x: pd.Timestamp(x).year == season
        )
        p = base / "markets" / f"nba_markets_{season}.parquet"
        markets_df[mask].to_parquet(p, index=False)
        print(f"    ✓ {p}  ({mask.sum():,} rows)")

    for season in NBA_SEASONS:
        season_tickers = markets_df[
            markets_df["open_time"].apply(lambda x: pd.Timestamp(x).year == season)
        ]["ticker"].tolist()
        mask = trades_df["ticker"].isin(season_tickers)
        p = base / "trades" / f"nba_trades_{season}.parquet"
        trades_df[mask].to_parquet(p, index=False)
        print(f"    ✓ {p}  ({mask.sum():,} rows)")


# ─────────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────────

def verify():
    print("\n  Running verification queries...")
    con = duckdb.connect()

    result = con.execute("""
        SELECT ticker, event_ticker, market_type, title, yes_sub_title, no_sub_title,
               status, yes_bid, yes_ask, no_bid, no_ask, last_price, volume,
               volume_24h, open_interest, result, created_time, open_time,
               close_time, _fetched_at
        FROM 'data/kalshi/markets/*.parquet'
        LIMIT 2
    """).df()
    print(f"  ✓ Markets schema: {list(result.columns)}")

    result = con.execute("""
        SELECT trade_id, ticker, count, yes_price, no_price, taker_side,
               created_time, _fetched_at
        FROM 'data/kalshi/trades/*.parquet'
        LIMIT 2
    """).df()
    print(f"  ✓ Trades schema:  {list(result.columns)}")

    result = con.execute("""
        WITH resolved AS (
            SELECT ticker, result
            FROM 'data/kalshi/markets/*.parquet'
            WHERE status = 'finalized' AND result IN ('yes', 'no')
        )
        SELECT
            t.yes_price,
            t.count,
            t.taker_side,
            m.result,
            CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS taker_won
        FROM 'data/kalshi/trades/*.parquet' t
        INNER JOIN resolved m ON t.ticker = m.ticker
        LIMIT 3
    """).df()
    print(f"  ✓ Join query: {len(result)} rows")

    result = con.execute("""
        SELECT
            regexp_extract(event_ticker, '^([A-Z0-9]+)', 1) AS category,
            COUNT(*) AS market_count
        FROM 'data/kalshi/markets/*.parquet'
        GROUP BY category
        ORDER BY market_count DESC
    """).df()
    print(f"  ✓ Categories: {result.to_dict(orient='records')}")

    result = con.execute("""
        SELECT ticker, title, yes_sub_title, status, result
        FROM 'data/kalshi/markets/*.parquet'
        LIMIT 5
    """).df()
    print(f"  ✓ Sample markets:\n{result.to_string()}")

    print("\n  All checks passed ✓")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  NBA Mock DB Generator — 1:1 with real Kalshi schema")
    print("=" * 60)

    markets_df = generate_markets()
    trades_df  = generate_trades(markets_df, target=400_000)
    save(markets_df, trades_df)
    verify()

    print(f"""
{'='*60}
  SUMMARY
  Markets : {len(markets_df):>10,}
  Trades  : {len(trades_df):>10,}

  Schema columns (matches SCHEMAS.md exactly):
  Markets: ticker, event_ticker, market_type, title,
           yes_sub_title, no_sub_title, status,
           yes_bid, yes_ask, no_bid, no_ask, last_price,
           volume, volume_24h, open_interest, result,
           created_time, open_time, close_time, _fetched_at
  Trades:  trade_id, ticker, count, yes_price, no_price,
           taker_side, created_time, _fetched_at

  Ticker format (matches real Kalshi):
  Game:    KXNBAGAME-{{YYMONDD}}{{HOME}}{{AWAY}}-{{SIDE}}
  Wins:    KXNBAWINS-{{TEAM}}-{{SEASON}}-T{{THRESHOLD}}
  Props:   KXNBASGPROP-{{YYMONDD}}{{PLAYER}}-{{STAT}}{{THRESHOLD}}
{'='*60}""")