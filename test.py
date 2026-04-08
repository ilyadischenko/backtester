"""
BTC Futures Ratio Chart
========================
Зависимости:
    pip install requests pandas plotly

Запуск:
    python btc_futures_ratio.py
"""

import time
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

# ── Настройки ──────────────────────────────────────────────────────────────
INTERVAL    = "1m"    # 1m, 3m, 5m, 15m, 1h, 4h, 1d ...
TOTAL_LIMIT = 30_000  # сколько свечей хотим получить (без ограничений сверху)

BASE_URL    = "https://fapi.binance.com"
KLINES_EP   = "/fapi/v1/klines"
EXCHANGE_EP = "/fapi/v1/exchangeInfo"

BATCH_SIZE  = 1500    # максимум за один запрос к Binance
PAUSE_SEC   = 0.15    # пауза между запросами, чтобы не получить 429
# ───────────────────────────────────────────────────────────────────────────

# Длительность одной свечи в миллисекундах
_INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
    "1w": 604_800_000,
}


def get_delivery_symbol() -> str:
    resp = requests.get(BASE_URL + EXCHANGE_EP, timeout=10)
    resp.raise_for_status()
    data = resp.json()["symbols"]
    for s in data:
        if (s["baseAsset"] == "ETH" and s["quoteAsset"] == "USDT"
                and s["contractType"] == "CURRENT_QUARTER"
                and s["status"] == "TRADING"):
            return s["symbol"]
    for s in data:
        if ("ETH" in s["symbol"] and s["quoteAsset"] == "USDT"
                and s["contractType"] in ("CURRENT_QUARTER", "NEXT_QUARTER")
                and s["status"] == "TRADING"):
            return s["symbol"]
    raise RuntimeError("Срочный ETH фьючерс не найден на бирже.")


def _parse_batch(raw: list) -> pd.DataFrame:
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qvol","trades","tbbase","tbquote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    df.set_index("open_time", inplace=True)
    return df[["close"]]


def fetch_klines(symbol: str, total: int = TOTAL_LIMIT) -> pd.DataFrame:
    """
    Загружает `total` свечей для символа, делая несколько запросов по
    BATCH_SIZE штук. Идёт назад во времени от текущего момента.
    """
    interval_ms = _INTERVAL_MS.get(INTERVAL)
    if interval_ms is None:
        raise ValueError(f"Неизвестный интервал: {INTERVAL}")

    chunks = []
    # end_time = None означает «до текущего момента»
    end_time_ms = None
    remaining   = total

    while remaining > 0:
        batch = min(remaining, BATCH_SIZE)
        params = {"symbol": symbol, "interval": INTERVAL, "limit": batch}
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        resp = requests.get(BASE_URL + KLINES_EP, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()

        if not raw:
            break

        df = _parse_batch(raw)
        chunks.append(df)
        remaining -= len(raw)

        # Следующий запрос — заканчиваем на свече ДО первой полученной
        first_open_ms = int(raw[0][0])
        end_time_ms   = first_open_ms - 1

        fetched_so_far = total - remaining
        print(f"    получено {fetched_so_far:>6} / {total}  "
              f"[от {df.index[0].strftime('%d.%m %H:%M')} UTC]", end="\r")

        if len(raw) < batch:
            # биржа вернула меньше — данных больше нет
            break

        if remaining > 0:
            time.sleep(PAUSE_SEC)

    print()  # перевод строки после \r

    if not chunks:
        raise ValueError(f"Нет данных для символа {symbol}")

    # chunks идут от новых к старым — переворачиваем
    result = pd.concat(chunks[::-1]).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    return result


# ═══════════════════════════════════════════════════════════════════════════
# БЭКТЕСТ  —  Basis Arbitrage (market-neutral pair trade)
# ═══════════════════════════════════════════════════════════════════════════

# ── Параметры бэктеста ─────────────────────────────────────────────────────
BT_CAPITAL  = 10_000.0  # стартовый капитал, USDT
BT_WINDOW   = 300       # окно rolling mean/std basis для Z-score
BT_ENTRY_Z  = 2.5       # открываем позицию при |Z| >= entry_z
BT_EXIT_Z   = 0.5       # закрываем при |Z| <= exit_z
BT_FEE      = 0.0002    # комиссия одной стороны (0.04 % taker Binance)
BT_SLIPPAGE = 0.0000    # проскальзывание одной стороны
BT_SIZE_PCT = 0.95      # доля капитала, выделяемая на каждую ногу
# ───────────────────────────────────────────────────────────────────────────


def run_backtest(m: pd.DataFrame):
    """
    Basis Arbitrage — классический market-neutral парный трейд.

    Basis = close_perp − close_delivery

    Сигнал (Z-score basis относительно скользящего окна):
      Z > +entry_z  →  basis широкий  →  SHORT perp + LONG  delivery
      Z < -entry_z  →  basis узкий    →  LONG  perp + SHORT delivery
      |Z| < exit_z  →  закрываем обе ноги

    PnL каждой ноги считается честно:
      short perp : (entry_p − exit_p) / entry_p
      long deliv : (exit_d  − entry_d) / entry_d
      (и зеркально для второго направления)

    Комиссия: 4 стороны × (fee + slippage) на сделку.

    Возвращает (bt_df, trades_df, stats_dict).
    """
    df = m.copy()

    # Z-score basis на скользящем окне
    roll = df["basis"].rolling(BT_WINDOW)
    df["b_mean"] = roll.mean()
    df["b_std"]  = roll.std()
    df["zscore"] = ((df["basis"] - df["b_mean"])
                    / df["b_std"].replace(0, float("nan")))

    # суммарные расходы на одну ногу (вход + выход)
    leg_cost = 2 * (BT_FEE + BT_SLIPPAGE)

    equity  = []          # equity curve
    trades  = []
    capital = BT_CAPITAL

    # состояние открытой позиции
    pos       = 0          # +1 = long perp/short deliv, -1 = обратное
    entry_p   = 0.0        # цена perp при входе
    entry_d   = 0.0        # цена delivery при входе
    entry_idx = None

    for i in range(len(df)):
        row   = df.iloc[i]
        z     = row["zscore"]
        p_now = row["close_p"]
        d_now = row["close_d"]
        ts    = df.index[i]

        if i < BT_WINDOW or pd.isna(z):
            equity.append(capital)
            continue

        # ── Закрытие ────────────────────────────────────────────────────
        if pos != 0 and abs(z) <= BT_EXIT_Z:
            notional = capital * BT_SIZE_PCT

            if pos == -1:
                # SHORT perp + LONG delivery
                ret_perp  = (entry_p - p_now) / entry_p   # short perp gain
                ret_deliv = (d_now - entry_d) / entry_d   # long  deliv gain
            else:
                # LONG perp + SHORT delivery
                ret_perp  = (p_now - entry_p) / entry_p   # long  perp gain
                ret_deliv = (entry_d - d_now) / entry_d   # short deliv gain

            # PnL каждой ноги минус её стоимость
            pnl_perp  = notional * (ret_perp  - leg_cost)
            pnl_deliv = notional * (ret_deliv - leg_cost)
            pnl_total = pnl_perp + pnl_deliv
            capital  += pnl_total

            direction = "short_basis" if pos == -1 else "long_basis"
            basis_entry = entry_p - entry_d
            basis_exit  = p_now   - d_now

            trades.append({
                "entry_time"  : entry_idx,
                "exit_time"   : ts,
                "direction"   : direction,
                # basis
                "basis_entry" : round(basis_entry, 4),
                "basis_exit"  : round(basis_exit,  4),
                "basis_change": round(basis_exit - basis_entry, 4),
                # цены
                "entry_p": round(entry_p, 2), "exit_p": round(p_now, 2),
                "entry_d": round(entry_d, 2), "exit_d": round(d_now, 2),
                # pnl
                "pnl_perp" : round(pnl_perp,  4),
                "pnl_deliv": round(pnl_deliv, 4),
                "pnl_total": round(pnl_total, 4),
                "capital"  : round(capital,   4),
                "entry_z"  : round(df.iloc[i - 1]["zscore"], 4),
                "exit_z"   : round(z, 4),
            })
            pos = 0

        # ── Открытие ────────────────────────────────────────────────────
        if pos == 0:
            if z >= BT_ENTRY_Z:      # basis широкий → продаём его
                pos = -1
                entry_p, entry_d, entry_idx = p_now, d_now, ts
            elif z <= -BT_ENTRY_Z:   # basis узкий  → покупаем его
                pos = 1
                entry_p, entry_d, entry_idx = p_now, d_now, ts

        equity.append(capital)

    # Дополнить до полной длины (первые BT_WINDOW баров)
    pad    = len(df) - len(equity)
    equity = [BT_CAPITAL] * pad + equity
    df["equity"] = equity

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # ── Статистика ──────────────────────────────────────────────────────
    eq_s   = pd.Series(equity)
    total  = eq_s.iloc[-1] - BT_CAPITAL
    pct    = total / BT_CAPITAL * 100
    max_dd = (eq_s / eq_s.cummax() - 1).min() * 100

    if len(trades_df):
        wins   = (trades_df["pnl_total"] > 0).sum()
        losses = (trades_df["pnl_total"] <= 0).sum()
        wr     = wins / (wins + losses) * 100 if (wins + losses) else 0
        avg_w  = trades_df.loc[trades_df["pnl_total"] > 0,  "pnl_total"].mean()
        avg_l  = trades_df.loc[trades_df["pnl_total"] <= 0, "pnl_total"].mean()
        pf     = abs(avg_w / avg_l) if avg_l and avg_l != 0 else float("inf")
        avg_hold_bars = (
            (pd.to_datetime(trades_df["exit_time"])
             - pd.to_datetime(trades_df["entry_time"]))
            .dt.total_seconds().mean() / 60
        )
    else:
        wins = losses = 0
        wr = pf = avg_w = avg_l = avg_hold_bars = 0

    stats = dict(
        total_pnl   = round(total,   2),
        pct         = round(pct,     2),
        max_dd      = round(max_dd,  2),
        trades      = int(wins + losses),
        wins        = int(wins),
        losses      = int(losses),
        win_rate    = round(wr,      1),
        profit_factor = round(pf,   2),
        avg_win     = round(float(avg_w) if avg_w else 0, 2),
        avg_loss    = round(float(avg_l) if avg_l else 0, 2),
        avg_hold_min= round(avg_hold_bars, 1),
    )

    return df, trades_df, stats


# ═══════════════════════════════════════════════════════════════════════════
# ГРАФИКИ
# ═══════════════════════════════════════════════════════════════════════════

def build_figure(perp: pd.DataFrame, deliv: pd.DataFrame,
                 perp_sym: str, deliv_sym: str) -> go.Figure:

    m = perp.join(deliv, how="inner", lsuffix="_p", rsuffix="_d")
    m["ratio"]  = m["close_p"] / m["close_d"]
    m["basis"]  = m["close_p"] - m["close_d"]
    m["basis%"] = m["basis"] / m["close_d"] * 100

    mean = m["ratio"].mean()
    std  = m["ratio"].std()
    x    = m.index

    # ── цвета ──
    C_PERP  = "#27500A"   # тёмно-зелёный
    C_DELIV = "#633806"   # тёмно-янтарный
    C_RATIO = "#185FA5"   # синий
    C_BAND  = "rgba(24,95,165,0.09)"
    C_MEAN  = "rgba(24,95,165,0.35)"
    C_GRID  = "rgba(180,180,180,0.2)"

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.44, 0.30, 0.26],
        subplot_titles=[
            f"Цены: {perp_sym} (вечный)  &  {deliv_sym} (срочный)",
            f"Ratio = {perp_sym} / {deliv_sym}",
            "Basis (Perp − Delivery), USDT",
        ],
    )

    # ── 1: цены на одном графике ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=m["close_p"], name=perp_sym,
        line=dict(color=C_PERP, width=1.5),
        hovertemplate="<b>Perp</b>: %{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=m["close_d"], name=deliv_sym,
        line=dict(color=C_DELIV, width=1.5),
        hovertemplate="<b>Delivery</b>: %{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # ── 2: ratio ─────────────────────────────────────────────────────────
    # полоса ±2σ
    fig.add_trace(go.Scatter(
        x=x, y=[mean + 2*std]*len(x),
        line=dict(color=C_MEAN, width=0.8, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=[mean - 2*std]*len(x),
        line=dict(color=C_MEAN, width=0.8, dash="dot"),
        fill="tonexty", fillcolor=C_BAND,
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    # средняя
    fig.add_trace(go.Scatter(
        x=x, y=[mean]*len(x),
        line=dict(color=C_MEAN, width=1, dash="dash"),
        name=f"Среднее {mean:.6f}", showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    # сам ratio
    fig.add_trace(go.Scatter(
        x=x, y=m["ratio"], name="Ratio P/D",
        line=dict(color=C_RATIO, width=2),
        hovertemplate="Ratio: %{y:.6f}<extra></extra>",
    ), row=2, col=1)

    # ── 3: basis (гистограмма) ────────────────────────────────────────────
    bar_colors = [C_PERP if v >= 0 else C_DELIV for v in m["basis"]]
    fig.add_trace(go.Bar(
        x=x, y=m["basis"], name="Basis",
        marker_color=bar_colors, opacity=0.75,
        hovertemplate="Basis: %{y:+.2f} USDT<extra></extra>",
    ), row=3, col=1)

    # ── аннотация последнего значения ─────────────────────────────────────
    last = m.iloc[-1]
    fig.add_annotation(
        x=0.01, y=0.975, xref="paper", yref="paper",
        text=(f"Ratio: <b>{last['ratio']:.6f}</b>   "
              f"Basis: <b>{last['basis']:+.2f} $</b>  "
              f"({last['basis%']:+.4f}%)"),
        font=dict(size=12, family="monospace"),
        showarrow=False, align="left",
        bgcolor="rgba(240,240,240,0.85)",
        bordercolor="rgba(0,0,0,0.15)", borderpad=5,
    )

    # ── оформление ────────────────────────────────────────────────────────
    ax = dict(gridcolor=C_GRID, zeroline=False,
              tickfont=dict(size=10), showline=False)
    fig.update_layout(
        title=dict(
            text=(f"<b>BTC Futures Ratio</b>  ·  {INTERVAL}  ·  "
                  f"{len(m)} свечей  ·  "
                  f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"),
            font=dict(size=15), x=0.5, xanchor="center",
        ),
        hovermode="x unified",
        legend=dict(orientation="h", x=0, y=1.02,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=55, r=55, t=80, b=45),
        bargap=0.1,
    )
    for i in range(1, 4):
        fig.update_xaxes(ax, row=i, col=1)
        fig.update_yaxes(ax, row=i, col=1)
    for ann in fig.layout.annotations:
        ann.font.size = 11

    return fig


def build_backtest_figure(bt: pd.DataFrame, trades: pd.DataFrame,
                          stats: dict, perp_sym: str, deliv_sym: str) -> go.Figure:
    """
    График бэктеста:
      Ряд 1 — Basis и его скользящее среднее (что на самом деле торгуем)
      Ряд 2 — Z-score basis с сигналами входа/выхода
      Ряд 3 — Equity curve
    """
    x = bt.index
    C_GRID   = "rgba(180,180,180,0.2)"
    C_BASIS  = "#534AB7"   # фиолетовый — basis
    C_BMEAN  = "rgba(83,74,183,0.35)"
    C_BUY    = "#3B6D11"   # зелёный — long basis (buy perp/sell deliv)
    C_SELL   = "#A32D2D"   # красный  — short basis (sell perp/buy deliv)
    C_EQ     = "#185FA5"   # синий    — equity

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.30, 0.28, 0.42],
        subplot_titles=[
            f"Basis = Perp − Delivery  (скользящее среднее {BT_WINDOW} баров)",
            f"Z-score basis  ·  вход ±{BT_ENTRY_Z}σ  ·  выход ±{BT_EXIT_Z}σ",
            "Equity curve  (basis arbitrage)",
        ],
    )

    # ── Ряд 1: Basis + rolling mean ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=bt["b_mean"],
        name="Rolling mean", line=dict(color=C_BMEAN, width=1.2, dash="dash"),
        hovertemplate="Mean: %{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=bt["basis"],
        name="Basis (P−D)", line=dict(color=C_BASIS, width=1.2),
        hovertemplate="Basis: %{y:+.2f} $<extra></extra>",
    ), row=1, col=1)

    # зоны входа (тонированные полосы ±entry_z * std)
    upper = bt["b_mean"] + BT_ENTRY_Z * bt["b_std"]
    lower = bt["b_mean"] - BT_ENTRY_Z * bt["b_std"]
    fig.add_trace(go.Scatter(
        x=x, y=upper, line=dict(color="rgba(163,45,45,0.25)", width=0.6),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=lower, line=dict(color="rgba(59,109,17,0.25)", width=0.6),
        fill="tonexty", fillcolor="rgba(200,200,200,0.06)",
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # ── Ряд 2: Z-score ───────────────────────────────────────────────────
    fig.add_hline(y=0,            line=dict(color="rgba(0,0,0,0.12)", width=1),        row=2, col=1)
    fig.add_hline(y= BT_ENTRY_Z,  line=dict(color=C_SELL, width=0.8, dash="dot"),      row=2, col=1)
    fig.add_hline(y=-BT_ENTRY_Z,  line=dict(color=C_BUY,  width=0.8, dash="dot"),      row=2, col=1)
    fig.add_hline(y= BT_EXIT_Z,   line=dict(color="gray", width=0.5, dash="dash"),     row=2, col=1)
    fig.add_hline(y=-BT_EXIT_Z,   line=dict(color="gray", width=0.5, dash="dash"),     row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=bt["zscore"], name="Z-score",
        line=dict(color=C_BASIS, width=1.3),
        hovertemplate="Z: %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    # маркеры сигналов
    if len(trades):
        short_b = trades[trades["direction"] == "short_basis"]  # sell perp/buy deliv
        long_b  = trades[trades["direction"] == "long_basis"]   # buy  perp/sell deliv

        def z_vals(times):
            idx = bt.index.searchsorted(pd.to_datetime(times))
            idx = idx.clip(0, len(bt) - 1)
            return bt["zscore"].iloc[idx].values

        if len(short_b):
            fig.add_trace(go.Scatter(
                x=short_b["entry_time"], y=z_vals(short_b["entry_time"]),
                mode="markers", name="Вход: sell basis",
                marker=dict(color=C_SELL, size=7, symbol="triangle-down"),
            ), row=2, col=1)
        if len(long_b):
            fig.add_trace(go.Scatter(
                x=long_b["entry_time"], y=z_vals(long_b["entry_time"]),
                mode="markers", name="Вход: buy basis",
                marker=dict(color=C_BUY, size=7, symbol="triangle-up"),
            ), row=2, col=1)

    # ── Ряд 3: Equity ────────────────────────────────────────────────────
    fig.add_hline(y=BT_CAPITAL,
                  line=dict(color="rgba(0,0,0,0.18)", width=1, dash="dash"),
                  row=3, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=bt["equity"],
        name="Equity", line=dict(color=C_EQ, width=2),
        fill="tozeroy", fillcolor="rgba(24,95,165,0.07)",
        hovertemplate="$%{y:,.2f}<extra>Equity</extra>",
    ), row=3, col=1)

    # закрашиваем просадку
    eq_s   = bt["equity"]
    peak   = eq_s.cummax()
    dd_abs = eq_s - peak   # <= 0
    fig.add_trace(go.Scatter(
        x=x, y=peak + dd_abs,   # = equity
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=peak,
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(163,45,45,0.12)",
        name="Просадка", hoverinfo="skip",
    ), row=3, col=1)

    # ── Заголовок со статистикой ─────────────────────────────────────────
    s    = stats
    sign = "+" if s["total_pnl"] >= 0 else ""
    title_text = (
        f"<b>Backtest · Basis Arbitrage · {INTERVAL} · окно {BT_WINDOW}</b><br>"
        f"<sup>"
        f"PnL: <b>{sign}${s['total_pnl']:,.0f}</b> ({sign}{s['pct']}%)  ·  "
        f"MaxDD: <b>{s['max_dd']}%</b>  ·  "
        f"Сделок: {s['trades']}  ·  "
        f"WinRate: {s['win_rate']}%  ·  "
        f"PF: {s['profit_factor']}  ·  "
        f"Avg hold: {s['avg_hold_min']} мин"
        f"</sup>"
    )

    ax = dict(gridcolor=C_GRID, zeroline=False, tickfont=dict(size=10))
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=13), x=0.5, xanchor="center"),
        hovermode="x unified",
        legend=dict(orientation="h", x=0, y=1.01,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=55, r=55, t=100, b=45),
    )
    for i in range(1, 4):
        fig.update_xaxes(ax, row=i, col=1)
        fig.update_yaxes(ax, row=i, col=1)
    for ann in fig.layout.annotations:
        ann.font.size = 11

    return fig


def main():
    print("▸ Ищем срочный фьючерс BTC...")
    deliv_sym = get_delivery_symbol()
    perp_sym  = "BTCUSDT"
    requests_needed = -(-TOTAL_LIMIT // BATCH_SIZE)
    print(f"  вечный  : {perp_sym}")
    print(f"  срочный : {deliv_sym}")
    print(f"  интервал: {INTERVAL}, свечей: {TOTAL_LIMIT} "
          f"(~{requests_needed} запросов на символ)")

    print(f"\n▸ Загружаем {perp_sym}...")
    perp = fetch_klines(perp_sym, TOTAL_LIMIT)
    print(f"  итого {len(perp)} свечей  "
          f"[{perp.index[0].strftime('%d.%m %H:%M')} — "
          f"{perp.index[-1].strftime('%d.%m %H:%M')} UTC]")

    print(f"▸ Загружаем {deliv_sym}...")
    deliv = fetch_klines(deliv_sym, TOTAL_LIMIT)
    print(f"  итого {len(deliv)} свечей  "
          f"[{deliv.index[0].strftime('%d.%m %H:%M')} — "
          f"{deliv.index[-1].strftime('%d.%m %H:%M')} UTC]")

    # ── Объединяем ───────────────────────────────────────────────────────
    m = perp.join(deliv, how="inner", lsuffix="_p", rsuffix="_d")
    m["ratio"]  = m["close_p"] / m["close_d"]
    m["basis"]  = m["close_p"] - m["close_d"]
    m["basis%"] = m["basis"]   / m["close_d"] * 100
    print(f"\n  Совпадающих свечей: {len(m)}")

    # ── График цен + ratio ───────────────────────────────────────────────
    print("\n▸ Строим график цен...")
    fig_prices = build_figure(perp, deliv, perp_sym, deliv_sym)
    out_prices = "btc_futures_ratio_chart.html"
    fig_prices.write_html(out_prices, include_plotlyjs="cdn")
    print(f"  ✓ {out_prices}")

    # ── Бэктест ──────────────────────────────────────────────────────────
    print("\n▸ Запускаем бэктест (basis arbitrage)...")
    bt_df, trades_df, stats = run_backtest(m)

    s    = stats
    sign = "+" if s["total_pnl"] >= 0 else ""
    print(f"\n  PnL        : {sign}${s['total_pnl']:,.2f}  ({sign}{s['pct']}%)")
    print(f"  Max DD     : {s['max_dd']}%")
    print(f"  Сделок     : {s['trades']}  (побед: {s['wins']}, убытков: {s['losses']})")
    print(f"  Win rate   : {s['win_rate']}%")
    print(f"  Profit F.  : {s['profit_factor']}")
    print(f"  Avg win    : +${s['avg_win']:,.2f}")
    print(f"  Avg loss   : ${s['avg_loss']:,.2f}")
    print(f"  Avg hold   : {s['avg_hold_min']} мин")

    if len(trades_df):
        out_trades = "btc_backtest_trades.csv"
        trades_df.to_csv(out_trades, index=False)
        print(f"\n  ✓ Сделки → {out_trades}")

    print("\n▸ Строим график бэктеста...")
    fig_bt = build_backtest_figure(bt_df, trades_df, stats, perp_sym, deliv_sym)
    out_bt = "btc_futures_backtest.html"
    fig_bt.write_html(out_bt, include_plotlyjs="cdn")
    print(f"  ✓ {out_bt}")

    fig_prices.show()
    fig_bt.show()


if __name__ == "__main__":
    main()