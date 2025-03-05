import pandas as pd
import numpy as np
from itertools import product
import warnings
import os
from multiprocessing import Pool, cpu_count

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# ------------------------------------------------
# 1) Data Loading
# ------------------------------------------------
def load_data(file_name):
    df = pd.read_csv(file_name, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_vix_data(vix_file):
    vix_df = pd.read_csv(vix_file, parse_dates=['DATE'], dayfirst=True)
    vix_df.rename(columns={'DATE': 'Date'}, inplace=True)
    vix_df['Date'] = pd.to_datetime(vix_df['Date'], dayfirst=True, errors='coerce')
    vix_df.dropna(subset=['Date'], inplace=True)
    vix_df.sort_values('Date', inplace=True)
    vix_df.reset_index(drop=True, inplace=True)
    return vix_df

# ------------------------------------------------
# 2) Gap & Trading Helpers
# ------------------------------------------------
def calculate_day_range(day):
    return day['High'] - day['Low']

def calculate_gap(prev_close, curr_open):
    return curr_open - prev_close

def is_gap_valid(gap, prev_range):
    """
    We check only if gap < 0 (i.e., the market opened lower).
    Then, for this negative gap, we check the absolute size relative to the previous day's range.
    """
    if gap >= 0:
        return False  # We only want gaps < 0 to go Long

    gap_abs = abs(gap)
    if gap_abs < 0.20 or prev_range == 0:
        return False
    
    gap_percentage = gap_abs / prev_range
    # The same ratio rules as before:
    return 0.15 <= gap_percentage <= 0.85

def determine_trade_direction():
    """
    For this updated strategy, we only enter a Long position on negative gaps.
    """
    return 'Long'

def check_gap_closure(trade_direction, curr_day, prev_close):
    """
    Since we're only going Long, the logic for 'Short' won't actually be used.
    Keeping the structure for clarity; it won't be called with 'Short'.
    """
    if trade_direction == 'Short':
        # This block is no longer relevant, but we keep it in case of future modifications
        if curr_day['Low'] <= prev_close:
            exit_price = prev_close
            gap_closed = True
        else:
            exit_price = curr_day['Close']
            gap_closed = False
    else:  # Long
        if curr_day['High'] >= prev_close:
            exit_price = prev_close
            gap_closed = True
        else:
            exit_price = curr_day['Close']
            gap_closed = False
    return exit_price, gap_closed

def calculate_profit_loss(trade_direction, entry_price, exit_price, position_size):
    number_of_shares = position_size / entry_price
    # In this strategy, trade_direction will always be 'Long'
    if trade_direction == 'Long':
        return number_of_shares * (exit_price - entry_price)
    else:  # We keep this for completeness, though it won't be used here.
        return number_of_shares * (entry_price - exit_price)

# ------------------------------------------------
# 3) Precompute All Potential Trades (Long-Only)
# ------------------------------------------------
def precompute_trades(df, vix_df, initial_investment):
    """
    Creates a DataFrame of all valid gap trades (now only for negative gaps => Long).
    """
    trades = []
    total_days = len(df)

    # Build a dictionary of current_date -> VIX open (from the previous day)
    vix_dates = set(vix_df['Date'])
    min_vix_date = vix_df['Date'].min()
    vix_open_dict = {}

    for current_date in df['Date']:
        prev_date = current_date - pd.Timedelta(days=1)
        while prev_date not in vix_dates and prev_date > min_vix_date:
            prev_date -= pd.Timedelta(days=1)
        if prev_date in vix_dates:
            vix_open = vix_df.loc[vix_df['Date'] == prev_date, 'OPEN'].values[0]
            vix_open_dict[current_date] = vix_open
        else:
            vix_open_dict[current_date] = np.nan

    for i_day in range(1, total_days):
        prev_day = df.iloc[i_day - 1]
        curr_day = df.iloc[i_day]
        current_date = curr_day['Date']
        vix_open = vix_open_dict.get(current_date, np.nan)

        if pd.isna(vix_open):
            continue

        prev_range = calculate_day_range(prev_day)
        if prev_range == 0:
            continue

        gap = calculate_gap(prev_day['Close'], curr_day['Open'])
        if not is_gap_valid(gap, prev_range):
            continue

        trade_direction = determine_trade_direction()  # Now always 'Long'
        entry_price = curr_day['Open']
        exit_price, gap_closed = check_gap_closure(trade_direction, curr_day, prev_day['Close'])
        profit_loss = calculate_profit_loss(trade_direction, entry_price, exit_price, initial_investment)

        trades.append({
            'Date': current_date,
            'Trade Direction': trade_direction,
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Gap Closed': gap_closed,
            'VIX_Open': vix_open,
            'Profit/Loss': profit_loss,
            'df_idx': i_day
        })

    return pd.DataFrame(trades)

# ------------------------------------------------
# 4) Drop Filter (Skip) Logic
# ------------------------------------------------
def precompute_drop_ratios(vix_df, maxY=10):
    """
    Creates drop_ratios[i, Y] = % drop in VIX over last Y days (based on vix_df index).
    """
    vix_df_sorted = vix_df.sort_values('Date').reset_index(drop=True)
    vix_open = vix_df_sorted['OPEN'].values
    n = len(vix_open)

    drop_ratios = np.full((n, maxY + 1), np.nan, dtype=np.float64)
    for i in range(n):
        for Y in range(1, maxY + 1):
            j = i - Y
            if j < 0:
                continue
            if vix_open[j] == 0:
                continue
            drop_ratios[i, Y] = (vix_open[j] - vix_open[i]) / vix_open[j] * 100.0

    dates = vix_df_sorted['Date'].values
    return drop_ratios, dates

def build_vix_date_index_map(vix_dates):
    """
    Returns dict: date -> index in VIX array
    """
    return { d: i for i, d in enumerate(vix_dates) }

def apply_drop_filter(trades_df, df, drop_ratios, vix_date_index, X, Y, Z):
    """
    If drop_ratios[vix_idx, Y] >= X on day i, skip trades on days i+1..i+Z.
    Cast Y, Z to int in case they come in as floats.
    """
    n = len(df)
    skip_days = np.zeros(n, dtype=bool)
    df_dates = df['Date'].values

    Y = int(Y)
    Z = int(Z)

    for i in range(n):
        date_i = df_dates[i]
        if date_i not in vix_date_index:
            continue
        vix_idx = vix_date_index[date_i]
        drop_val = np.nan

        if Y < drop_ratios.shape[1]:
            drop_val = drop_ratios[vix_idx, Y]

        if not np.isnan(drop_val) and drop_val >= X:
            start_skip = i + 1
            end_skip = min(i + Z, n - 1)
            skip_days[start_skip:end_skip+1] = True

    return trades_df[~skip_days[trades_df['df_idx'].values]]

# ------------------------------------------------
# 5) Drop Filter Optimization by Success Rate
# ------------------------------------------------
def optimize_drop_filter_by_success_rate(trades_df, df, drop_ratios, vix_date_index,
                                         X_values, Y_values, Z_values,
                                         min_occurrence_count=0):
    """
    Instead of maximizing total profit, we maximize success rate = (# profitable trades / # trades).
    We also keep track of total profit as a secondary metric.
    """
    results = []
    best_success_rate = -1.0

    for X in X_values:
        for Y in Y_values:
            for Z in Z_values:
                filtered_trades = apply_drop_filter(trades_df, df, drop_ratios, vix_date_index, X, Y, Z)
                occurrence_count = len(filtered_trades)

                if occurrence_count < min_occurrence_count:
                    success_rate = 0.0
                    success_count = 0
                    total_profit = 0.0
                else:
                    success_count = sum(filtered_trades['Profit/Loss'] > 0)
                    success_rate = success_count / occurrence_count if occurrence_count > 0 else 0.0
                    total_profit = filtered_trades['Profit/Loss'].sum()

                results.append((X, Y, Z, occurrence_count, success_count, success_rate, total_profit))

                # Track the best success rate
                if success_rate > best_success_rate:
                    best_success_rate = success_rate

    results_df = pd.DataFrame(results, columns=[
        'X','Y','Z','OccurrenceCount','SuccessCount','SuccessRate','TotalProfit'
    ])
    results_df.sort_values(by=['SuccessRate','OccurrenceCount','TotalProfit'],
                           ascending=[False,False,False], inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    if not results_df.empty:
        top_row = results_df.iloc[0]
        best_X = top_row['X']
        best_Y = top_row['Y']
        best_Z = top_row['Z']
        best_occurrence = top_row['OccurrenceCount']
        best_successes = top_row['SuccessCount']
        best_sr = top_row['SuccessRate']
        best_pnl = top_row['TotalProfit']
    else:
        best_X = best_Y = best_Z = None
        best_occurrence = 0
        best_successes = 0
        best_sr = 0.0
        best_pnl = 0.0

    return (best_X, best_Y, best_Z, best_sr, best_pnl, best_occurrence, best_successes, results_df)

# ------------------------------------------------
# 6) VIX Optimization
# ------------------------------------------------
def find_best_vix_conditions_optimized(trades_df, i_min=10.0, i_max=30.0,
                                       j_min=10.0, j_max=30.0, step=0.1,
                                       min_occurrence_count=0):
    i_values = np.round(np.arange(i_min, j_max + step, step), 1)
    j_values = np.round(np.arange(j_min, j_max + step, step), 1)

    i_grid, j_grid = np.meshgrid(i_values, j_values)
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    valid_pairs_mask = i_flat < j_flat
    i_valid = i_flat[valid_pairs_mask]
    j_valid = j_flat[valid_pairs_mask]

    vix_open_array = trades_df['VIX_Open'].values
    profit_loss_array = trades_df['Profit/Loss'].values

    results = []
    for iv, jv in zip(i_valid, j_valid):
        cond = (vix_open_array > iv) & (vix_open_array < jv)
        occurrence_count = cond.sum()
        if occurrence_count >= min_occurrence_count:
            subset_profits = profit_loss_array[cond]
            success_count = (subset_profits > 0).sum()
            total_profit = subset_profits.sum()
            success_rate = success_count / occurrence_count if occurrence_count > 0 else 0.0
        else:
            success_count = 0
            total_profit = 0.0
            success_rate = 0.0

        results.append((iv, jv, occurrence_count, success_count, success_rate, total_profit))

    df_vix = pd.DataFrame(results, columns=[
        'i','j','OccurrenceCount','SuccessCount','SuccessRate','TotalProfit'
    ])
    df_vix = df_vix[df_vix['OccurrenceCount'] >= min_occurrence_count]
    df_vix.sort_values(by=['SuccessRate','OccurrenceCount'], ascending=[False,False], inplace=True)
    df_vix.reset_index(drop=True, inplace=True)

    if not df_vix.empty:
        best_i = df_vix.loc[0, 'i']
        best_j = df_vix.loc[0, 'j']
        best_oc = df_vix.loc[0, 'OccurrenceCount']
        best_sc = df_vix.loc[0, 'SuccessCount']
        best_sr = df_vix.loc[0, 'SuccessRate']
        best_pn = df_vix.loc[0, 'TotalProfit']
    else:
        best_i = None
        best_j = None
        best_oc = 0
        best_sc = 0
        best_sr = 0.0
        best_pn = 0.0

    return best_i, best_j, best_oc, best_sc, best_sr, best_pn, df_vix

# ------------------------------------------------
# 7) ATR Calculation & Optimization
# ------------------------------------------------
def calculate_atr(df, n):
    df = df.copy()
    df['PrevClose'] = df['Close'].shift(1)
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['PrevClose'])
    df['Low-PrevClose'] = abs(df['Low'] - df['PrevClose'])
    df['TR'] = df[['High-Low','High-PrevClose','Low-PrevClose']].max(axis=1)
    df['ATR'] = df['TR'].rolling(n).mean()
    return df

def compute_atr_metrics_sequential(args):
    trades_df, df, n, T = args
    df_with_atr = calculate_atr(df, n)
    df_atr = df_with_atr[['Date','ATR']].dropna()
    trades_merged = trades_df.merge(df_atr, on='Date', how='left')
    trades_merged = trades_merged.dropna(subset=['ATR'])

    if not trades_merged.empty:
        filtered = trades_merged[trades_merged['ATR'] <= T]
        occ_count = len(filtered)
        if occ_count > 0:
            profit_array = filtered['Profit/Loss'].values
            total_profit = profit_array.sum()
            success_count = (profit_array > 0).sum()
            success_rate = success_count / occ_count
        else:
            occ_count = 0
            success_count = 0
            total_profit = 0.0
            success_rate = 0.0
    else:
        occ_count = 0
        success_count = 0
        total_profit = 0.0
        success_rate = 0.0

    return (n, T, occ_count, success_count, success_rate, total_profit)

def optimize_atr_parameters(trades_df, df, n_values, T_values):
    param_combinations = list(product(n_values, T_values))
    args_list = [(trades_df.copy(), df.copy(), n, T) for (n, T) in param_combinations]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(compute_atr_metrics_sequential, args_list)

    results_df = pd.DataFrame(results, columns=[
        'n','T','OccurrenceCount','SuccessCount','SuccessRate','TotalProfit'
    ])
    # Sort primarily by total profit
    results_df.sort_values(by='TotalProfit', ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    if not results_df.empty:
        best_n = results_df.loc[0, 'n']
        best_T = results_df.loc[0, 'T']
        best_occ = results_df.loc[0, 'OccurrenceCount']
        best_sc = results_df.loc[0, 'SuccessCount']
        best_sr = results_df.loc[0, 'SuccessRate']
        best_pnl= results_df.loc[0, 'TotalProfit']
    else:
        best_n = None
        best_T = None
        best_occ = 0
        best_sc = 0
        best_sr = 0.0
        best_pnl= 0.0

    return best_n, best_T, best_occ, best_sc, best_sr, best_pnl, results_df

# ------------------------------------------------
# 8) Main Combined Function
# ------------------------------------------------
def run_combined_drop_vix_atr(
    main_file, 
    vix_file, 
    initial_investment=16000,
    # DROP FILTER SEARCH SPACE
    X_values=np.round(np.arange(1.0, 20.1, 0.5), 1),
    Y_values=np.arange(1.0, 6.0, 1.0),
    Z_values=np.arange(1.0, 6.0, 1.0),
    min_occurrence_drop=0,
    # VIX optimization
    vix_i_min=10.0, 
    vix_i_max=30.0, 
    vix_j_min=10.0, 
    vix_j_max=30.0, 
    vix_step=0.5,
    min_occurrence_vix=30,
    # ATR optimization
    atr_n_min=2, 
    atr_n_max=10, 
    atr_T_min=3.0, 
    atr_T_max=15.0, 
    atr_T_step=1.0
):
    if not os.path.exists(main_file):
        print(f"Error: Main data file '{main_file}' not found.")
        return
    if not os.path.exists(vix_file):
        print(f"Error: VIX data file '{vix_file}' not found.")
        return

    # 1) Load Data
    print("Loading main stock data...")
    df = load_data(main_file)
    print("Loading VIX data...")
    vix_df = load_vix_data(vix_file)

    # 2) Precompute All Gap Trades (Long-Only)
    print("\nPrecomputing all potential gap trades (LONG ONLY)...")
    trades_df = precompute_trades(df, vix_df, initial_investment)
    if trades_df.empty:
        print("No valid gap trades for Long strategy. Exiting.")
        return
    print(f"Total gap trades found (Long-only): {len(trades_df)}")

    # 3) Drop Filter Optimization (by SUCCESS RATE)
    print("\n=== DROP FILTER OPTIMIZATION (by Success Rate) ===")
    maxY = int(max(Y_values))  # ensure the drop_ratios can handle up to Y
    drop_ratios, vix_dates = precompute_drop_ratios(vix_df, maxY=maxY)
    vix_date_index = build_vix_date_index_map(vix_dates)

    (best_X, best_Y, best_Z,
     best_sr, best_pnl, best_occ, best_succ,
     drop_results_df) = optimize_drop_filter_by_success_rate(
         trades_df, df, drop_ratios, vix_date_index,
         X_values, Y_values, Z_values,
         min_occurrence_count=min_occurrence_drop
     )

    print(f"Best Drop Filter: X={best_X}, Y={best_Y}, Z={best_Z}")
    print(f"Occurrences: {best_occ}, Successes: {best_succ}, Success Rate: {best_sr:.2f}, Total Profit: {best_pnl:,.2f}")
    print("\nTop 10 drop filter combos (sorted by SuccessRate, OccurrenceCount, Profit):")
    print(drop_results_df.head(10).to_string(index=False))

    trades_after_drop = apply_drop_filter(trades_df, df, drop_ratios, vix_date_index, best_X, best_Y, best_Z)
    print(f"\nTrades after applying best drop filter: {len(trades_after_drop)} (out of {len(trades_df)})")
    if trades_after_drop.empty:
        print("No trades remain after drop filter. Exiting.")
        return

    # 4) VIX Optimization
    print("\n=== VIX RANGE OPTIMIZATION ===")
    (best_i, best_j, best_occ_vix, best_succ_vix,
     best_sr_vix, best_pnl_vix, vix_results_df) = find_best_vix_conditions_optimized(
         trades_after_drop,
         i_min=vix_i_min, i_max=vix_i_max,
         j_min=vix_j_min, j_max=vix_j_max,
         step=vix_step,
         min_occurrence_count=min_occurrence_vix
    )

    if best_i is None or best_j is None:
        print("No valid VIX range found. Exiting.")
        return

    print(f"Best VIX range: i={best_i}, j={best_j}")
    print(f"Occurrences: {best_occ_vix}, Successes: {best_succ_vix}, Success Rate: {best_sr_vix:.2f}, Total Profit: {best_pnl_vix:,.2f}")
    print("\nTop 10 VIX conditions:")
    print(vix_results_df.head(10).to_string(index=False))

    trades_best_vix = trades_after_drop[
        (trades_after_drop['VIX_Open'] > best_i) &
        (trades_after_drop['VIX_Open'] < best_j)
    ]
    print(f"\nTrades remaining after drop filter + best VIX range: {len(trades_best_vix)}")
    if trades_best_vix.empty:
        print("No trades left after applying VIX condition. Exiting.")
        return

    # 5) ATR Optimization
    print("\n=== ATR OPTIMIZATION ===")
    n_values = range(atr_n_min, atr_n_max + 1)
    T_values = np.round(np.arange(atr_T_min, atr_T_max + atr_T_step, atr_T_step), 1)

    (best_n, best_T, best_occ_atr, best_succ_atr,
     best_sr_atr, best_pnl_atr, atr_results_df) = optimize_atr_parameters(
         trades_best_vix, df, n_values, T_values
    )

    if best_n is None or best_T is None:
        print("No valid ATR parameters found. Exiting.")
        return

    print(f"Best ATR params: n={best_n}, T={best_T}")
    print(f"Occurrences: {best_occ_atr}, Successes: {best_succ_atr}, Success Rate: {best_sr_atr:.2f}, Total Profit: {best_pnl_atr:,.2f}")
    print("\nTop 10 ATR results (by TotalProfit):")
    print(atr_results_df.head(10).to_string(index=False))

    # 6) Final Combined Trades
    df_atr = calculate_atr(df.copy(), int(best_n))
    df_atr = df_atr[['Date','ATR']].dropna()
    final_trades = trades_after_drop.merge(df_atr, on='Date', how='left')
    final_trades = final_trades[
        (final_trades['VIX_Open'] > best_i) &
        (final_trades['VIX_Open'] < best_j) &
        (final_trades['ATR'] <= best_T)
    ]

    if final_trades.empty:
        print("\nNo trades remain after applying all filters.")
    else:
        final_occ = len(final_trades)
        final_succ = sum(final_trades['Profit/Loss'] > 0)
        final_sr = final_succ / final_occ if final_occ > 0 else 0.0
        final_pnl = final_trades['Profit/Loss'].sum()

        print("\n=== FINAL COMBINED RESULTS (LONG-ONLY STRATEGY) ===")
        print(f"Drop Filter (X,Y,Z): ({best_X}, {best_Y}, {best_Z})")
        print(f"VIX Range (i,j): ({best_i}, {best_j})")
        print(f"ATR (n,T): ({best_n}, {best_T})")
        print(f"Occurrences: {final_occ}, Successes: {final_succ}, Success Rate: {final_sr:.2f}, Total Profit: {final_pnl:,.2f}")
        print("\nFinal Trades:")
        print(final_trades.to_string(index=False))

# ------------------------------------------------
# Example Usage
# ------------------------------------------------
if __name__ == "__main__":
    main_file = 'D6CK32X6.csv'
    vix_file = 'VIX_History.csv'
    run_combined_drop_vix_atr(
        main_file,
        vix_file,
        initial_investment=16000,
        X_values=np.round(np.arange(1.0, 20.1, 0.5), 1),
        Y_values=np.arange(0.0, 6.0, 1.0),
        Z_values=np.arange(0.0, 6.0, 1.0),
        min_occurrence_drop=20,
        vix_i_min=10.0,
        vix_i_max=30.0,
        vix_j_min=10.0,
        vix_j_max=30.0,
        vix_step=0.5,
        min_occurrence_vix=50,
        atr_n_min=2,
        atr_n_max=10,
        atr_T_min=3.0,
        atr_T_max=15.0,
        atr_T_step=1.0
    )
