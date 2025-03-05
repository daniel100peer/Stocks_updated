import pandas as pd
import numpy as np
from itertools import product
import warnings
from multiprocessing import Pool, cpu_count
import os

# Suppress SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# Optional: Adjust pandas display options
pd.set_option('display.max_rows', None)      # Display all rows
pd.set_option('display.max_columns', None)   # Display all columns
pd.set_option('display.width', None)         # Adjust display width to fit the console
pd.set_option('display.max_colwidth', None)  # Display full content in each column

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

def calculate_day_range(day):
    return day['High'] - day['Low']

def calculate_gap(prev_close, curr_open):
    return curr_open - prev_close

def is_gap_valid(gap, prev_range):
    gap_abs = abs(gap)
    if gap_abs < 0.20 or prev_range == 0:
        return False
    gap_percentage = gap_abs / prev_range
    return 0.15 <= gap_percentage <= 0.85

def determine_trade_direction(gap):
    return 'Short' if gap > 0 else 'Long'

def check_gap_closure(trade_direction, curr_day, prev_close):
    if trade_direction == 'Short':
        # Gap closed if price ≤ prev_close
        if curr_day['Low'] <= prev_close:
            exit_price = prev_close
            gap_closed = True
        else:
            exit_price = curr_day['Close']
            gap_closed = False
    else:  # Long
        # Gap closed if price ≥ prev_close
        if curr_day['High'] >= prev_close:
            exit_price = prev_close
            gap_closed = True
        else:
            exit_price = curr_day['Close']
            gap_closed = False
    return exit_price, gap_closed

def calculate_profit_loss(trade_direction, entry_price, exit_price, position_size):
    number_of_shares = position_size / entry_price
    if trade_direction == 'Long':
        profit_loss = number_of_shares * (exit_price - entry_price)
    else:
        profit_loss = number_of_shares * (entry_price - exit_price)
    return profit_loss

def precompute_trades(df, vix_df, initial_investment):
    trades = []
    total_days = len(df)

    vix_open_dict = {}
    vix_dates = set(vix_df['Date'])
    min_vix_date = vix_df['Date'].min()
    
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

        trade_direction = determine_trade_direction(gap)
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
            'Profit/Loss': profit_loss
        })

    trades_df = pd.DataFrame(trades)
    return trades_df

def find_best_vix_conditions_optimized(trades_df, i_min=10.0, i_max=30.0, j_min=10.0, j_max=30.0, step=0.1, min_occurrence_count=0):
    """
    Modified to filter by a minimum occurrence count before selecting the best conditions.
    """
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
        if occurrence_count > 0:
            subset_profits = profit_loss_array[cond]
            success_count = (subset_profits > 0).sum()
            total_profit = subset_profits.sum()
            success_rate = success_count / occurrence_count
        else:
            success_count = 0
            total_profit = 0.0
            success_rate = 0.0
        results.append((iv, jv, occurrence_count, success_count, success_rate, total_profit))

    results_df = pd.DataFrame(results, columns=['i', 'j', 'OccurrenceCount', 'SuccessCount', 'SuccessRate', 'TotalProfit'])

    # Filter by min_occurrence_count first
    results_df = results_df[results_df['OccurrenceCount'] >= min_occurrence_count]

    # Sort by SuccessRate desc, then by OccurrenceCount desc
    results_df_sorted = results_df.sort_values(by=['SuccessRate', 'OccurrenceCount'], ascending=[False, False]).reset_index(drop=True)

    if not results_df_sorted.empty:
        best_i = results_df_sorted.loc[0, 'i']
        best_j = results_df_sorted.loc[0, 'j']
        best_occurrence_count = results_df_sorted.loc[0, 'OccurrenceCount']
        best_success_count = results_df_sorted.loc[0, 'SuccessCount']
        best_success_rate = results_df_sorted.loc[0, 'SuccessRate']
        best_profit = results_df_sorted.loc[0, 'TotalProfit']
    else:
        best_i = None
        best_j = None
        best_occurrence_count = 0
        best_success_count = 0
        best_success_rate = 0.0
        best_profit = 0.0

    print(f"\n=== VIX Optimization Complete (Success-Based) with min_occurrence_count={min_occurrence_count} ===")
    if best_i is not None:
        print(f"Best VIX Conditions: i={best_i}, j={best_j}")
        print(f"Occurrences: {best_occurrence_count}, Successes: {best_success_count}, Success Rate: {best_success_rate:.2f}, Total Profit: ${best_profit:,.2f}\n")
    else:
        print("No valid conditions found given the minimum occurrence threshold.\n")

    print("Top 10 VIX Conditions (by SuccessRate, then OccurrenceCount):")
    if not results_df_sorted.empty:
        print(results_df_sorted.head(10).to_string(index=False))
    else:
        print("No conditions to display.")

    return best_i, best_j, results_df_sorted

def calculate_atr(df, n):
    n = int(n)
    df['Previous Close'] = df['Close'].shift(1)
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Previous Close'])
    df['Low-PrevClose'] = abs(df['Low'] - df['Previous Close'])
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=n).mean()
    return df

def compute_atr_metrics_sequential(args):
    trades_df, df, n, T, initial_investment = args
    df_with_atr = calculate_atr(df.copy(), n)
    trades_with_atr = trades_df.merge(df_with_atr[['Date', 'ATR']], on='Date', how='left')
    trades_with_atr = trades_with_atr.dropna(subset=['ATR'])
    trades_filtered = trades_with_atr[trades_with_atr['ATR'] <= T]

    if not trades_filtered.empty:
        occurrence_count = trades_filtered.shape[0]
        profit_losses = trades_filtered['Profit/Loss'].values
        success_count = (profit_losses > 0).sum()
        total_profit = profit_losses.sum()
        success_rate = success_count / occurrence_count if occurrence_count > 0 else 0.0
    else:
        occurrence_count = 0
        success_count = 0
        total_profit = 0.0
        success_rate = 0.0

    return (n, T, occurrence_count, success_count, success_rate, total_profit)

def optimize_atr_parameters(trades_df, df, initial_investment, n_values, T_values):
    """
    ATR to be based on the amount of money that can be earned (Total Profit).
    """
    param_combinations = list(product(n_values, T_values))
    args_list = [(trades_df.copy(), df.copy(), n, T, initial_investment) for n, T in param_combinations]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(compute_atr_metrics_sequential, args_list)

    results_df = pd.DataFrame(results, columns=['n', 'T', 'OccurrenceCount', 'SuccessCount', 'SuccessRate', 'TotalProfit'])

    # Sort by TotalProfit descending (since we prioritize profit for ATR)
    results_df_sorted = results_df.sort_values(by='TotalProfit', ascending=False).reset_index(drop=True)

    if not results_df_sorted.empty:
        best_n = int(results_df_sorted.loc[0, 'n'])
        best_T = results_df_sorted.loc[0, 'T']
        best_occurrence_count = results_df_sorted.loc[0, 'OccurrenceCount']
        best_success_count = results_df_sorted.loc[0, 'SuccessCount']
        best_success_rate = results_df_sorted.loc[0, 'SuccessRate']
        best_total_profit_atr = results_df_sorted.loc[0, 'TotalProfit']
    else:
        best_n = None
        best_T = None
        best_occurrence_count = 0
        best_success_count = 0
        best_success_rate = 0.0
        best_total_profit_atr = 0.0

    print("\n=== ATR Optimization Complete (Profit-Based) ===")
    if best_n is not None:
        print(f"Best ATR Parameters: n={best_n}, T={best_T}")
        print(f"Occurrences: {best_occurrence_count}, Successes: {best_success_count}, Success Rate: {best_success_rate:.2f}, Total Profit: ${best_total_profit_atr:,.2f}\n")
    else:
        print("No ATR parameter combinations yielded any valid trades.\n")

    print("Top 100 ATR Parameter Combinations (by TotalProfit):")
    if not results_df_sorted.empty:
        print(results_df_sorted.head(100).to_string(index=False))
    else:
        print("No ATR parameter combinations yielded any profit.")

    return best_n, best_T, results_df_sorted

def run_full_optimization(df, vix_df, initial_investment, 
                          vix_i_min, vix_i_max, vix_j_min, vix_j_max, vix_step,
                          atr_n_min, atr_n_max, atr_T_min, atr_T_max, atr_T_step,
                          min_occurrence_count):
    """
    Runs the full VIX and ATR optimization with a given min_occurrence_count threshold
    for the VIX conditions.
    """
    print(f"\n--- Running Optimization with min_occurrence_count={min_occurrence_count} ---\n")

    trades_df = precompute_trades(df, vix_df, initial_investment)

    if trades_df.empty:
        print("No valid trades found based on initial criteria.")
        return None, None, None, None, None

    print(f"Total Valid Trades: {len(trades_df)}")

    # VIX optimization based on success rate (and occurrence), filtered by min_occurrence_count
    best_i, best_j, all_vix_results = find_best_vix_conditions_optimized(
        trades_df=trades_df,
        i_min=vix_i_min,
        i_max=vix_i_max,
        j_min=vix_j_min,
        j_max=vix_j_max,
        step=vix_step,
        min_occurrence_count=min_occurrence_count
    )

    if best_i is None or best_j is None:
        print("No suitable VIX conditions found after applying minimum occurrence count filter.")
        return None, None, None, None, None

    trades_best_vix = trades_df[(trades_df['VIX_Open'] > best_i) & (trades_df['VIX_Open'] < best_j)]
    if trades_best_vix.empty:
        print("No trades found with the best VIX conditions.")
        return best_i, best_j, None, None, None

    # ATR optimization based on total profit
    n_values = range(atr_n_min, atr_n_max + 1)
    T_values = np.round(np.arange(atr_T_min, atr_T_max + atr_T_step, atr_T_step), 1)

    best_n, best_T, all_atr_results = optimize_atr_parameters(
        trades_df=trades_best_vix,
        df=df,
        initial_investment=initial_investment,
        n_values=n_values,
        T_values=T_values
    )

    if best_n is None or best_T is None:
        print("No suitable ATR parameters found.")
        return best_i, best_j, None, None, None

    print(f"\nAnalyzing trades with best VIX (i={best_i}, j={best_j}) and ATR (n={best_n}, T={best_T}) parameters")

    df_with_atr = calculate_atr(df.copy(), best_n)
    trades_final = trades_df.merge(df_with_atr[['Date', 'ATR']], on='Date', how='left')
    trades_final = trades_final[
        (trades_final['VIX_Open'] > best_i) & 
        (trades_final['VIX_Open'] < best_j) & 
        (trades_final['ATR'] <= best_T)
    ]

    if not trades_final.empty:
        total_profit = trades_final['Profit/Loss'].sum()
        occurrence_count = len(trades_final)
        success_count = (trades_final['Profit/Loss'] > 0).sum()
        success_rate = success_count / occurrence_count if occurrence_count > 0 else 0.0
    else:
        total_profit = 0.0
        occurrence_count = 0
        success_count = 0
        success_rate = 0.0

    print(f"\nFinal Results with min_occurrence_count={min_occurrence_count}:")
    print(f"Occurrences: {occurrence_count}, Successes: {success_count}, Success Rate: {success_rate:.2f}, Total Profit: ${total_profit:,.2f}")

    # Optionally, save results
    all_vix_results.to_csv(f'combined_vix_atr_vix_optimization_results_{min_occurrence_count}.csv', index=False)
    all_atr_results.to_csv(f'combined_vix_atr_atr_optimization_results_{min_occurrence_count}.csv', index=False)

    num_winning_trades = trades_final[trades_final['Profit/Loss'] > 0].shape[0]
    num_losing_trades = trades_final[trades_final['Profit/Loss'] < 0].shape[0]

    print(f"\nNumber of Winning Trades: {num_winning_trades}")
    print(f"Number of Losing Trades: {num_losing_trades}")

    print("\nTrade Details:")
    if not trades_final.empty:
        print(trades_final.to_string(index=False))
    else:
        print("No trades meet both VIX and ATR conditions.")

    print("\n=== Combined Optimization Summary ===")
    print(f"Optimal VIX Conditions (Success-Based): i={best_i}, j={best_j}")
    print(f"Optimal ATR Parameters (Profit-Based): n={best_n}, T={best_T}")
    print(f"Occurrences: {occurrence_count}, Successes: {success_count}, Success Rate: {success_rate:.2f}, Total Profit: ${total_profit:,.2f}")

    return best_i, best_j, best_n, best_T, total_profit

def analyze_spy_gaps_combined(main_file, vix_file, initial_investment, 
                              vix_i_min=10.0, vix_i_max=30.0, 
                              vix_j_min=10.0, vix_j_max=30.0, vix_step=0.1,
                              atr_n_min=2, atr_n_max=20, atr_T_min=10.0, atr_T_max=20.0, atr_T_step=0.1):
    print("Loading stock data...")
    df = load_data(main_file)
    print("Loading VIX data...")
    vix_df = load_vix_data(vix_file)

    # Run full optimization with minimum occurrence count = 30
    run_full_optimization(df, vix_df, initial_investment, 
                          vix_i_min, vix_i_max, vix_j_min, vix_j_max, vix_step,
                          atr_n_min, atr_n_max, atr_T_min, atr_T_max, atr_T_step,
                          min_occurrence_count=30)

    # Run full optimization with minimum occurrence count = 50
    run_full_optimization(df, vix_df, initial_investment, 
                          vix_i_min, vix_i_max, vix_j_min, vix_j_max, vix_step,
                          atr_n_min, atr_n_max, atr_T_min, atr_T_max, atr_T_step,
                          min_occurrence_count=61)

# Example usage:
if __name__ == "__main__":
    initial_investment = 16000  # $16,000 initial investment
    main_file = 'D6CK32X6.csv'
    vix_file = 'VIX_History.csv'

    if not os.path.exists(main_file):
        print(f"Error: Main data file '{main_file}' not found.")
    elif not os.path.exists(vix_file):
        print(f"Error: VIX data file '{vix_file}' not found.")
    else:
        analyze_spy_gaps_combined(
            main_file=main_file,
            vix_file=vix_file,
            initial_investment=initial_investment,
            vix_i_min=10.0,
            vix_i_max=30.0,
            vix_j_min=10.0,
            vix_j_max=30.0,
            vix_step=0.1,
            atr_n_min=2,
            atr_n_max=20,
            atr_T_min=3.0,
            atr_T_max=20.0,
            atr_T_step=0.1
        )
