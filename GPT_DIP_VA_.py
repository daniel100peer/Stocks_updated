import pandas as pd
import numpy as np
import os
from itertools import product
from multiprocessing import Pool, cpu_count

# =========== פונקציות בסיסיות ===========

def load_data(file_name):
    """
    טוען קובץ CSV המכיל עמודות [Date, Open, High, Low, Close, ...].
    ממיין לפי תאריך ומאפס את האינדקס.
    """
    df = pd.read_csv(file_name, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_rolling_average_drop(df, window=21):
    """
    פונקציית עזר לאסטרטגיית 'Red Days':
    
    1. מגדיר 'יום אדום' אם Close < Open.
    2. מחשב daily_drop = (Open - Close) עבור יום אדום, ואפס ליום ירוק.
    3. מחשב rolling_avg_drop כממוצע נע של daily_drop (window=21 ברירת מחדל),
       ומבצע shift(1) כדי שהיום הנוכחי לא ישפיע על עצמו.
    """
    df = df.copy()
    df['IsRedDay'] = df['Close'] < df['Open']
    df['DailyDrop'] = np.where(df['IsRedDay'], df['Open'] - df['Close'], 0.0)
    
    # rolling mean של window ימים, עם shift(1)
    df['RollingAvgDrop'] = df['DailyDrop'].rolling(window=window).mean().shift(1)
    
    # ממלא NaN ב-0 עבור הימים הראשונים
    df['RollingAvgDrop'].fillna(0.0, inplace=True)
    
    return df

def calculate_profit_loss_long(entry_price, exit_price, position_size):
    """
    מחשב רווח/הפסד עבור פוזיציית לונג: (exit_price - entry_price) * (position_size / entry_price).
    """
    if entry_price <= 0:
        return 0.0  # להימנע מחלוקה ב-0
    shares = position_size / entry_price
    return shares * (exit_price - entry_price)

def generate_red_day_trades(df, initial_investment=16000, rolling_window=21):
    """
    מייצר את הטריידים לפי אסטרטגיית 'Red Days':
    
    1. מחשב RollingAvgDrop דרך calculate_rolling_average_drop.
    2. בכל יום, בודק אם Low של היום קטן או שווה ל- (Open - RollingAvgDrop).
       אם כן, נכנס ל-Long במחיר הזה ויוצא ב-Close של אותו יום.
    3. מחזיר DataFrame עם כל הטריידים: [Date, Entry Price, Exit Price, Profit/Loss, ...]
    """
    df_calc = calculate_rolling_average_drop(df, window=rolling_window)
    trades = []
    
    for i in range(1, len(df_calc)):
        current_day = df_calc.iloc[i]
        
        avg_drop = current_day['RollingAvgDrop']
        if avg_drop <= 0:
            continue
        
        open_price = current_day['Open']
        low_price = current_day['Low']
        close_price = current_day['Close']
        date = current_day['Date']
        
        desired_entry = open_price - avg_drop
        if low_price <= desired_entry:
            entry_price = desired_entry
            exit_price = close_price
            profit_loss = calculate_profit_loss_long(entry_price, exit_price, initial_investment)
            
            trades.append({
                'Date': date,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Daily Open': open_price,
                'Daily Low': low_price,
                'Daily Close': close_price,
                'Rolling Avg Drop': avg_drop,
                'Profit/Loss': profit_loss
            })
    
    trades_df = pd.DataFrame(trades)
    return trades_df

# =========== חישוב ATR ואופטימיזציית ATR ===========

def calculate_atr(df, n):
    """
    מחשב ATR על בסיס n ימי מסחר.
    TR = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
    ATR = ממוצע נע של TR על פני n ימים.
    """
    df = df.copy()
    df['Previous Close'] = df['Close'].shift(1)
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Previous Close'])
    df['Low-PrevClose'] = abs(df['Low'] - df['Previous Close'])
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=n).mean()
    return df

def compute_atr_metrics_sequential(args):
    """
    פונקציית עזר לתוך pool.map:
    בודקת עבור קומבינציית (n, T), כמה טריידים עוברים את הסף (ATR <= T), ואז מחשבת:
      - כמות טריידים
      - כמות הצלחות
      - אחוז הצלחה
      - רווח כולל
    """
    trades_df, df, n, T, initial_investment = args
    
    # מחשב ATR על ה-Daily Data
    df_with_atr = calculate_atr(df.copy(), n)
    
    # מוסיף את עמודת ATR לטריידים לפי תאריך
    trades_with_atr = trades_df.merge(df_with_atr[['Date', 'ATR']], on='Date', how='left')
    trades_with_atr.dropna(subset=['ATR'], inplace=True)
    
    # מסנן טריידים ש-ATR שלהם קטן או שווה לסף T
    trades_filtered = trades_with_atr[trades_with_atr['ATR'] <= T]
    
    if not trades_filtered.empty:
        occurrence_count = len(trades_filtered)
        profit_losses = trades_filtered['Profit/Loss'].values
        success_count = (profit_losses > 0).sum()
        total_profit = profit_losses.sum()
        success_rate = success_count / occurrence_count
    else:
        occurrence_count = 0
        success_count = 0
        total_profit = 0.0
        success_rate = 0.0
    
    return (n, T, occurrence_count, success_count, success_rate, total_profit)

def optimize_atr_parameters(trades_df, df, initial_investment,
                            n_values, T_values):
    """
    אופטימיזציה ל-ATR:
    - מחזיר DataFrame עם עמודות:
      [n, T, OccurrenceCount, SuccessCount, SuccessRate, TotalProfit]
    - ממיין לפי SuccessRate (מהגבוה לנמוך).
    """
    param_combinations = list(product(n_values, T_values))
    args_list = [(trades_df.copy(), df.copy(), n, T, initial_investment) for (n, T) in param_combinations]
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(compute_atr_metrics_sequential, args_list)
    
    columns = ['n','T','OccurrenceCount','SuccessCount','SuccessRate','TotalProfit']
    results_df = pd.DataFrame(results, columns=columns)
    
    # ממיין לפי SuccessRate מהגבוה לנמוך
    results_df_sorted = results_df.sort_values(by='SuccessRate', ascending=False).reset_index(drop=True)
    
    return results_df_sorted

# =========== פונקציה מרכזית לניתוח האסטרטגיה בשילוב אופטימיזציית ATR בלבד ===========

def analyze_spy_red_days_atr_only(main_file, vix_file,
                                  initial_investment=16000,
                                  rolling_window=21,
                                  atr_n_min=2, atr_n_max=20,
                                  atr_T_min=3.0, atr_T_max=20.0, atr_T_step=1.0):
    """
    1. טוען נתוני מניה מ-main_file (למשל SPY).
    2. (vix_file לא בשימוש בפועל, רק כפרמטר שצריך להישאר בקוד).
    3. מייצר טריידים לפי אסטרטגיית Red Days.
    4. מריץ אופטימיזציית ATR (n, T) וממיין לפי Success Rate.
    5. מדפיס את הטופ של התוצאות + סיכום.
    """
    print(f"Loading main data from: {main_file}")
    df = load_data(main_file)
    
    # (לא משתמשים ב-vix_file בקוד הזה, רק מציגים הודעה)
    print(f"Ignoring vix file (not in use): {vix_file}")
    
    # 1) יצירת טריידים לפי Red Days
    red_day_trades_df = generate_red_day_trades(df, initial_investment=initial_investment, rolling_window=rolling_window)
    print(f"Total trades from Red Days strategy: {len(red_day_trades_df)}")
    if red_day_trades_df.empty:
        print("No trades were generated. Exiting.")
        return
    
    # 2) אופטימיזציית ATR
    print("\n=== Optimizing ATR (sorting by Success Rate) ===")
    n_values = range(atr_n_min, atr_n_max + 1)  # לדוגמה: 2 עד 20
    T_values = np.round(np.arange(atr_T_min, atr_T_max + atr_T_step, atr_T_step), 1)  # לדוגמה: 3.0 עד 20.0 בצעדי 1.0
    
    results_df_sorted = optimize_atr_parameters(
        trades_df=red_day_trades_df,
        df=df,
        initial_investment=initial_investment,
        n_values=n_values,
        T_values=T_values
    )
    
    # הצגת תוצאות
    if results_df_sorted.empty:
        print("No ATR parameter combinations yielded any trades. Exiting.")
        return
    
    print("\nTop 20 ATR Combinations (sorted by Success Rate):")
    print(results_df_sorted.head(20).to_string(index=False))
    
    # הטוב ביותר לפי Success Rate
    best_n = results_df_sorted.loc[0, 'n']
    best_T = results_df_sorted.loc[0, 'T']
    best_occurrence_count = results_df_sorted.loc[0, 'OccurrenceCount']
    best_success_count = results_df_sorted.loc[0, 'SuccessCount']
    best_success_rate = results_df_sorted.loc[0, 'SuccessRate']
    best_total_profit = results_df_sorted.loc[0, 'TotalProfit']
    
    print(f"\n=== Best ATR Parameters by Success Rate ===")
    print(f"n={best_n}, T={best_T}")
    print(f"Occurrences: {best_occurrence_count}, Successes: {best_success_count}, Success Rate: {best_success_rate:.2f}, Total Profit: ${best_total_profit:,.2f}")
    
    # אופציונלי: יצירת DataFrame סופי של הטריידים שממלאים את התנאים הטובים ביותר
    df_best_atr = calculate_atr(df.copy(), best_n)
    trades_best_atr = red_day_trades_df.merge(df_best_atr[['Date','ATR']], on='Date', how='left')
    trades_best_atr.dropna(subset=['ATR'], inplace=True)
    trades_best_atr = trades_best_atr[trades_best_atr['ATR'] <= best_T]
    
    print(f"\nNumber of Trades meeting best (n={best_n}, T={best_T}): {len(trades_best_atr)}")
    if not trades_best_atr.empty:
        final_success_count = (trades_best_atr['Profit/Loss'] > 0).sum()
        final_success_rate = final_success_count / len(trades_best_atr)
        final_profit = trades_best_atr['Profit/Loss'].sum()
        
        print(f"Final Success Rate on these trades: {final_success_rate:.2f}")
        print(f"Final Total Profit: ${final_profit:,.2f}")
        
        print("\n=== Trades Details (Top best ATR) ===")
        print(trades_best_atr.to_string(index=False))
    else:
        print("No trades meet the best ATR conditions.")


# =========== דוגמה לשימוש ===========

if __name__ == "__main__":
    # שים לב: vix_file לא בשימוש, אך נשמר כפרמטר
    main_file = "7J0OK1VR.csv"   # להחליף בשם קובץ הנתונים הרלוונטי
    vix_file  = "VIX_History.csv"  # לא בשימוש בקוד הזה, רק התייחסות לפי הבקשה
    
    if not os.path.exists(main_file):
        print(f"Error: Main data file '{main_file}' not found.")
    else:
        analyze_spy_red_days_atr_only(
            main_file=main_file,
            vix_file=vix_file,
            initial_investment=16000,
            rolling_window=21,
            atr_n_min=2,
            atr_n_max=20,
            atr_T_min=3.0,
            atr_T_max=20.0,
            atr_T_step=1.0
        )
