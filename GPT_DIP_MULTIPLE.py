import pandas as pd
import numpy as np

def load_data(file_name):
    """
    Loads the CSV file into a DataFrame, sorts it by date, and resets the index.
    """
    df = pd.read_csv(file_name, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_rolling_average_drop(df, window=21):
    """
    1. מגדיר 'יום אדום' אם Close < Open.
    2. מחשב daily_drop = (Open - Close) עבור יום אדום, ואפס ליום ירוק.
    3. מחשב ממוצע נע ל-21 ימים אחרונים על מנת לקבל rolling_avg_drop (shift(1) כדי שלא לכלול את היום הנוכחי).
    """
    df = df.copy()
    df['IsRedDay'] = df['Close'] < df['Open']
    df['DailyDrop'] = np.where(df['IsRedDay'], df['Open'] - df['Close'], 0.0)
    
    df['RollingAvgDrop'] = df['DailyDrop'].rolling(window=window).mean().shift(1)
    df['RollingAvgDrop'].fillna(0.0, inplace=True)
    return df

def calculate_profit_loss_long(entry_price, exit_price, position_size):
    """
    מחשב רווח/הפסד עבור פוזיציית לונג: (exit_price - entry_price) * number_of_shares
    כאשר number_of_shares = position_size / entry_price.
    """
    if entry_price <= 0:
        return 0.0
    shares = position_size / entry_price
    return shares * (exit_price - entry_price)

def analyze_red_days_strategy_for_factor(df, factor, initial_investment=16000):
    """
    אסטרטגיה ליום בודד עבור Factor נתון:
    1. במקום open_price - RollingAvgDrop, נשתמש ב-open_price - factor * RollingAvgDrop
    2. נבדוק אם הגיעו ל-Low <= נקודת הכניסה הצפויה
    3. אם כן, נכנסים ל-Long במחיר הכניסה (desired_entry) וסוגרים בסגירה של אותו יום
    4. מחזירים DataFrame של כל הטריידים
    """
    trades = []
    
    for i in range(len(df)):
        if i == 0:
            continue  # אין יום קודם
    
        current_day = df.iloc[i]
        avg_drop = current_day['RollingAvgDrop']
        
        # אם הממוצע <= 0 אין לנו מה לעשות
        if avg_drop <= 0:
            continue
        
        open_price = current_day['Open']
        low_price = current_day['Low']
        close_price = current_day['Close']
        
        desired_entry = open_price - factor * avg_drop
        
        if low_price <= desired_entry:
            # נכנסים ל-Long במחיר desired_entry
            entry_price = desired_entry
            exit_price = close_price
            profit_loss = calculate_profit_loss_long(entry_price, exit_price, initial_investment)
            
            trades.append({
                'Date': current_day['Date'],
                'Factor': factor,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Daily Open': open_price,
                'Daily Low': low_price,
                'Daily Close': close_price,
                'Rolling Avg Drop': avg_drop,
                'Profit/Loss': profit_loss
            })
    
    return pd.DataFrame(trades)

def calculate_statistics_for_trades(trades_df):
    """
    מחזירה מילון (dict) עם סטטיסטיקות בסיסיות:
    - כמות טריידים
    - אחוז הצלחה
    - רווח כולל
    """
    if trades_df.empty:
        return {
            'Total Trades': 0,
            'Success Rate': 0.0,
            'Total Profit': 0.0
        }
    
    total_trades = len(trades_df)
    profits = trades_df['Profit/Loss']
    
    winning = trades_df[profits > 0]
    losing = trades_df[profits < 0]
    
    success_rate = len(winning) / total_trades if total_trades > 0 else 0
    total_profit = profits.sum()
    
    stats = {
        'Total Trades': total_trades,
        'Success Rate': success_rate,
        'Total Profit': total_profit
    }
    return stats

def analyze_factors(file_name, initial_investment=16000, rolling_window=21):
    """
    1. טוען את הנתונים.
    2. מחשב rolling_average_drop.
    3. בלופ: עובר על כל Factor מ-1.1 עד 3.0 (בצעדים של 0.1).
    4. עבור כל Factor מפעיל את האסטרטגיה, מחשב סטטיסטיקות ושומר בטבלה.
    5. מציג את 10 הכפולות הטובות ביותר לפי אחוז הצלחה (ומדפיס את הרווח הכולל).
    """
    # 1. טוענים את הנתונים
    df = load_data(file_name)
    # 2. מחשבים rolling_average_drop
    df = calculate_rolling_average_drop(df, window=rolling_window)
    
    results = []
    
    factors = np.arange(1.1, 3.01, 0.1)  # 1.1, 1.2, 1.3, ..., 3.0
    # נוודא שהעיגול של 0.1 לא מייצר floating inaccuracies (אפשר גם לעגל כל Factor)
    
    for factor in factors:
        factor = round(factor, 2)  # לעגל לשתי ספרות אחרי הנקודה, למען ניקיון
        trades_df = analyze_red_days_strategy_for_factor(
            df, factor, initial_investment=initial_investment
        )
        stats = calculate_statistics_for_trades(trades_df)
        results.append({
            'Factor': factor,
            'Total Trades': stats['Total Trades'],
            'Success Rate': stats['Success Rate'],
            'Total Profit': stats['Total Profit']
        })
    
    # ממירים ל-DataFrame
    results_df = pd.DataFrame(results)
    # ממיינים לפי אחוז הצלחה מהגבוה לנמוך
    results_df.sort_values(by='Success Rate', ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    
    print("=== Top 10 Factors by Success Rate ===")
    # מציגים רק את 10 הכפולות הטובות ביותר
    print(results_df.head(10).to_string(index=False))
    
    return results_df

# דוגמה לשימוש
if __name__ == "__main__":
    file_name = "D6CK32X6.csv"  # החלף בקובץ הנכון
    results = analyze_factors(file_name, initial_investment=16000, rolling_window=21)
