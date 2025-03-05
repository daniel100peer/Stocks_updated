Project README

Overview

This project consists of Python scripts designed to analyze and optimize trading strategies based on stock and VIX data. It primarily focuses on strategies involving gap trades, VIX conditions, rolling averages of daily price drops, and ATR (Average True Range) optimizations to enhance profitability and trade success rate.

File Descriptions:

1. GPT_DIP.py

Implements a strategy based on identifying "Red Days" (days when closing price < opening price).

Calculates rolling averages of daily price drops and uses them to determine entry points for Long trades.

Evaluates and simulates trading performance.

2. GPT_DIP_MULTIPLE.py

Similar to GPT_DIP.py but extends the strategy by testing multiple entry factors (ranging from 1.1 to 3.0).

Optimizes and evaluates each factor, identifying the best-performing ones based on success rate.

3. GPT_DIP_VA_.py

Enhances the "Red Days" strategy with ATR optimizations, filtering trades by ATR values to maximize trade success.

Runs a comprehensive ATR parameter optimization to find the best trade conditions.

4. gpt_VA_succes_30_50.py

An advanced script for gap trade analysis, optimized based on VIX and ATR conditions.

Evaluates optimal VIX ranges and ATR parameters, employing parallel computing to improve performance.

4. gpt_VA_Skip_VIX_Drops(bySR)_short.py

Specializes in short trades, focusing on gaps where the market opens higher.

Incorporates logic to skip trades based on significant recent drops in the VIX.

Optimizes parameters for skipping trades to maximize the success rate.

5. gpt_VA_Skip_VIX_Drops(bySR)_long.py

Similar to the previous script but tailored specifically for long trades where the market opens lower.

Uses the same optimization techniques for VIX and ATR conditions.

Strategy Overview:

Gap Trades: Enter positions based on opening gaps relative to previous close and daily range.

VIX Conditions: Identify the optimal range of VIX values to enhance trade performance.

Rolling Average Drops: Use historical daily price drops to determine trade entry points.

Drop Filters: Skip trades following large drops in the VIX index to avoid adverse conditions.

ATR Optimization: Select the best ATR parameters to filter trades, optimizing the risk-reward ratio and profitability.

Usage:

Each script has an executable example at the bottom, which you can customize by changing parameters such as:

CSV file names (main_file, vix_file)

Initial investment amount (initial_investment)

Parameter ranges for optimization (X_values, Y_values, Z_values, atr_n_min, atr_n_max, etc.)

Ensure your CSV files contain the required columns:

Stock data (Date, Open, High, Low, Close)

VIX historical data (DATE, OPEN)

Dependencies:

Python libraries: pandas, numpy, itertools, multiprocessing

Notes:

All scripts include robust error handling and provide informative messages if data files are missing or no trades meet the specified conditions.

Warnings related to Pandas' operations have been suppressed for cleaner outputs.

This README summarizes the functionality and purpose of each provided script, helping you manage and optimize your trading strategies effectively.

