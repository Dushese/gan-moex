import pandas as pd
import torch

###################################################################################
# Stocks universe definition
###################################################################################

# Define number of tickers in universe
N_ASSETS = 5

# Define number of trading days in a year
N_TRADING_DAYS = 252

###################################################################################
# Constants to assess portfolio performance
###################################################################################

# Define confidence interval for statistics
CONFIDENCE_INTERVAL = 0.95

# Number of periods to shift returns:
# 1. We make trading decision at the end of the day t
# 2. Enter position at day (t + 1)
# 3. Exit it at day (t + 2)
# 4. Get the return at (t + 2)
SHIFT_RETURNS = 2


def get_shifted_returns(df_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Shift returns
    df_shifted_returns[i] corresponds to return of signal[i]
    """
    return df_returns.shift(-SHIFT_RETURNS)

###################################################################################
# Constants to train GAN
###################################################################################


WINDOW_SIZE = 360#N_TRADING_DAYS // 2 # TODO
BATCH_SIZE = 256
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
