"""
config.py — Paper parameters and contract definitions
References: Zhang, Zohren, Roberts (2019); [4] Baz et al. 2015; [27] Lim et al. 2019
"""

# =============================================================================
# Paper Parameters (Table 1 + references)
# =============================================================================
BP = 0.0020              # Transaction cost rate (20 bps)
TRADING_DAYS = 252        # Trading days per year
SIGMA_TGT_ANNUAL = 0.15   # Annualised volatility target [27]: "15%"
PORT_TGT_STD = 0.97       # Table 2 portfolio-level target std
EWMA_SPAN = 60            # EWMA span for σ_t estimation
SIGN_LOOKBACK = 252       # Sign(R) lookback window
MACD_PAIRS = [(8,24),(16,48),(32,96)]  # MACD time-scale pairs [4]
MACD_VOL_WINDOW = 63      # MACD price volatility normalisation window
MACD_STD_WINDOW = 252     # MACD signal standardisation window

# =============================================================================
# Test Period
# =============================================================================
TEST_START = '2011-01-01'
TEST_END   = '2019-12-31'
# Warmup: need at least 252 trading days before test start for indicators
# Using ALL available CLC data (from 1988+) for warmup

# =============================================================================
# CLC Contracts — 50 paper contracts minus 3 with no 2011-2019 data
# ZH (Heating Oil Electronic): all zeros in 2011-2019
# ZU (Crude Oil Electronic): all zeros in 2011-2019
# US (T-Bonds Composite): all NaN in 2011-2019
# =============================================================================
ASSET_CLASSES = {
    'Commodity': [t for t in [
        'CC','DA','GI','JO','KC','KW','LB','NR','SB','ZA',
        'ZC','ZF','ZG','ZH','ZI','ZK','ZL','ZN','ZO','ZP',
        'ZR','ZT','ZU','ZW','ZZ'
    ] if t not in ['ZH','ZU']],
    'Equity Index': ['CA','EN','ER','ES','LX','MD','SC','SP','XU','XX','YM'],
    'Fixed Income': [t for t in ['DT','FB','TY','UB','US'] if t != 'US'],
    'Forex': ['AN','BN','CN','DX','FN','JN','MP','NK','SN'],
}

EXCLUDED_CONTRACTS = {
    'ZH': 'Heating Oil Electronic — all zeros in 2011-2019',
    'ZU': 'Crude Oil Electronic — all zeros in 2011-2019',
    'US': 'T-Bonds Composite — all NaN in 2011-2019',
}

# =============================================================================
# Paper Target Values — Table 3 (Appendix B, "Raw Signal")
# =============================================================================
PAPER_TABLE3 = {
    'Commodity': {
        'Long':   {'E(R)':-0.298,'std(R)':0.412,'DD':0.258,'Sharpe':-0.723,'Sortino':-1.152,'MDD':0.248,'Calmar':-0.130,'% +ve':0.473,'Ave P/L':0.987},
        'Sign(R)': {'E(R)':0.101,'std(R)':0.312,'DD':0.185,'Sharpe':0.325,'Sortino':0.548,'MDD':0.082,'Calmar':0.115,'% +ve':0.494,'Ave P/L':1.081},
        'MACD':   {'E(R)':-0.039,'std(R)':0.227,'DD':0.136,'Sharpe':-0.174,'Sortino':-0.290,'MDD':0.132,'Calmar':-0.059,'% +ve':0.486,'Ave P/L':1.024},
    },
    'Equity Index': {
        'Long':   {'E(R)':0.504,'std(R)':0.928,'DD':0.606,'Sharpe':0.543,'Sortino':0.831,'MDD':0.127,'Calmar':0.466,'% +ve':0.541,'Ave P/L':0.928},
        'Sign(R)': {'E(R)':0.168,'std(R)':0.799,'DD':0.526,'Sharpe':0.211,'Sortino':0.319,'MDD':0.299,'Calmar':0.075,'% +ve':0.528,'Ave P/L':0.928},
        'MACD':   {'E(R)':-0.068,'std(R)':0.586,'DD':0.385,'Sharpe':-0.117,'Sortino':-0.178,'MDD':0.351,'Calmar':-0.041,'% +ve':0.519,'Ave P/L':0.904},
    },
    'Fixed Income': {
        'Long':   {'E(R)':0.605,'std(R)':0.939,'DD':0.561,'Sharpe':0.645,'Sortino':1.081,'MDD':0.108,'Calmar':0.455,'% +ve':0.515,'Ave P/L':1.048},
        'Sign(R)': {'E(R)':0.189,'std(R)':0.795,'DD':0.496,'Sharpe':0.237,'Sortino':0.381,'MDD':0.165,'Calmar':0.103,'% +ve':0.504,'Ave P/L':1.024},
        'MACD':   {'E(R)':0.136,'std(R)':0.609,'DD':0.367,'Sharpe':0.224,'Sortino':0.371,'MDD':0.124,'Calmar':0.131,'% +ve':0.485,'Ave P/L':1.102},
    },
    'Forex': {
        'Long':   {'E(R)':-0.198,'std(R)':0.472,'DD':0.285,'Sharpe':-0.420,'Sortino':-0.696,'MDD':0.219,'Calmar':-0.101,'% +ve':0.491,'Ave P/L':0.966},
        'Sign(R)': {'E(R)':-0.113,'std(R)':0.551,'DD':0.341,'Sharpe':-0.207,'Sortino':-0.332,'MDD':0.170,'Calmar':-0.071,'% +ve':0.499,'Ave P/L':0.968},
        'MACD':   {'E(R)':0.016,'std(R)':0.424,'DD':0.259,'Sharpe':0.037,'Sortino':0.061,'MDD':0.156,'Calmar':0.016,'% +ve':0.493,'Ave P/L':1.034},
    },
}

# =============================================================================
# Paper Target Values — Table 2 (with portfolio-level vol scaling)
# =============================================================================
PAPER_TABLE2 = {
    'Commodity': {
        'Long':   {'E(R)':-0.710,'std(R)':0.979,'DD':0.604,'Sharpe':-0.726,'Sortino':-1.177,'MDD':0.350,'Calmar':-0.140,'% +ve':0.473,'Ave P/L':0.989},
        'Sign(R)': {'E(R)':0.347,'std(R)':0.980,'DD':0.572,'Sharpe':0.354,'Sortino':0.606,'MDD':0.116,'Calmar':0.119,'% +ve':0.494,'Ave P/L':1.084},
        'MACD':   {'E(R)':-0.171,'std(R)':0.978,'DD':0.584,'Sharpe':-0.175,'Sortino':-0.293,'MDD':0.190,'Calmar':-0.060,'% +ve':0.486,'Ave P/L':1.026},
    },
    'Equity Index': {
        'Long':   {'E(R)':0.668,'std(R)':0.970,'DD':0.606,'Sharpe':0.688,'Sortino':1.102,'MDD':0.132,'Calmar':0.509,'% +ve':0.542,'Ave P/L':0.948},
        'Sign(R)': {'E(R)':0.228,'std(R)':0.966,'DD':0.610,'Sharpe':0.236,'Sortino':0.374,'MDD':0.344,'Calmar':0.077,'% +ve':0.528,'Ave P/L':0.930},
        'MACD':   {'E(R)':0.016,'std(R)':0.962,'DD':0.618,'Sharpe':0.017,'Sortino':0.027,'MDD':0.311,'Calmar':0.006,'% +ve':0.519,'Ave P/L':0.927},
    },
    'Fixed Income': {
        'Long':   {'E(R)':0.680,'std(R)':0.975,'DD':0.576,'Sharpe':0.698,'Sortino':1.180,'MDD':0.061,'Calmar':0.444,'% +ve':0.515,'Ave P/L':1.054},
        'Sign(R)': {'E(R)':0.214,'std(R)':0.972,'DD':0.592,'Sharpe':0.221,'Sortino':0.363,'MDD':0.080,'Calmar':0.083,'% +ve':0.504,'Ave P/L':1.019},
        'MACD':   {'E(R)':0.219,'std(R)':0.967,'DD':0.579,'Sharpe':0.228,'Sortino':0.380,'MDD':0.065,'Calmar':0.123,'% +ve':0.486,'Ave P/L':1.101},
    },
    'Forex': {
        'Long':   {'E(R)':-0.344,'std(R)':0.973,'DD':0.583,'Sharpe':-0.353,'Sortino':-0.590,'MDD':0.423,'Calmar':-0.097,'% +ve':0.491,'Ave P/L':0.979},
        'Sign(R)': {'E(R)':-0.297,'std(R)':0.973,'DD':0.592,'Sharpe':-0.306,'Sortino':-0.502,'MDD':0.434,'Calmar':-0.111,'% +ve':0.499,'Ave P/L':0.954},
        'MACD':   {'E(R)':0.006,'std(R)':0.970,'DD':0.582,'Sharpe':0.007,'Sortino':0.011,'MDD':0.329,'Calmar':0.002,'% +ve':0.493,'Ave P/L':1.029},
    },
}

METRIC_NAMES = ['E(R)','std(R)','DD','Sharpe','Sortino','MDD','Calmar','% +ve','Ave P/L']
