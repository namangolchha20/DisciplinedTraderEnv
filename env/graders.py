from .environment import DisciplinedTraderEnv

EPS = 1e-9

def grade_easy(env: DisciplinedTraderEnv) -> float:
    final_cash = env.cash
    score = min(1.0, max(0, (final_cash - 10000) / 5000)) * 0.7
    if len(env.trades) <= 5:
        score += 0.3
    return max(EPS, min(1.0 - EPS, score))

def grade_medium(env: DisciplinedTraderEnv) -> float:
    total_profit = sum(p for _,_,p in env.trades)
    profit_score = min(1.0, max(0, total_profit / 2000)) * 0.5
    wins = sum(1 for _,_,p in env.trades if p>0)
    win_rate = wins / len(env.trades) if env.trades else 0.0
    wr_score = win_rate * 0.3
    score = profit_score + wr_score
    return max(EPS, min(1.0 - EPS, score))

def grade_hard(env: DisciplinedTraderEnv) -> float:
    total_profit = sum(p for _,_,p in env.trades)
    profit_score = min(1.0, max(0, total_profit / 3000)) * 0.4
    trade_count = len(env.trades)
    overtrade_penalty = min(1.0, trade_count / 30) * 0.3
    score = profit_score + (1 - overtrade_penalty)
    return max(EPS, min(1.0 - EPS, score))