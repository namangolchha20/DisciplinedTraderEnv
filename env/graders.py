"""Task graders for DisciplinedTraderEnv.

Each grader returns a score in the open interval (0, 1) and is built from
independent, bounded components so no single behaviour can be exploited:

  * profit      - realised + unrealised PnL vs. a task-specific target
  * win rate    - fraction of closed trades that were profitable
  * discipline  - traded enough to engage, but did not overtrade
  * drawdown    - peak-to-trough equity loss vs. a task-specific budget

All component weights sum to exactly 1.0, so scores are well calibrated
across tasks (a do-nothing agent can never score above its discipline /
drawdown floor, and a profitable-but-reckless agent is capped too).
"""

from .environment import DisciplinedTraderEnv

EPS = 1e-9
INITIAL_CAPITAL = 10_000.0


def _clamp01(x: float) -> float:
    """Clamp into the open interval (0, 1) expected by OpenEnv."""
    return max(EPS, min(1.0 - EPS, x))


def _unit(x: float) -> float:
    """Clamp a raw ratio into [0, 1]."""
    return max(0.0, min(1.0, x))


def _final_equity(env: DisciplinedTraderEnv) -> float:
    return env.account_value()


def _win_rate(env: DisciplinedTraderEnv) -> float:
    if not env.trades:
        return 0.0
    wins = sum(1 for _, _, p in env.trades if p > 0)
    return wins / len(env.trades)


def _drawdown_score(env: DisciplinedTraderEnv, budget: float) -> float:
    """1.0 when max drawdown is 0, linearly down to 0.0 at `budget`."""
    return 1.0 - _unit(env.max_drawdown / budget)


def _discipline_score(env: DisciplinedTraderEnv, min_trades: int, max_trades: int) -> float:
    """Full credit inside [min_trades, max_trades], fading to 0 outside."""
    n = len(env.trades)
    if n < min_trades:
        return n / max(min_trades, 1)
    if n <= max_trades:
        return 1.0
    # Linear fade: zero credit at 2x the allowed trade count.
    overshoot = (n - max_trades) / max(max_trades, 1)
    return _unit(1.0 - overshoot)


def grade_easy(env: DisciplinedTraderEnv) -> float:
    """Easy: gentle uptrend. Reward capturing it with a few clean trades."""
    profit = _final_equity(env) - INITIAL_CAPITAL
    profit_score = _unit(profit / 1_500.0) * 0.5
    discipline = _discipline_score(env, min_trades=1, max_trades=8) * 0.2
    drawdown = _drawdown_score(env, budget=0.10) * 0.3
    return _clamp01(profit_score + discipline + drawdown)


def grade_medium(env: DisciplinedTraderEnv) -> float:
    """Medium: noisier market. Consistency (win rate) matters as much as PnL."""
    profit = _final_equity(env) - INITIAL_CAPITAL
    profit_score = _unit(profit / 2_000.0) * 0.4
    wr_score = _win_rate(env) * 0.3
    drawdown = _drawdown_score(env, budget=0.15) * 0.3
    return _clamp01(profit_score + wr_score + drawdown)


def grade_hard(env: DisciplinedTraderEnv) -> float:
    """Hard: trendless, high volatility. Survival and restraint are the test."""
    profit = _final_equity(env) - INITIAL_CAPITAL
    profit_score = _unit(profit / 3_000.0) * 0.4
    discipline = _discipline_score(env, min_trades=1, max_trades=30) * 0.3
    drawdown = _drawdown_score(env, budget=0.20) * 0.3
    return _clamp01(profit_score + discipline + drawdown)


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade(env: DisciplinedTraderEnv, task: str = None) -> float:
    """Grade an environment with the grader matching its task."""
    task = task or env.task or "easy"
    return GRADERS.get(task, grade_easy)(env)
