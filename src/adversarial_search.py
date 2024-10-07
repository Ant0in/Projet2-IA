from typing import Optional
from competitive_env import CompetitiveEnv, S, A



class Minimax:

    @staticmethod
    def value(mdp: CompetitiveEnv[A, S], cstate: S, depth: int, maxdepth: int) -> float:
        if mdp.is_final(state=cstate) or depth >= maxdepth: return cstate
        elif cstate.current_agent == 0: return Minimax.min_value(mdp=mdp, cstate=cstate, depth=depth, maxdepth=maxdepth)
        elif cstate.current_agent == 1: return Minimax.max_value(mdp=mdp, cstate=cstate, depth=depth, maxdepth=maxdepth)
        else: raise ValueError(f'[E] Unknown val function case for state {cstate}')

    @staticmethod
    def max_value(mdp: CompetitiveEnv, cstate: S, maxdepth: int, depth: int = 0) -> float:
        v: float = -float('inf')
        for sucessor in mdp.available_actions(state=cstate):
            v = max(v, Minimax.value(mdp=mdp, cstate=sucessor, depth=depth, maxdepth=maxdepth), key=lambda s: s.value)
        return v

    @staticmethod
    def min_value(mdp: CompetitiveEnv, cstate: S, maxdepth: int, depth: int = 0) -> float:
        v: float = float('inf')
        for sucessor in mdp.available_actions(state=cstate):
            v = min(v, Minimax.value(mdp=mdp, cstate=sucessor, depth=depth, maxdepth=maxdepth), key=lambda s: s.value)
        return v


def minimax(mdp: CompetitiveEnv[A, S], state: S, max_depth: int) -> Optional[A]:
    
    best_state: S = Minimax.value(mdp=mdp, cstate=state, depth=0, maxdepth=max_depth)
    return best_state


def alpha_beta(mdp: CompetitiveEnv[A, S], state: S, max_depth: int) -> Optional[A]:
    return None

def expectimax(mdp: CompetitiveEnv[A, S], state: S, max_depth: int) -> Optional[A]:
    return None

