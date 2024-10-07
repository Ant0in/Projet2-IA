

from typing import Optional
from competitive_env import CompetitiveEnv, S, A
from lle import Action


class Minimax:

    @staticmethod
    def value(mdp: CompetitiveEnv[A, S], cstate: S, depth: int, maxdepth: int) -> float:
        
        # Si état final ou maxdepth reached, alors on return le cstate.
        if mdp.is_final(state=cstate) or depth >= maxdepth:
            return cstate.value
        
        # Si l'agent actuel est le joueur max, alors l'action à faire est une action max.
        elif cstate.current_agent == 0:
            return Minimax.max_value(mdp=mdp, cstate=cstate, depth=depth+1, maxdepth=maxdepth)
        
        # Si l'agent actuel est le joueur min, alors l'action à faire est une action min.
        elif cstate.current_agent == 1:
            return Minimax.min_value(mdp=mdp, cstate=cstate, depth=depth+1, maxdepth=maxdepth)
        
        else: raise ValueError(f'[E] Unknown procedure for state {cstate}.')

    @staticmethod
    def max_value(mdp: CompetitiveEnv, cstate: S, maxdepth: int, depth: int = 0) -> float:
        
        v: float = -float('inf')
        # Pour chacun des sucesseurs, on va compute sa valeur. 
        # Si elle est plus petite que v, on peut la choisir.
        for action in mdp.available_actions(state=cstate):
            sucessor = mdp.step(state=cstate, action=action)
            v = max(v, Minimax.value(mdp=mdp, cstate=sucessor, depth=depth, maxdepth=maxdepth))
        return v

    @staticmethod
    def min_value(mdp: CompetitiveEnv, cstate: S, maxdepth: int, depth: int = 0) -> float:

        v: float = float('inf')
        # Pour chacun des sucesseurs, on va compute sa valeur. 
        # Si elle est plus petite que v, on peut la choisir.
        for action in mdp.available_actions(state=cstate):
            sucessor = mdp.step(state=cstate, action=action)
            v = min(v, Minimax.value(mdp=mdp, cstate=sucessor, depth=depth, maxdepth=maxdepth))
        return v


def minimax(mdp: CompetitiveEnv[A, S], state: S, max_depth: int) -> Optional[A]:
    
    if state.current_agent != 0: raise ValueError(f'[E] Minimax needs current_agent to be the player (=0). (got {state.current_agent})')

    best_move: A = None
    best_val: float = -float('inf')

    # On obtient la meilleure action disponible, munie de son meilleur score.
    for a in mdp.available_actions(state=state):

        sucessor: S = mdp.step(state=state, action=a)
        sucessor_eval: float = Minimax.value(mdp=mdp, cstate=sucessor, depth=1, maxdepth=max_depth)
        
        if sucessor_eval >= best_val:
            best_move = a
            best_val = sucessor_eval

        print(a, sucessor_eval)

    return best_move


def alpha_beta(mdp: CompetitiveEnv[A, S], state: S, max_depth: int) -> Optional[A]:
    return None

def expectimax(mdp: CompetitiveEnv[A, S], state: S, max_depth: int) -> Optional[A]:
    return None

