

from typing import Optional
from competitive_env import CompetitiveEnv, S, A

from minimax import MinimaxUtil
from expectimax import ExpectimaxUtil



class AlphaBetaPruningUtil:

    @staticmethod
    def value(mdp: CompetitiveEnv[A, S], cstate: S, depth: int, maxdepth: int, alpha: float, beta: float) -> float:
        
        # Si état final ou maxdepth reached, alors on return le cstate.
        if mdp.is_final(state=cstate) or depth >= maxdepth:
            return cstate.value
        
        # Si l'agent actuel est le joueur max, alors l'action à faire est une action max.
        elif cstate.current_agent == 0:
            return AlphaBetaPruningUtil.max_value(mdp=mdp, cstate=cstate, depth=depth+1, maxdepth=maxdepth, alpha=alpha, beta=beta)
        
        # Si l'agent actuel est le joueur min, alors l'action à faire est une action min.
        elif cstate.current_agent == 1:
            return AlphaBetaPruningUtil.min_value(mdp=mdp, cstate=cstate, depth=depth+1, maxdepth=maxdepth, alpha=alpha, beta=beta)
        
        else: raise ValueError(f'[E] Unknown procedure for state {cstate}.')

    @staticmethod
    def max_value(mdp: CompetitiveEnv[A, S], cstate: S, depth: int, maxdepth: int, alpha: float, beta: float) -> float:

        v: float = -float('inf')

        for action in mdp.available_actions(state=cstate):
            sucessor = mdp.step(state=cstate, action=action)
            v = max(v, MinimaxUtil.value(mdp=mdp, cstate=sucessor, depth=depth, maxdepth=maxdepth))
            alpha = max(v, alpha)
            if beta <= alpha:
                break
        return v
    
    @staticmethod
    def min_value(mdp: CompetitiveEnv[A, S], cstate: S, depth: int, maxdepth: int, alpha: float, beta: float) -> float:

        v: float = float('inf')

        for action in mdp.available_actions(state=cstate):
            sucessor = mdp.step(state=cstate, action=action)
            v = min(v, MinimaxUtil.value(mdp=mdp, cstate=sucessor, depth=depth, maxdepth=maxdepth))
            beta = min(v, beta)
            if beta <= alpha:
                break
        return v



def minimax(mdp: CompetitiveEnv[A, S], state: S, max_depth: int) -> Optional[A]:
    
    if state.current_agent != 0: raise ValueError(f'[E] Minimax needs current_agent to be the player (=0). (got {state.current_agent})')

    best_move: A = None
    best_val: float = -float('inf')

    # On obtient la meilleure action disponible, munie de son meilleur score.
    for a in mdp.available_actions(state=state):

        sucessor: S = mdp.step(state=state, action=a)
        sucessor_eval: float = MinimaxUtil.value(mdp=mdp, cstate=sucessor, depth=1, maxdepth=max_depth)
        
        if sucessor_eval >= best_val:
            best_move = a
            best_val = sucessor_eval

    return best_move

def expectimax(mdp: CompetitiveEnv[A, S], state: S, max_depth: int) -> Optional[A]:
    
    if state.current_agent != 0: raise ValueError(f'[E] Expectimax needs current_agent to be the player (=0). (got {state.current_agent})')

    best_move: A = None
    best_val: float = -float('inf')

    # On obtient la meilleure action disponible, munie de son meilleur score.
    for a in mdp.available_actions(state=state):

        sucessor: S = mdp.step(state=state, action=a)
        sucessor_eval: float = ExpectimaxUtil.value(mdp=mdp, cstate=sucessor, depth=1, maxdepth=max_depth)
        
        if sucessor_eval >= best_val:
            best_move = a
            best_val = sucessor_eval

    return best_move

def alpha_beta(mdp: CompetitiveEnv[A, S], state: S, max_depth: int) -> Optional[A]:
    
    if state.current_agent != 0: raise ValueError(f'[E] Expectimax needs current_agent to be the player (=0). (got {state.current_agent})')

    best_move: A = None
    best_val: float = -float('inf')

    # On obtient la meilleure action disponible, munie de son meilleur score.
    for a in mdp.available_actions(state=state):

        sucessor: S = mdp.step(state=state, action=a)
        sucessor_eval: float = AlphaBetaPruningUtil.value(mdp=mdp, cstate=sucessor, depth=1, maxdepth=max_depth, alpha=-float('inf'), beta=float('inf'))
        
        if sucessor_eval >= best_val:
            best_move = a
            best_val = sucessor_eval

    return best_move
