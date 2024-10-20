

from typing import Optional
from .competitive_env import CompetitiveEnv, S, A

from .minimax import MinimaxUtil
from .expectimax import ExpectimaxUtil
from .alphabetapruning import AlphaBetaPruningUtil




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
