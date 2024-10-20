from .competitive_env import CompetitiveEnv, A, S
from functools import lru_cache



class AlphaBetaPruningUtil:

    @staticmethod
    @lru_cache(maxsize=None)
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
    @lru_cache(maxsize=None)
    def max_value(mdp: CompetitiveEnv[A, S], cstate: S, depth: int, maxdepth: int, alpha: float, beta: float) -> float:

        v: float = -float('inf')

        # On récupère chaque sucesseur, puis on calcule la valeur de alpha.
        # Si alpha est + grand que beta, alors on peut prune et éviter de visiter les autres sucesseurs.
        for action in mdp.available_actions(state=cstate):
            sucessor = mdp.step(state=cstate, action=action)
            v = max(v, AlphaBetaPruningUtil.value(mdp=mdp, cstate=sucessor, depth=depth, maxdepth=maxdepth, alpha=alpha, beta=beta))
            alpha = max(v, alpha)
            if beta <= alpha:
                break  # prune
        return v
    
    @staticmethod
    @lru_cache(maxsize=None)
    def min_value(mdp: CompetitiveEnv[A, S], cstate: S, depth: int, maxdepth: int, alpha: float, beta: float) -> float:

        v: float = float('inf')

        # On récupère chaque sucesseur, puis on calcule la valeur de beta.
        # Si alpha est + grand que beta, alors on peut prune et éviter de visiter les autres sucesseurs.
        for action in mdp.available_actions(state=cstate):
            sucessor = mdp.step(state=cstate, action=action)
            v = min(v, AlphaBetaPruningUtil.value(mdp=mdp, cstate=sucessor, depth=depth, maxdepth=maxdepth, alpha=alpha, beta=beta))
            beta = min(v, beta)
            if beta <= alpha:
                break  # prune
        return v
    
