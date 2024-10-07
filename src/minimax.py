
from competitive_env import CompetitiveEnv, A, S


class MinimaxUtil:

    @staticmethod
    def value(mdp: CompetitiveEnv[A, S], cstate: S, depth: int, maxdepth: int) -> float:
        
        # Si état final ou maxdepth reached, alors on return le cstate.
        if mdp.is_final(state=cstate) or depth >= maxdepth:
            return cstate.value
        
        # Si l'agent actuel est le joueur max, alors l'action à faire est une action max.
        elif cstate.current_agent == 0:
            return MinimaxUtil.max_value(mdp=mdp, cstate=cstate, depth=depth+1, maxdepth=maxdepth)
        
        # Si l'agent actuel est le joueur min, alors l'action à faire est une action min.
        elif cstate.current_agent == 1:
            return MinimaxUtil.min_value(mdp=mdp, cstate=cstate, depth=depth+1, maxdepth=maxdepth)
        
        else: raise ValueError(f'[E] Unknown procedure for state {cstate}.')

    @staticmethod
    def max_value(mdp: CompetitiveEnv, cstate: S, maxdepth: int, depth: int = 0) -> float:
        
        v: float = -float('inf')
        # Pour chacun des sucesseurs, on va compute sa valeur. 
        # Si elle est plus petite que v, on peut la choisir.
        for action in mdp.available_actions(state=cstate):
            sucessor = mdp.step(state=cstate, action=action)
            v = max(v, MinimaxUtil.value(mdp=mdp, cstate=sucessor, depth=depth, maxdepth=maxdepth))
        return v

    @staticmethod
    def min_value(mdp: CompetitiveEnv, cstate: S, maxdepth: int, depth: int = 0) -> float:

        v: float = float('inf')
        # Pour chacun des sucesseurs, on va compute sa valeur. 
        # Si elle est plus petite que v, on peut la choisir.
        for action in mdp.available_actions(state=cstate):
            sucessor = mdp.step(state=cstate, action=action)
            v = min(v, MinimaxUtil.value(mdp=mdp, cstate=sucessor, depth=depth, maxdepth=maxdepth))
        return v
