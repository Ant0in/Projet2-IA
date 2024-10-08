
from competitive_env import CompetitiveEnv, A, S
from functools import lru_cache



class ExpectimaxUtil:

    @staticmethod
    @lru_cache(maxsize=None)
    def value(mdp: CompetitiveEnv[A, S], cstate: S, depth: int, maxdepth: int) -> float:
        
        # Si état final ou maxdepth reached, alors on return le cstate.
        if mdp.is_final(state=cstate) or depth >= maxdepth:
            return cstate.value
        
        # Si l'agent actuel est le joueur max, alors l'action à faire est une action max.
        elif cstate.current_agent == 0:
            return ExpectimaxUtil.max_value(mdp=mdp, cstate=cstate, depth=depth+1, maxdepth=maxdepth)
        
        # Si l'agent actuel est le joueur min, alors l'action à faire est une action min.
        elif cstate.current_agent == 1:
            return ExpectimaxUtil.exp_value(mdp=mdp, cstate=cstate, depth=depth+1, maxdepth=maxdepth)
        
        else: raise ValueError(f'[E] Unknown procedure for state {cstate}.')

    @staticmethod
    @lru_cache(maxsize=None)
    def max_value(mdp: CompetitiveEnv, cstate: S, maxdepth: int, depth: int = 0) -> float:
        
        v: float = -float('inf')
        # Pour chacun des sucesseurs, on va compute sa valeur. 
        # Si elle est plus petite que v, on peut la choisir.
        for action in mdp.available_actions(state=cstate):
            sucessor = mdp.step(state=cstate, action=action)
            v = max(v, ExpectimaxUtil.value(mdp=mdp, cstate=sucessor, depth=depth, maxdepth=maxdepth))
        return v

    @staticmethod
    @lru_cache(maxsize=None)
    def exp_value(mdp: CompetitiveEnv, cstate: S, maxdepth: int, depth: int = 0) -> float:

        v: float = 0.0
        # On définit alpha comme la probabilité d'un sucesseur pour une distribution uniforme.
        # Pas besoin de vérifier qu'il y'a des sucessors, la vérification est faite dans value.
        sucessors: list = mdp.available_actions(state=cstate)
        alpha: float = 1 / len(sucessors)
        # On va récupérer l'espérance de chacun des sucesseurs.
        for action in sucessors:
            sucessor = mdp.step(state=cstate, action=action)
            v += alpha * ExpectimaxUtil.value(mdp=mdp, cstate=sucessor, depth=depth, maxdepth=maxdepth)
        return v

