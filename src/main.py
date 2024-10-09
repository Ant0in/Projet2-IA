

from adversarial_search import minimax, alpha_beta, expectimax
from competitive_env import CompetitiveEnv, State, S, A
from competitive_world import CompetitiveWorld
from lle import World




if __name__ == '__main__':

    w: str = \
    """
    S0 . G G
    G  @ @ @
    .  . X X
    S1 . . .
    """

    mdp: CompetitiveWorld = CompetitiveWorld(World(w))
    a = alpha_beta(mdp=mdp, state=mdp.reset(), max_depth=5)
    print(a)