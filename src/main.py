from lle import World
from competitive_world import CompetitiveWorld
import random
from adversarial_search import minimax



if __name__ == "__main__":
    
    world = CompetitiveWorld(
        World(
            """
        .  L1W
        S0 S1
        X  X"""
        )
    )

    minimax(mdp=world, state=world.reset().world_state, max_depth=5)