from lle import World
from competitive_world import CompetitiveWorld
import random


def main():
    """Vous pouvez tester et débugger votre programme ici"""
    world = CompetitiveWorld(World("S0 X"))
    state = world.reset()
    while not world.is_final(state):
        action = random.choice(world.available_actions(state))
        state = world.step(state, action)
    print(state.value)


if __name__ == "__main__":
    main()
