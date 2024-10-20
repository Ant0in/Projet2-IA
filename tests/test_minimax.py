from lle import World, Action
from src.adversarial_search import minimax
from src.competitive_world import CompetitiveWorld
from .competitive_graph import GraphMDP


def test_raise_value_error():
    mdp = GraphMDP.parse("tests/graphs/vary-depth.graph")
    s0 = mdp.reset()
    s = mdp.step(s0, "Right")
    try:
        minimax(mdp, s, 2)
        assert False, "Should raise ValueError"
    except ValueError:
        assert True


def test_graph_mdp():
    mdp = GraphMDP.parse("tests/graphs/vary-depth.graph")
    assert minimax(mdp, mdp.reset(), 1) == "Right"
    assert minimax(mdp, mdp.reset(), 2) == "Left"
    assert minimax(mdp, mdp.reset(), 3) == "Right"


def test_minimax_two_agents():
    world = CompetitiveWorld(
        World(
            """
        .  L1W
        S0 S1
        X  X"""
        )
    )
    world.reset()
    action = minimax(world, world.reset(), 5)
    assert action in [Action.SOUTH, Action.STAY]

    world.reset()
    action = minimax(world, world.reset(), 1)
    assert action == Action.SOUTH


def test_minimax_greedy():
    world = CompetitiveWorld(
        World(
            """
S0 . G G
G  @ @ @
.  . X X
S1 . . .
"""
        )
    )
    action = minimax(world, world.reset(), 1)
    assert action == Action.SOUTH


def test_minimax_greedy_3steps():
    world = CompetitiveWorld(
        World(
            """
S0 . G G
G  @ @ @
.  . X X
.  . . .
S1 . . .
"""
        )
    )
    action = minimax(world, world.reset(), 5)
    assert action in [
        Action.EAST,
        Action.SOUTH,
    ], "When the agent sees 5 steps in the future, it should realise that going east is the best option to get both gems."


def test_minimax_greedy_5steps():
    world = CompetitiveWorld(
        World(
            """
S0 . G G . . .
G  @ @ @ . . X
.  . . . . . X
S1 . . . . . .
"""
        )
    )
    action = minimax(world, world.reset(), 10)
    assert (
        action == Action.SOUTH
    ), "When the agent sees 10 steps in the future, it should realise that it should first take the bottom gem before the other two."


def test_two_agents_laser():
    world = CompetitiveWorld(
        World(
            """
        S0 G  .  X
        .  .  .  .
        X L1N S1 .
"""
        )
    )
    action = minimax(world, world.reset(), 3)
    assert action == Action.SOUTH

    action = minimax(world, world.reset(), 2)
    assert action != Action.EAST
