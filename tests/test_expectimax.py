from lle import World, Action
from src.adversarial_search import expectimax
from src.competitive_world import CompetitiveWorld
from .competitive_graph import GraphMDP


def test_raise_value_error():
    mdp = GraphMDP.parse("tests/graphs/vary-depth.graph")
    s0 = mdp.reset()
    s = mdp.step(s0, "Right")
    try:
        expectimax(mdp, s, 2)
        assert False, "Should raise ValueError"
    except ValueError:
        assert True


def test_expectimax_two_agents():
    world = CompetitiveWorld(
        World(
            """
        .  L1W
        S0 S1
        X  X"""
        )
    )
    action = expectimax(world, world.reset(), 5)
    assert action in [Action.SOUTH, Action.STAY]

    world.reset()
    action = expectimax(world, world.reset(), 1)
    assert action == Action.SOUTH


def test_expectimax_greedy():
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
    action = expectimax(world, world.reset(), 1)
    assert action == Action.SOUTH


def test_expectimax_greedy_3steps():
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
    action = expectimax(world, world.reset(), 5)
    assert action in [
        Action.EAST,
        Action.SOUTH,
    ], "When the agent sees 5 steps in the future, it should realise that going east is the best option to get both gems."


def test_expectimax_greedy_5steps():
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
    action = expectimax(world, world.reset(), 10)
    assert (
        action == Action.SOUTH
    ), "When the agent sees 10 steps in the future, it should realise that it should first take the bottom gem before the other two."


def test_two_agents2():
    """In this test, Agent 2 should take the gem on top of him
    in order to prevent Agent 0 from getting it, even if Agent 2
    could deny two gems by going left."""
    world = CompetitiveWorld(
        World(
            """
        . . G G S0
        . @ @ @ G
        . . X X G
        . . G G S1
"""
        )
    )
    action = expectimax(world, world.reset(), 1)
    assert action in [Action.SOUTH, Action.WEST]

    action = expectimax(world, world.reset(), 3)
    assert action == Action.WEST

    action = expectimax(world, world.reset(), 7)
    assert action == Action.SOUTH


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
    action = expectimax(world, world.reset(), 3)
    assert action == Action.SOUTH

    world.reset()
    action = expectimax(world, world.reset(), 2)
    assert action != Action.EAST
