from lle import World, Action
from src.adversarial_search import alpha_beta
from src.competitive_world import CompetitiveWorld
from .competitive_graph import GraphMDP


def test_raise_value_error():
    mdp = GraphMDP.parse("tests/graphs/vary-depth.graph")
    s0 = mdp.reset()
    s = mdp.step(s0, "Right")
    try:
        alpha_beta(mdp, s, 2)
        assert False, "Should raise ValueError"
    except ValueError:
        assert True


def test_alpha_beta_graph_mdp():
    mdp = GraphMDP.parse("tests/graphs/vary-depth.graph")
    assert alpha_beta(mdp, mdp.reset(), 1) == "Right"
    assert mdp.nodes_expanded == 2

    assert alpha_beta(mdp, mdp.reset(), 2) == "Left"
    assert mdp.nodes_expanded == 5

    assert alpha_beta(mdp, mdp.reset(), 3) == "Right"
    assert mdp.nodes_expanded == 9


def test_alpha_beta_two_agents():
    world = CompetitiveWorld(
        World(
            """
        .  L1W
        S0 S1
        X  X"""
        )
    )
    action = alpha_beta(world, world.reset(), 5)
    assert action in [Action.SOUTH, Action.STAY]

    action = alpha_beta(world, world.reset(), 1)
    assert action == Action.SOUTH


def test_alpha_beta_greedy():
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
    action = alpha_beta(world, world.reset(), 1)
    assert action == Action.SOUTH


def test_alpha_beta_greedy_3steps():
    world = CompetitiveWorld(
        World(
            """
S0 G X .
G  @ @ @
.  . . X
S1 . . .
"""
        )
    )
    action = alpha_beta(world, world.reset(), 3)
    assert action in [Action.EAST, Action.SOUTH]


def test_alpha_beta_greedy_5steps():
    world = CompetitiveWorld(
        World(
            """
S0 G X .
G  @ @ @
.  . . X
S1 . . .
"""
        )
    )
    action = alpha_beta(world, world.reset(), 10)
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
    action = alpha_beta(world, world.reset(), 3)
    assert action == Action.SOUTH

    world.reset()
    action = alpha_beta(world, world.reset(), 2)
    assert action != Action.EAST
