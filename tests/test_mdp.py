from lle import World, Action
from src.competitive_world import CompetitiveWorld


def test_state_turn():
    world = CompetitiveWorld(
        World("""S0 . G
                 S1 X X""")
    )
    s = world.reset()
    assert s.current_agent == 0
    assert s.value == 0
    assert not world.is_final(s)

    actions = [Action.EAST, Action.STAY, Action.EAST, Action.STAY, Action.SOUTH]
    scores = [0, 0, 1, 1, 2]
    assert len(actions) == len(scores), "The test is not well defined."
    for i, (action, score) in enumerate(zip(actions, scores)):
        assert not world.is_final(s)
        s = world.step(s, action)
        turn = i + 1
        assert s.current_agent == turn % 2
        assert s.value == score
    assert world.is_final(s)

    s = world.reset()
    assert s.current_agent == 0
    assert s.value == 0
    assert not world.is_final(s)

    actions = [Action.EAST, Action.STAY, Action.EAST, Action.EAST]
    scores = [0, 0, 1, 0]
    assert len(actions) == len(scores), "The test is not well defined."
    for i, (action, score) in enumerate(zip(actions, scores)):
        assert not world.is_final(s)
        s = world.step(s, action)
        turn = i + 1
        assert s.current_agent == turn % world.world.n_agents
        assert s.value == score
    assert world.is_final(s)


def test_state_is_final():
    world = CompetitiveWorld(World("S0 X S1 X"))
    s = world.reset()
    assert not world.is_final(s)
    s = world.step(s, Action.EAST)
    assert world.is_final(s)


def test_score_death():
    world = CompetitiveWorld(
        World(
            """
    S0  .  X
    S1 L1N X
"""
        )
    )
    s = world.reset()
    assert s.value == 0
    s = world.step(s, Action.EAST)
    assert s.value == -1

    world = CompetitiveWorld(
        World(
            """
    S1  .  X
    S0 L0N X
"""
        )
    )
    s = world.reset()
    assert s.value == 0
    s = world.step(s, Action.STAY)
    assert s.value == 0
    s = world.step(s, Action.EAST)
    assert s.value == 1


def test_score_overwritten():
    world = CompetitiveWorld(
        World(
            """
    S0 G  .  X
    S1 . L1N X
"""
        )
    )
    s = world.reset()
    assert s.value == 0

    # Agent 0
    s = world.step(s, Action.EAST)
    assert s.value == 1

    # Agent 1
    s = world.step(s, Action.STAY)

    # Agent 0
    s = world.step(s, Action.EAST)
    assert s.value == 0, "When Agent 0 dies, the score should be decreased by 1."


def test_available_actions():
    world = CompetitiveWorld(
        World(
            """
    .  S0 G
    S1 X  X
"""
        )
    )
    s = world.reset()
    available = world.available_actions(s)
    expected = [Action.EAST, Action.STAY, Action.WEST, Action.SOUTH]
    assert all(a in available for a in expected)

    s = world.step(s, Action.STAY)
    available = world.available_actions(s)
    expected = [Action.EAST, Action.NORTH, Action.EAST]
    assert all(a in available for a in expected)


def test_available_actins_when_dead():
    world = CompetitiveWorld(
        World(
            """
    S0 V X
    S1 . X
"""
        )
    )
    s = world.reset()
    assert len(world.available_actions(s)) == 2
    s = world.step(s, Action.STAY)
    assert len(world.available_actions(s)) == 2
    s = world.step(s, Action.STAY)
    # Kill agent 0
    s = world.step(s, Action.EAST)
    s = world.step(s, Action.STAY)
    available = world.available_actions(s)
    assert len(available) == 1
    assert available[0] == Action.STAY
