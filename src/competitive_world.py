from dataclasses import dataclass
import lle
from lle import World, Action, WorldEvent, EventType
from competitive_env import CompetitiveEnv, State


@dataclass
class CWorldState(State):
    world_state: lle.WorldState

    def __init__(self, value: float, current_agent: int, world_state: lle.WorldState):
        super().__init__(value, current_agent)
        self.world_state = world_state

    def is_alive(self, agent_id: int):
        return self.world_state.agents_alive[agent_id]


class CompetitiveWorld(CompetitiveEnv[Action, CWorldState]):
    """Competitive World"""

    def __init__(self, world: World):
        super().__init__()
        assert world.n_agents == 2
        self.world = world

    def reset(self):
        self.world.reset()
        return CWorldState(
            value=0,
            current_agent=0,
            world_state=self.world.get_state(),
        )

    def available_actions(self, state: CWorldState):
        if not state.is_alive(state.current_agent):
            return [Action.STAY]
        self.world.set_state(state.world_state)
        return self.world.available_actions()[state.current_agent]

    def is_final(self, state: CWorldState) -> bool:
        self.world.set_state(state.world_state)
        return any(agent.has_arrived for agent in self.world.agents)

    def step(self, state: CWorldState, action: Action):
        self.world.set_state(state.world_state)
        actions = [lle.Action.STAY] * self.world.n_agents
        actions[state.current_agent] = action
        events = self.world.step(actions)
        r = self.reward(events, state.current_agent)
        return CWorldState(
            value=state.value + r,
            current_agent=(state.current_agent + 1) % self.world.n_agents,
            world_state=self.world.get_state(),
        )

    def reward(self, events: list[WorldEvent], current_agent: int):
        r = 0.0
        for e in events:
            if e.event_type == EventType.GEM_COLLECTED:
                r += 1.0
            elif e.event_type == EventType.AGENT_EXIT:
                r += 1.0
            elif e.event_type == EventType.AGENT_DIED:
                r = -1.0
                break
        if current_agent != 0:
            r = -r
        return r


class BetterValueFunction(CompetitiveWorld):
    def transition(self, state: CWorldState, action: Action) -> CWorldState:
        return super().step(state, action)
