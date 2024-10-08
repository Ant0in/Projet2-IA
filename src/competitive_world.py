from dataclasses import dataclass
import lle
from lle import World, Action, WorldEvent, EventType
from competitive_env import CompetitiveEnv, State



@dataclass
class CWorldState(State):

    # Competitive world state
    # contains value of current worldstate + agent state (whose turn)
    world_state: lle.WorldState

    def __init__(self, value: float, current_agent: int, world_state: lle.WorldState) -> None:
        super().__init__(value, current_agent)
        self.world_state: lle.World = world_state

    def is_alive(self, agent_id: int) -> bool:
        return self.world_state.agents_alive[agent_id]
    
    def __hash__(self) -> int:
        # Je réécris __hash__ pour permettre le cache dans les algos.
        return hash(self.current_agent) + hash(self.value) + hash(self.world_state)



class CompetitiveWorld(CompetitiveEnv[Action, CWorldState]):

    # Competitive world

    def __init__(self, world: World) -> None:
        super().__init__()
        assert world.n_agents == 2, f'[E] World must contain 2 agents. (={world.n_agents})'
        self.world = world

    def reset(self) -> CWorldState:
        self.world.reset()
        return CWorldState(
            value=0,
            current_agent=0,
            world_state=self.world.get_state(),
        )

    def available_actions(self, state: CWorldState) -> list[Action]:
        # returns available move (list) in CWorldState 'state'.
        if not state.is_alive(state.current_agent): return [Action.STAY]
        self.world.set_state(state.world_state)
        return self.world.available_actions()[state.current_agent]

    def is_final(self, state: CWorldState) -> bool:
        # returns a bool indicating if one the agents is arrived
        self.world.set_state(state.world_state)
        return any(agent.has_arrived for agent in self.world.agents)

    def step(self, state: CWorldState, action: Action) -> CWorldState:
        
        # returns CWorldState 'state' whereas a 'step' has been made (using action 'action').
        self.world.set_state(state.world_state)

        # fill actions list with STAY, so that the player not playing STAY in place.
        actions: list[Action] = [lle.Action.STAY] * self.world.n_agents
        actions[state.current_agent] = action  # load 'action' for current_player
        events = self.world.step(actions)

        # calculate the reward for given score
        r: float = self.reward(events, state.current_agent)

        return CWorldState(
            value=state.value + r,
            current_agent=(state.current_agent + 1) % self.world.n_agents,
            world_state=self.world.get_state(),
        )
    
    def reward(self, events: list[WorldEvent], current_agent: int):
        
        r: float = 0.0
        
        for e in events:
            if e.event_type == EventType.GEM_COLLECTED:
                r += 1.0
            elif e.event_type == EventType.AGENT_EXIT:
                r += 1.0
            elif e.event_type == EventType.AGENT_DIED:
                r = -1.0
                break

        # adversarial proprety
        if current_agent != 0: r = -r
        return r




class BetterValueFunction(CompetitiveWorld):
    
    def transition(self, state: CWorldState, action: Action) -> CWorldState:
        return super().step(state, action)
