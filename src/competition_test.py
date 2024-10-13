

from adversarial_search import minimax, alpha_beta, expectimax
from competitive_world import CompetitiveWorld, CWorldState
from lle import World, Action
import random
import inspect
import numpy as np
import matplotlib.pyplot as plt



class Competition:

    def __init__(self, mdp: CompetitiveWorld, seed: int = None) -> None:

        if seed: random.seed(seed)
        self.mdp: CompetitiveWorld = mdp
        self.current_state: CWorldState = mdp.reset()
        self.turn: int = 0

    def play_move(self, move: Action) -> None:
        new_state: CWorldState = self.mdp.step(state=self.current_state, action=move)
        self.current_state = new_state
        self.set_cstate(new_state)

    def set_cstate(self, cstate: CWorldState) -> None:
        self.mdp.world.set_state(cstate.world_state)
        self.current_state = cstate

    def aversarial_search(self, algorithm: callable, max_depth: int) -> Action:
        assert callable(algorithm)  # On vérifie que c'est une fonction
        assert {'mdp', 'state', 'max_depth'}.issubset(set(inspect.signature(algorithm).parameters.keys()))  # On vérifie les paramètres
        return algorithm(mdp=self.mdp, state=self.current_state, max_depth=max_depth)

    def get_random_move(self) -> Action:
        possible: list[Action] = self.mdp.available_actions(state=self.current_state)
        if len(possible) > 1:
            possible.remove(Action.STAY)
            return random.choice(possible)  # Si + que stay, on select au pif sans stay
        else: return Action.STAY

    def display_state(self) -> None:
        print(f'\n[T] Turn {self.turn} ({"Opponent" if self.turn % 2 == 1 else "Player"})\n[S] {self.mdp.world.get_state()}, score={self.current_state.value}')

    def swap_players(self) -> None:
        
        str_world: str = self.mdp.world.world_string
        swapped: str = str_world.replace('S0', 'temp').replace('S1', 'S0').replace('temp', 'S1')
        self.mdp = CompetitiveWorld(World(swapped))
        self.set_cstate(self.mdp.reset())

    def run_match(self, algorithm: callable, maxdepth: int, swap_players: bool = False, verbose: bool = True) -> dict:
        
        if swap_players: self.swap_players()
        self.turn = 0

        while not self.mdp.is_final(self.current_state):
            
            if verbose: self.display_state()
            # player turn
            if self.current_state.current_agent == 0:
                move: Action = self.aversarial_search(algorithm=algorithm, max_depth=maxdepth)
                self.play_move(move=move)
                if verbose: print(f'[P] Player played {move}')
            
            # opponent turn
            else:
                move: Action = self.get_random_move()
                self.play_move(move=move)
                if verbose: print(f'[O] Opponent played {move}')

            self.turn += 1
        
        player_score: int = self.current_state.value
        winner: str = "Opponent" if player_score < 0 else ("Player" if player_score > 0 else None)
        if verbose: print(f'\n--- END ---\n[S] Final state value : {player_score} ({self.turn} move(s) played.)\n[W] Winner : {winner}')
        return {'winner': winner, 'score': player_score, 'turns': self.turn}

    def run_simulation(self, it: int, algorithm: callable, maxdepth: int, swap_players: bool = False, verbose: bool = False) -> list:
        
        turns: list[int] = list()
        winners: list[str] = list()
        scores: list[float] = list()

        for i in range(it):
            print(f'[SIM] Simulation n° {i} ...')
            r = self.run_match(algorithm=algorithm, maxdepth=maxdepth, swap_players=swap_players, verbose=verbose)
            self.set_cstate(self.mdp.reset())
            turns.append(r['turns'])
            winners.append(r['winner'])
            scores.append(r['score'])

        res: list = [s for s in (v for v in zip(turns, scores, winners))]
        return self.compute_statistics(res=res, algorithm=algorithm, maxdepth=maxdepth, it=it)

    @staticmethod
    def compute_statistics(res: list[tuple], algorithm: callable, maxdepth: int, it: int) -> None:

        turns, scores, winners = ([t[i] for t in res] for i in range(3))
        
        turns_std: float = float(np.std(turns))
        turns_avg: float = float(np.average(turns))
        min_turns: int = min(turns)
        max_turns: int = max(turns)
        
        winrate: float = winners.count('Player') / len(winners)
        looserate: float = winners.count('Opponent') / len(winners)
        drawrate: float = winners.count(None) / len(winners)
        
        scores_std: float = float(np.std(scores))
        scores_avg: float = float(np.average(scores))
        min_score: float = min(scores)
        max_score: float = max(scores)

        stats: dict = {
            'turns_std': turns_std,
            'turns_avg': turns_avg,
            'min_turns': min_turns,
            'max_turns': max_turns,
            'winrate': winrate,
            'looserate': looserate,
            'drawrate': drawrate,
            'scores_std': scores_std,
            'scores_avg': scores_avg,
            'min_score': min_score,
            'max_score': max_score,
            'algorithm': algorithm.__name__,
            'max_depth': maxdepth,
            'iterations': it
        }

        return {k: round(v, 2) if isinstance(v, float) or isinstance(v, int) else v for k, v in stats.items()}

    @staticmethod
    def plot(data: dict) -> None:

        _, axs = plt.subplots(1, 3, figsize=(16, 5))

        rates: list[float | int] = [data['winrate'], data['looserate'], data['drawrate']]
        labels: list[str] = ['Winrate', 'Looserate', 'Drawrate']
        colors: list[str] = ['#4CAF50', '#F44336', '#2196F3']
        axs[0].bar(labels, rates, color=colors)
        axs[0].set_title('Rates (Win/Loss/Draw)')
        axs[0].set_ylim(0, 1)

        turns_data: list[float | int] = [data['min_turns'], data['turns_avg'], data['max_turns']]
        axs[1].boxplot(turns_data, patch_artist=True, boxprops=dict(facecolor='#FFC107'))
        axs[1].set_title('Turns Statistics')
        axs[1].set_xticks([1], labels=['Turns'])
        axs[1].scatter(1, data['turns_avg'], color='red', zorder=3, label='Average')
        axs[1].legend(loc='upper left')
        axs[1].set_title(f'Turns Statistics\nMin: {data["min_turns"]}, Avg: {data["turns_avg"]}, Max: {data["max_turns"]}')

        scores_data: list[float | int] = [data['min_score'], data['scores_avg'], data['max_score']]
        axs[2].boxplot(scores_data, patch_artist=True, boxprops=dict(facecolor='#03A9F4'))
        axs[2].set_title('Scores Statistics')
        axs[2].set_xticks([1], labels=['Scores'])
        axs[2].scatter(1, data['scores_avg'], color='purple', zorder=3, label='Average Score')
        axs[2].scatter(1, data['min_score'], color='blue', label='Min Score')
        axs[2].scatter(1, data['max_score'], color='green', label='Max Score')
        axs[2].legend(loc='upper left')
        axs[2].set_title(f'Scores Statistics\nMin: {data["min_score"]}, Avg: {data["scores_avg"]}, Max: {data["max_score"]}')

        plt.suptitle(f"Algorithm: {data['algorithm']} | Max Depth: {data['max_depth']} | Iterations: {data['iterations']}", fontsize=16, fontweight='bold')  
        plt.subplots_adjust(top=0.80, wspace=0.1)
        #plt.savefig(f"../{data['algorithm']}depth{data['max_depth']}.png")
        plt.show()




if __name__ == '__main__':

    avantage_w: str = \
    """
    S0 G G G G G G
    G  G @ @ @ G G
    G  @ . . . G G
    G  G G G G X G
    .  . . @ . . .
    S1 @ @ @ . . X

    """

    fiftyfifty: str = \
    """
    G  G . @ G G
    S0 @ . @ . G
    G  . G X . G
    @  . @ @ @ G
    G S1 . G G X    
    """

    bigmaze: str = \
    """
    S0 . G G . G . . . @ @ 
    G  @ @ . . . . . . . . 
    .  . . @ @ @ @ . G G G 
    @  @ G . . . . . . . @ 
    .  . . @ @ @ @ @ @ . . 
    .  G G . . . . . . G . 
    @  @ . @ @ @ @ @ @ @ . 
    .  . . . . . . . . . X 
    G  G @ @ @ @ @ @ @ @ X 
    S1 . . . . . . . . . .
    """

    c = Competition(mdp=CompetitiveWorld(World(fiftyfifty)))
    r = c.run_simulation(it=50, algorithm=alpha_beta, maxdepth=5, swap_players=False, verbose=False)
    c.plot(r)