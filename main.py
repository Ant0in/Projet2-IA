
import argparse
from src import expectimax, minimax, alpha_beta, Competition, CompetitiveWorld
from lle import World


ALGORITHMS: dict = {
    'minimax': minimax,
    'alphabeta': alpha_beta,
    'expectimax': expectimax
}

def main():

    parser = argparse.ArgumentParser(description='Play an agent in a world using different algorithms.')
    parser.add_argument('world_file', type=str, help='Path to the text file containing the world representation.')
    parser.add_argument('--algo', required=True, choices=ALGORITHMS.keys(), help='Algorithm to use (minimax, alphabeta, expectimax).')
    parser.add_argument('--maxdepth', required=True, type=int, help='Maximum search depth for adversarial search.')
    
    args = parser.parse_args()

    with open(args.world_file, 'r') as f:
        str_world = f.read()

    world: World = World(str_world)
    c: Competition = Competition(mdp=CompetitiveWorld(world))
    c.run_match(algorithm=ALGORITHMS[args.algo], maxdepth=args.maxdepth, swap_players=False, verbose=True)


if __name__ == '__main__':
    main()
