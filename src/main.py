

from graph_parser import GraphMDP
from adversarial_search import minimax
from competitive_env import CompetitiveEnv, State, S, A
from competitive_world import CompetitiveWorld


if __name__ == '__main__':

    mdp = GraphMDP.parse(r"C:\Users\antoi\Desktop\Projet2-IA\tests\graphs\vary-depth.graph")
    
    print(minimax(mdp=mdp, state=mdp.reset(), max_depth=2))