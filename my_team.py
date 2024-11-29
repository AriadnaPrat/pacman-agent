# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point

import math

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########
class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions.
    """

    def __init__(self, index):
        super().__init__(index)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        super().register_initial_state(game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action, food_left):
        return util.Counter()

    def get_weights(self, game_state, action):
        return {}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        food_left = len(self.get_food(game_state).as_list())
       
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a, food_left) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

    
        if food_left<= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)
    
    def evaluate(self, game_state, action, food_left):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action, food_left)
        weights = self.get_weights(game_state, action, food_left)
        return features * weights


    def get_features(self, game_state, action, food_left):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
            ###########
           
            ###########
            my_state = successor.get_agent_state(self.index)
            my_pos = my_state.get_position()
            if food_left <= 19:
                
         #       print("\n  1  food_left=", food_left, "_old=",self.food_left_old)
                features = util.Counter()
                successor = self.get_successor(game_state, action)

                # Computes distance to invaders we can see
                enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
                invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
                features['num_invaders'] = len(invaders)
                if len(invaders) > 0:
                    dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                    features['invader_distance'] = min(dists)
                
                if action == Directions.STOP: 
                    features['stop'] = 1
                rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
                print("\n  1.1  action=", action)
                if action == rev: 
                    features['reverse'] = 1
                    print("\n  1.2  action=", action)

        return features

    def get_weights(self, game_state, action, food_left):
        if food_left<= 19:
        #    print("\n  2  food_left=", food_left, "_old=",self.food_left_old)
            return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
        
        return {'successor_score': 100, 'distance_to_food': -1}

#alpha-beta no funciona bien
class DefensiveReflexAgent(CaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free using Alpha-Beta pruning.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.previous_positions = []  # Asegúrate de definir el atributo
        self.target_food = None
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Selecciona la mejor acción usando una combinación de heurísticas.
        """
        actions = game_state.get_legal_actions(self.index)
        
        # Si no hay acciones legales, devuelve STOP
        if not actions:
            return Directions.STOP

        # Evita quedarse quieto
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        # Generar sucesores para cada acción
        successors = []
        for action in actions:
            try:
                successor = self.get_successor(game_state, action)
                successors.append((action, successor))
            except Exception as e:
                print(f"Error al generar sucesor para la acción {action}: {e}")
                continue  # Si ocurre un error, omite esta acción

        # Evita repetir movimientos para no quedarse atascado
        current_pos = game_state.get_agent_state(self.index).get_position()
        self.previous_positions.append(current_pos)
        if len(self.previous_positions) > 6:
            self.previous_positions.pop(0)

        # Evaluar las acciones y elegir la mejor
        best_action = None
        best_value = -float('inf')
        for action, successor in successors:
            value = self.evaluate(successor, action)  # Aquí es donde se usa evaluate
            next_pos = successor.get_agent_state(self.index).get_position()
            if next_pos in self.previous_positions:
                value -= 50  # Penaliza quedarse en un ciclo
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def evaluate(self, game_state, action):
        """
        Evalúa una acción combinando características y pesos.
        Esta es la implementación del método evaluate para el agente defensivo.
        """
        features = self.get_features(game_state)
        weights = self.get_weights()
        return features * weights

    def get_features(self, game_state):
        """
        Define las características relevantes para el agente defensivo.
        """
        features = util.Counter()
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Característica: el agente está en defensa
        features['on_defense'] = 1 if not my_state.is_pacman else 0

        # Característica: número de invasores
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # Característica: distancia a los invasores
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # Penaliza detenerse
        if game_state.get_legal_actions(self.index):
            features['stop'] = 1

        # Penaliza si el agente invierte la dirección
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if game_state.get_agent_state(self.index).configuration.direction == rev:
            features['reverse'] = 1

        return features

    def get_weights(self):
        """
        Asigna pesos a las características.
        """
        return {
            'num_invaders': -1000,  # Penaliza los invasores
            'on_defense': 100,      # Prefiere estar en defensa
            'invader_distance': -10,  # Prefiere estar lejos de los invasores
            'stop': -100,            # Penaliza detenerse
            'reverse': -2            # Penaliza invertir la dirección
        }

    def get_successor(self, game_state, action):
        """
        Genera el estado sucesor después de realizar una acción.
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Si no está en un punto exacto de la cuadrícula, intenta generar otro sucesor
            return successor.generate_successor(self.index, action)
        else:
            return successor
