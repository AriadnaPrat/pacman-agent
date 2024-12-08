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
    def __init__(self, index):
        super().__init__(index)
        self.return_to_base = False  
        self.food_eaten_count = 0  
        

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)

    # Obtener la posición actual
        my_pos = game_state.get_agent_state(self.index).get_position()
        

    # Si debe regresar a la base, calcula la acción para acercarse
        if self.return_to_base:
            best_dist = float('inf')
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist

        # Si ya está en la base, cambia el estado para buscar comida
            if my_pos == self.start:
                self.return_to_base = False
            return best_action

    # Si no está regresando, busca comida
        food_list = self.get_food(game_state).as_list()
        if len(food_list) > 0:
            best_dist = float('inf')
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = min([self.get_maze_distance(pos2, food) for food in food_list])
                if dist < best_dist:
                    best_action = action
                    best_dist = dist

        # Si come una bola de comida, activa el regreso
            successor = self.get_successor(game_state, best_action)
            if successor.get_agent_position(self.index) in food_list:
                if self.food_eaten_count == 0:  
                    self.start = (10, 7)
                elif self.food_eaten_count == 1: 
                    self.start = (25, 10)
                else:
                    self.start = (15, 7)
                self.food_eaten_count += 1  # Incrementar el contador
    #           self.start = (25, 10)  # Cambia la posición de inicio cada vez que come comida 15,7->  22,10   22,9   24,10  25,10
                self.return_to_base = True
            return best_action
      
           
    # Si no hay comida, detente
        return Directions.STOP
    
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
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

    # Característica: Comida restante
        features['successor_score'] = -len(food_list)

    # Característica: Distancia a la comida más cercana
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

    # Nueva característica: Distancia a enemigos visibles
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        visible_ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None]
        if visible_ghosts:
            ghost_distances = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in visible_ghosts]
            features['ghost_distance'] = min(ghost_distances)
        else:
            features['ghost_distance'] = 0

    # Penalización por detenerse
        if action == Directions.STOP:
            features['stop'] = 1

    # Penalización por ir hacia atrás
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

    # Si queda poca comida, priorizar acercarse al inicio (zona segura)
        if food_left <= 5:
            features['distance_to_start'] = self.get_maze_distance(my_pos, self.start)

        return features

    def get_weights(self, game_state, action, food_left):
        if food_left <= 5:
            return {
                'successor_score': 100,
                'distance_to_food': -1,
                'ghost_distance':15,  # Evitar fantasmas
                'distance_to_start': -2,  # Regresar al inicio
                'stop': -100,
                'reverse': -2
            }
        return {
            'successor_score': 100,
            'distance_to_food': -1,
            'ghost_distance': 15,  # Evitar fantasmas
            'stop': -100,
            'reverse': -2
        }

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
        Evaluates an action by combining features and weights.
        This is the implementation of the evaluate method for the defensive agent.
        """
        features = self.get_features(game_state)
        weights = self.get_weights()
        return features * weights

    def get_features(self, game_state):
        """
        Defines the relevant features for the defensive agent.
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
        Assigns weights to the features.
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
        Generates the successor state after taking an action.
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Si no está en un punto exacto de la cuadrícula, intenta generar otro sucesor
            return successor.generate_successor(self.index, action)
        else:
            return successor
