import numpy as np
from typing import Optional, Tuple
from gmpy2 import popcount
from agents.common import Board, BoardPiece, Bitmap, PlayerAction, \
    SavedState, NO_PLAYER, GameState, board_to_bitmap, check_end_state, \
    apply_action_cp, check_top_row, valid_actions


# class Tree(object):
#     "Generic tree node."
#
#     def __init__(self, name='root', children=None):
#         self.name = name
#         self.children = []
#         if children is not None:
#             for child in children:
#                 self.add_child(child)
#
#     def __repr__(self):
#         return self.name
#
#     def add_child(self, node):
#         assert isinstance(node, Tree)
#         self.children.append(node)


class Connect4Node:
    """ Class that stores the game state and statistics for that state

    Attributes
        board : Bitmap
            Represents the positions of the current player on the game board.
        mask : Bitmap
            Represents the positions of both players on the game board.
        shape : tuple
            The shape of the game board.
        node_col : int
            The column which was played to produce this game state from the
             previous game state.
        max_player : boolean
            Indicates whether the current node is maximizing player node.
        actions : list
            The legal moves available from this game state. The list has been
             randomized so that new children can be created in the sequence of
             these actions.
        children : list[Connect4Node]
            Contains the children of the current node.
        c : float
            Exploration parameter used in UCB1 calculations
        si : int
            Number of simulations node is involved in
        wi : int
            Number of simulation wins node is involved in

    Methods
        make_move()
            Determines next child to visit in the game tree.

    """

    def __init__(self, board: Bitmap, mask: Bitmap, board_shp: Tuple,
                 node_col: PlayerAction, max_player: bool):
        """
        Parameters
            board = bitmap representing positions of current player
            mask = bitmap representing positions of both players
            board_shp = tuple giving the shape of the board (rows, columns)
            node_col = the column played to create this node
            max_player = indicates whether current player is the max player
        """

        # TODO: Should store some variable that indicates whether node is a
        #  terminal state?
        # TODO: Find a way to neatly index the children, without creating all
        #  children when a new node is expanded
        # TODO: Don't need to save the game state variables if I just pass a
        #  set of moves to each node during selection, expansion and simulation
        #  phases of MCTS (i.e. only save stats and children that way)
        # TODO: Make UCB1 a function that determines which child to visit
        # TODO: Make a root class that inherits this class. Root class should
        #  initialize with all children and be the only one that cares which
        #  child is associated with which action.
        # TODO: Make all other nodes agnostic to what actions are associated
        #  with which children. In this case, the children store the game state

        # Update the game state and save game state attributes
        self.board, self.mask = apply_action_cp(board, mask, node_col,
                                                board_shp[0])
        self.shape = board_shp
        self.node_col = node_col
        self.max_player = max_player

        # Node attributes
        # self.actions = valid_actions(mask, board_shp)
        all_act = valid_actions(mask, board_shp)
        # Randomize the order of actions (i.e. the order of node creation)
        self.actions = np.random.choice(all_act, len(all_act), replace=False)
        self.children = []
        self.children_ucb1 = []
        # for col in range(self.shape[1]):
        #     if check_top_row(self.mask, PlayerAction(col), self.shape):
        #         self.actions.append(col)

        # Upper Confidence Bound 1 (UCB1) attributes
        self.c = np.sqrt(2)
        self.si = 0
        self.wi = 0
        # self.UCB1 = self.wi / self.si + self.c * np.sqrt(np.log(sp) / self.si)

    def add_child(self, node):
        """ Adds a child to the given node

        Most commonly called by the MCTS tree at the end of the selection
        phase. Also called by the root node when filling out its children.

        Parameters
            node = the Connect4Node to be added to the game tree
        """

        # Ensure node is an instance of the proper class, then add as a child
        assert isinstance(node, Connect4Node)
        self.children.append(node)

    def next_action(self, node):
        """ Take an action determined by the UCB1 criteria

        This function is called recursively during the selection phase of MCTS.
        Recursion ceases once it reaches a node with unexpanded children. At
        this point, a new child is created from the node's list of actions, and
        the remainder of the game is simulated. The stats are then updated and
        propagated up to the root node, which made the original call.

        Parameters
            node = node selected by root node or previous select_action call
        """

        # If any children are unexpanded, expand them and run a simulation
        # TODO: make sure node and self are used in the correct places
        # TODO: figure out more descriptive name for "node" variable
        if len(node.children) < len(node.actions):
            # Select the next randomized action in the list
            action = PlayerAction(node.actions[len(node.children)])
            # Add the new child to the node
            new_child = Connect4Node(node.board, node.mask, node.shape,
                                     action, not node.max_player)
            node.add_child(new_child)
            # Simulate the game to completion
            winner = node.sim_game(new_child)
        # Else, continue tree traversal
        else:
            next_node_ind = node.ucb1_select()
            winner = self.next_action(self.children[next_node_ind])

        # Update new child's stats based on the result of a simulation
        node.update_stats(winner)
        # self.update_stats(node, winner)
        # if winner == -1:
        #     pass
        # elif winner ^ node.max_player:
        #     node.wi += 1
        # node.si += 1

        return winner

    def sim_game(self, new_node):
        """ Simulates one iteration of a game from the current game state

        Actions are chosen randomly by recursively calling sim_actions until a
        terminal state is reached. At this point, the result is passed back
        to new_node and propagated back up the tree to the root, updating
        the stats along the way.

        Parameters
            new_node = node created during expansion phase of MCTS
        """

        # Simulate a game recursively
        winner = self.sim_actions(new_node.board, new_node.mask,
                                  new_node.max_player)
        # Update the stats of the new_node
        self.update_stats(new_node, winner)
        # if winner == -1:
        #     pass
        # elif winner ^ new_node.max_player:
        #     new_node.wi += 1
        # new_node.si += 1

        return winner

    def sim_actions(self, board: Bitmap, mask: Bitmap, max_player: bool):
        """ Randomly selects and applies actions to simulate game

        This function applies random actions until the game reaches a terminal
        state, either a win or a draw. It then returns the value associated
        with this state. Before returning, it updates the stats of new_node.

        Parameters
            board = bitmap representing positions of current player
            mask = bitmap representing positions of both players
            max_player = indicates whether current player is the max player

        Returns
            True if max_player wins
            False if min_player wins
            -1 if the result is a draw
        """

        # Randomly choose a valid action
        action = -1
        invalid_action = True
        while invalid_action:
            action = np.random.choice(self.shape[1])
            invalid_action = check_top_row(mask, action, self.shape)

        # Simulate the next move
        new_board, new_mask = apply_action_cp(board, mask, action,
                                              self.shape[0])
        # Update the max_player boolean
        max_player = not max_player
        # Check the game state after the new action is applied
        game_state = check_end_state(new_board, new_mask, self.shape)
        if game_state == GameState.IS_WIN:
            if max_player:
                return True
            else:
                return False
        elif game_state == GameState.IS_DRAW:
            return -1
        elif game_state == GameState.STILL_PLAYING:
            return self.sim_actions(new_board, new_mask, max_player)
        else:
            print('Error in Simulation')

    def update_stats(self, node, winner):
        """ Updates the MCTS stats for a given node

        """

        if winner == -1:
            pass
        elif winner ^ node.max_player:
            node.wi += 1
        node.si += 1

    def ucb1_select(self):
        """ Determines which node to select during selection phase

        Parameters
        """

        # TODO: make sure this function works properly
        max_ucb1 = 0
        select_node = []
        # Calculate UCB1 for each child and select child with largest
        # for ind, child in enumerate(self.children):
        #     child_ucb1 = self.calc_ucb1(child.wi, child.si, self.si)
        #     if child_ucb1 > max_ucb1:
        #         max_ucb1 = child_ucb1
        #         select_node = [ind]
        #     elif child_ucb1 == max_ucb1:
        #         select_node.append(ind)
        # Implementation where UCB1's are stored as an attribute list
        for ind, child in enumerate(self.children):
            if child.ucb1 > max_ucb1:
                max_ucb1 = child.ucb1
                select_node = [ind]
            elif child.ucb1 == max_ucb1:
                select_node.append(ind)

        return np.random.choice(select_node)

    def calc_ucb1(self, wi, si, sp):
        """ Calculates the Upper Confidence Bound 1 score for a given node

        Parameters
            wi = number of wins for current node
            si = number of simulations for current node
            sp = number of simulations for parent node
        """

        return wi / si + self.c * np.sqrt(np.log(sp) / si)


class Connect4Root(Connect4Node):
    """ Class that inherits the node class to create a root node.

    The root node should initialize with all of its children because this is
    the only node that is matters which child is associated with which action.
    Therefore, being able to reference the child with the unexpanded list.

    """

    def __init__(self, board: Bitmap, mask: Bitmap, board_shp: Tuple,
                 node_col: int, max_player: bool):
        super().__init__(board, mask, board_shp, node_col, max_player)
        # self.children = []
        # self.unexpanded = []
        # for col in range(self.shape[1]):
        #     if check_top_row(self.mask, PlayerAction(col), self.shape):
        #         self.unexpanded.append(col)
        #         self.children.append(self.add_child())
        # self.actions = valid_actions(mask, board_shp)
        for action in self.actions:
            self.add_child(Connect4Node(board, mask, board_shp, action,
                                        not max_player))

    def mcts(self):
        """ Top-level function in MCTS 4-phase process.

        Find a node to expand.
        Uses UCB1 to select nodes beginning at the root node. Once a node with
        unexplored children is reached, simulate a game.
        If the child exists, make_move() is called recursively until it reaches
        a node where the selected child node does not exist. In this case, it
        creates a new child and simulates a game. It then returns the result of
        the simulated game to update the node statistics.

        """

        # # If the node has no children, create one and simulate a game
        # if not self.children:
        #     action = PlayerAction(self.actions[0])
        #     self.add_child(Connect4Node(self.board, self.mask, self.shape,
        #                                 action, not self.max_player, self.sp))
        #     return self.sim_game()
        # # If node has any unexpanded children, create one and simulate a game
        # elif len(self.children) < len(self.actions):
        # if len(self.children) < len(self.actions):
        #     action = PlayerAction(self.actions[len(self.children)])
        #     self.add_child(Connect4Node(self.board, self.mask, self.shape,
        #                                 action, not self.max_player, self.sp))
        #     return self.sim_game()

        n_iters = 100
        for itr in range(n_iters):
            action = self.ucb1_select()
            winner = self.next_action(self.children[action])
            self.update_stats(self, winner)
            # if winner:
            #     self.wi += 1
            # self.si += 1

        # most_si = 0
        # best_action = []
        # for action, child in enumerate(self.children):
        #     if child.si > most_si:
        #         most_si = child.si
        #         best_action = [action]
        #     if child.si == most_si:
        #         best_action.append(action)
        #
        # return np.random.choice(best_action)
        return self.ucb1_select()


def generate_move_mcts(board: Board, player: BoardPiece,
                       saved_state: Optional[SavedState]
                       ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Agent selects a move based on a minimax depth first search, with
    alpha-beta pruning.

    :param board: 2d array representing current state of the game
    :param player: the player who made the last move (active player)
    :param saved_state: ???

    :return: the agent's selected move
    """

    # If the board is empty, play in the center column
    if np.all(board == NO_PLAYER):
        action = np.floor(np.median(np.arange(board.shape[1])))
        return PlayerAction(action), saved_state

    # Calculate the board shape
    bd_shp = board.shape
    # Convert the board to bitmaps and define the min_player board
    max_board, mask_board = board_to_bitmap(board, player)
    # Create a root node
    root_mcts = Connect4Root(max_board, mask_board, bd_shp, 0, True)
    # Call MCTS
    action = root_mcts.mcts()

    return PlayerAction(action), saved_state
