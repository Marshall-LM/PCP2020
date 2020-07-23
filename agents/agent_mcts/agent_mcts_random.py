import numpy as np
from typing import Optional, Tuple, List
from time import time
from agents.common import Board, BoardPiece, Bitmap, PlayerAction, \
    SavedState, NO_PLAYER, GameState, board_to_bitmap, check_end_state, \
    apply_action_cp, valid_actions


# Declare a global constant for calculating the UCB1 score
C = np.sqrt(2)


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

    # TODO: return chosen action subtree using saved_state, to improve
    #  performance
    # Calculate the board shape
    bd_shp = board.shape
    # If the board is empty, play in the center column
    if np.all(board == NO_PLAYER):
        action = np.floor(np.median(np.arange(bd_shp[1])))
        return PlayerAction(action), saved_state

    # Convert the board to bitmaps and define the min_player board
    max_board, mask_board = board_to_bitmap(board, player)
    # Create a root node
    root_mcts = Connect4Node(max_board, mask_board, bd_shp, -1, True)
    # Call MCTS
    action = mcts(root_mcts)

    return PlayerAction(action), saved_state


def mcts(root_node):
    """ Top-level function in MCTS 4-phase process.

    Find a node to expand.
    Uses UCB1 to select nodes beginning at the root node. Once a node with
    unexplored children is reached, simulate a game.
    If the child exists, make_move() is called recursively until it reaches
    a node where the selected child node does not exist. In this case, it
    creates a new child and simulates a game. It then returns the result of
    the simulated game to update the node statistics.

    """

    # TODO: return chosen action subtree using saved_state, to improve
    #  performance
    start_time = time()
    curr_time = time()
    while (curr_time - start_time) < 1.0:
        max_win = root_node.traverse()
        if max_win:
            root_node.wi += 1
        root_node.si += 1
        curr_time = time()

    most_si = 0
    best_action = []
    for ind, child in enumerate(root_node.children):
        if child.si > most_si:
            most_si = child.si
            best_action = [child.node_col]
        if child.si == most_si:
            best_action.append(child.node_col)

    return np.random.choice(best_action)


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
        state : GamesState
            The current game state of the game (win, draw or still playing)
        actions : list
            The legal moves available from this game state. The list has been
             randomized so that new children can be created in the sequence of
             these actions.
        children : list[Connect4Node]
            Contains the children of the current node.
        si : int
            Number of simulations node is involved in
        wi : int
            Number of simulation wins node is involved in

    Methods
        add_child()
            Adds a Connect4Node to the current node's list of children
        traverse()
            Searches the tree until a node with unexpanded children is found
        sim_game()
            Simulates a game to completion beginning with current node's state
        update_stats()
            Updates the MCTS stats for a given node
        ucb1_select()
            Determines which child node to select using UCB1 criteria
        ucb1_calc()
            Calculates the UCB1 value for a particular child node

    """

    def __init__(self, board: Bitmap, mask: Bitmap, board_shp: Tuple,
                 node_col: PlayerAction, max_player: bool):
        """
        Parameters
            board = bitmap representing positions of current player
            mask = bitmap representing positions of both players
            board_shp = tuple giving the shape of the board (rows, columns)
            node_col = the column played to create this node (game state)
            max_player = indicates whether current player is the max player
        """

        # Update the game state and save game state attributes
        self.board, self.mask = board, mask
        self.shape: Tuple = board_shp
        self.node_col: int = node_col
        self.max_player: bool = max_player
        self.state = check_end_state(self.board, self.mask, self.shape)

        # Node attributes
        # Randomize the order of actions (i.e. the order of node creation)
        self.actions: List[int] = valid_actions(self.mask, board_shp)
        np.random.shuffle(self.actions)
        self.children: List[Connect4Node] = []
        self.children_ucb1: List[float] = []

        # Upper Confidence Bound 1 (UCB1) attributes
        self.si: float = 0
        self.wi: float = 0

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

    def traverse(self):
        """ Searches the tree until a node with unexpanded children is found

        This function is called recursively during the selection phase of MCTS.
        Recursion ceases once it reaches a node with unexpanded children. At
        this point, a new child is created from the node's list of actions, and
        the remainder of the game is simulated. The stats are then updated and
        propagated up to the root node, which made the original call.

        Parameters
            node = node selected by root node or previous select_action call
        """

        # Check whether the current node is a terminal state
        if self.state == GameState.IS_WIN:
            if self.max_player:
                return True
            else:
                return False
        elif self.state == GameState.IS_DRAW:
            return -1

        # If any children are unexpanded, expand them and run a simulation
        if len(self.children) < len(self.actions):
            # Select the next randomized action in the list
            action = PlayerAction(self.actions[len(self.children)])
            # Apply the action to the current board
            child_bd, child_msk = apply_action_cp(self.board, self.mask,
                                                  action, self.shape)
            # Add the new child to the node
            new_child = Connect4Node(child_bd, child_msk, self.shape,
                                     action, not self.max_player)
            # If the game does not end, continue building the tree
            self.add_child(new_child)
            # Simulate the game to completion
            max_win = new_child.sim_game()
            # Update the child's stats
            new_child.update_stats(max_win)
        # Else, continue tree traversal
        else:
            next_node_ind = self.ucb1_select()
            next_child = self.children[next_node_ind]
            max_win = next_child.traverse()

        # Update new child's stats based on the result of a simulation
        self.update_stats(max_win)

        return max_win

    def sim_game(self):
        """ Simulates one iteration of a game from the current game state

        This function applies random actions until the game reaches a terminal
        state, either a win or a draw. It then returns the value associated
        with this state, which is propagated back up the tree to the root,
        updating the stats along the way.

        Returns
            True if max_player wins
            False if min_player wins
            -1 if the result is a draw
        """

        # Randomly choose a valid action until the game ends
        sim_board, sim_mask = self.board, self.mask
        game_state = check_end_state(sim_board, sim_mask, self.shape)
        curr_max_p = self.max_player
        while game_state == GameState.STILL_PLAYING:
            # Randomly select an action
            action = np.random.choice(valid_actions(sim_mask, self.shape))
            # Apply the action to the board
            sim_board, sim_mask = apply_action_cp(sim_board, sim_mask, action,
                                                  self.shape)
            # Update the max_player boolean
            curr_max_p = not curr_max_p
            # Check the game state after the new action is applied
            game_state = check_end_state(sim_board, sim_mask, self.shape)

        if game_state == GameState.IS_WIN:
            # TODO: possibly change how the score calculation works
            #  (i.e. return integers here instead of booleans)
            if curr_max_p:
                return True
            else:
                return False
        elif game_state == GameState.IS_DRAW:
            return -1
        else:
            print('Error in Simulation')

    def update_stats(self, max_win):
        """ Updates the MCTS stats for a given node

        Parameters
            max_win = value indicating whether the simulated game's winner
                      was max_player, or whether the game was a draw
        """

        # Set score values
        win_score = 1.0
        draw_score = 0.5

        if max_win == -1:
            self.wi += draw_score
        elif max_win != self.max_player:
            self.wi += win_score
        self.si += 1

    def ucb1_select(self):
        """ Determines which child node to select using UCB1 criteria """

        max_ucb1 = 0
        select_node = []
        # Calculate UCB1 for each child and select child with largest
        for ind, child in enumerate(self.children):
            child_ucb1 = self.calc_ucb1(child.wi, child.si)
            if child_ucb1 > max_ucb1:
                max_ucb1 = child_ucb1
                select_node = [ind]
            elif child_ucb1 == max_ucb1:
                select_node.append(ind)

        return np.random.choice(select_node)

    def calc_ucb1(self, wi, si):
        """ Calculates the Upper Confidence Bound 1 score for a given node

        Parameters
            wi = number of wins for child node
            si = number of simulations for child node
        """

        return wi / si + C * np.sqrt(np.log(self.si) / si)
