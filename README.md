# Programming Course: Connect 4 Bot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Project Description

This project was completed as part of a python programming course, with the goal of creating a bot that can play the game [Connect 4](https://en.wikipedia.org/wiki/Connect_Four). The main.py file, agents/agent_random, and some skeleton code in agents/common.py were provided by the course instructor, Owen Mackwood. There were two constraints as part of the project:
- Agents had a 1 second time limit in which to make a move
- The main.py file could not be altered, to allow students to test their agents against each other

I also had some guidance by following [this blog](https://towardsdatascience.com/creating-the-perfect-connect-four-ai-bot-c165115557b0) in how to convert the game from numpy arrays to a bitmap representation. The rest of the code, including bitmap operations and manipulations, is written entirely by me. 

The current version of the bot was recently (spontaneously) tested at a friend's wedding, with a record of 15-1 against (slightly) inebriated human competition. To be perfectly honest, I'm not entirely sure the Being that won against the bot it isn't a cyborg, as his trivia record on the night was suspiciously excellent as well. Therefore, I feel safe in saying that the bot currently performs at approximately human level.

One interesting side note: It was funny to watch the people at the wedding personify the program when playing against it. When it made a move that blocked their plan, they would get "upset" and complain about it like it was sitting there playing against them. Furthermore, I ran it with a 3 second time limit on making moves, and they were always surprised at how quickly it played, especially after they had spent much longer thinking about their own move. You could see the speed at which it played back would intimidate them. It was a very interesting psychological experiment, as well as a lot of fun!


## Project Outline

### Setting up the environment

The repo includes an `environment.yml` file, which may be used to create a conda environment containing the packages requiured for the project to run.

To create the conda environment from the provided `environment.yml` file, simply run the follwing command from the directory in which the file is located. For further information, refer to the conda [documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) regarding managing environments.

```
    conda env create -f environment.yml
```

### Agents

#### Random Agent

The random agent does exactly what the name says. It plays a random move. This agent was used simply to test the functions required for game play to function correctly. For example, testing whether an action is a legal move, printing the current game state, or ensuring the action is recorded in the proper column. 

#### Minimax Agent

This was the simplest working agent. It operated by performing a depth-first tree search using the [minimax algorithm](https://en.wikipedia.org/wiki/Minimax) to a given depth, then scoring the game state at each child node using a simple heuristic I defined within the agent. The purpose of this agent was primarily to get a working knowledge of how to recursively code a tree search, before moving on to more effective algorithms.

#### Minimax with Alpha-Beta Pruning

This agent is an extension of the initial minimax agent. [Alpha-beta pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) allows the minimax algorithm to reduce the number of paths it follows down the search tree, by trimming those that cannot be better than an alternate path. This allows the algorithm to reach a greater depth, or the same depth in less time. This was important to the project, as maximizing the algorithm's depth wihtin the time limit significantly improved performance. 

#### Monte-Carlo Tree Search (Current Version)

This agent works on a different principle than the minimax algorithm. Instead of performing an exhaustive depth- or breadth-first search to a given depth, then scoring the game state based on a heuristic, the [Monte-Carlo tree search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) (MCTS) works by choosing a node to expand, then sampling a path along the game tree to completion (win, loss or tie). The result of the game is then passed back up the path, updating the statistics of the expanded node. Nodes are chosen to expand using a method that balances exploitation with exploration. In this case, the Upper Confidence Bound 1 (UCB1) formula is used, with the theoretical exploration parameter value $c=/sqrt{2}$. 

This allows greater control of the time taken, as paths may simply be sampled until the time limit is reached. This is much more efficient than having to set the depth to a value such that the full tree is explored to that depth within the allotted time, which generally results in some unused time remaining. At the end of the time limit, the action chosen is that which corresponds to the best statistics.

An easy way to improve the performance of MCTS is to save the explored paths and their associated statistics, rather than simply updating the statistics of the expanded node. During the course, I ran out of time to include this, but this will be the first update I make to this project once I have time to pick it up again.

### Testing

A major requirement of this project was to create proper unit tests of our functions as we progressed. In fact, the Owen encouraged test-driven design (TDD) throughout the project. To this end, he provided multiple tutorials regarding how to think about TDD and how it can result in stronger projects. The main points I took from these tutorials were:
- TDD forces you to think about use and edge cases before writing the function, which helps avoid getting tunnel vision on particular use cases
- Take advantage of random sampling to avoid hard-coding all test cases, and to make testing more robust
- Write simple functions to test more complex functions
- Utilizing previously tested functions to create more thorough tests for new functions (for example, using numpy array functions to test the bitmap functions during conversion)
- How to use an oracle to test an agent's ability to find guaranteed wins

The oracle: a quick note on this, as I thought it was a super interesting way to test the bot's performance. At the suggestion of Owen, I created an oracle function. This worked by creating a lot of randomly sampled games, until it found a selection of game states with guaranteed wins above a certain depth. I then set the MCTS agent to play against itself, starting from these game states, to see how many of the guaranteed wins it would find. 


## Lessons Learned

This project taught me a lot about the proper organization of a python project. It was my first introduction to a project using a module and package structure. I learned a lot about how to write thorough unit tests, and how to effectively employ random sampling to avoid having to hard code test scenarios and make testing more robust. I would like to thank the course instructor Owen Mackwood for his expert guidance in introducing me to proper python programming practices.

The skeleton code provided included a framework for using numpy arrays throughout the project. However, I found the blog mentioned above, and decided to try implementing the project entirely in bitmap format. There were multiple steps in the algorithmic process at which I could convert the game state from arrays to bitmaps, but I knew the conversion was a significant computation, so the earlier in the process I could perform the operation, the more time I would be saving. However, this involved rewriting the entire program to operate in bitmaps. It was challenging, but also extremely rewarding.

One of the most confusing experiences of this project was watching the performance of my agents improve. At first, they were frustratingly terrible, constantly missing wins and giving away games. For a short time during the middle of the project, the agents were about an even match for me, and there were some close games, where we both made mistakes. Eventually, they turned a corner and became frustrating in an entirely new way. Now the only way I can beat the program is to force it into a situation where it has no choice but to give me the win. I am thoroughly convinced that I now only win out of luck and by avoiding silly mistakes in the early game. 


## Future Work

Although this project has been dormant now for some time, I still hope to pick it up again and train an agent using reinforcement learning. Hopefully that is a project I can undertake in the coming year.
