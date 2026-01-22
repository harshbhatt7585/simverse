# Farmtila

Farmtila is an grid based RL ENV of multi-agents. It consist of large land which can be harvested. The main goal of the agent is to harvest as much as land using seeds, the winner will have the most harvested land. 

In the start of the simulation, seeds will be distributed randomly all over the land, agent has to find and aquire seeds 
so they can harvet the land. One seed can harvest one box of the grid. Seeds will again put randomly over than land after every 
(X) number of total agents steps. 


## Action Space

Discrete (6)
0 - Move up
1 - Move down
2 - Move left
3 - Move right
4 - pickup the seed
5 - harvest the land
