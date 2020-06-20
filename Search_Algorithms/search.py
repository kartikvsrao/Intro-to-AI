# search.py
# ---------
#
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
#
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)
#

###################################

"""
 Project 1 : Search Algorithms
 Team Members : Kartik Rao
		Saral Anand

"""

###################################
"""
  Code Based on:
  Author: Ioannis Karamouzas
  Date of Retrieval: 09/15/2019
  Availibility: https://89fa4bd4-a-1e6e9713-s-sites.googlegroups.com/a/g.clemson.edu/cpsc-ai/schedule/lec04.pdf?attachauth=ANoY7cq4bmGQsYIulNLZvZxka5zCruNV9z0Px1mBndHBu8evrdxX8JM4WCMP--Vh4kN-vY9EVqYvZQBynW7eehaKy7g7HNe-P2rJaITYUPO6aOu8DTfnsLzw3RXf-pF6L2NwPfWTANjxeQw0_DiD90yQx8JdHB2JfbAml5XGahOiihHZEKPcCXL-ywHNPgzY4C1-X_sYo9pJbm1QLlU0sTbDu2WAOyxgOQ%3D%3D&attredirects=0
"""
###################################
"""
  Code Based on:
  Author: Ioannis Karamouzas
  Date of Retrieval: 09/17/2019
  Availibility: https://89fa4bd4-a-1e6e9713-s-sites.googlegroups.com/a/g.clemson.edu/cpsc-ai/schedule/lec05.pdf?attachauth=ANoY7cpjcxMzrtkbL6sPKEMo0MJ1Ylhe9HaZ_P_gmiTAdttcTwJ9kNNrs-DLLVktBrArhpdkLGJzQ5-KX6jzlC0xI0xdUibgUIH0UKTU6nioSo1fV3RsxA3fUgchhOljg1wuiX2ALo0xrzIQodT3EP9DdKjRmoQZj6hVi4G69SgiwxnKlMlLGgofWBMvWGSVDaeK5ltuyEmTUDiNVHW9lqGOeglAT2XmHA%3D%3D&attredirects=0
"""
###################################

"""
 In this assignment, the task is to implement different search algorithms to find
 a path from a given start cell to the goal cell in a 2D grid map.

 To complete the assignment, you must finish three functions:
   depth_first_search (line 148), uniform_cost_search (line 224),
   and astar_search (line 264).

 During the search, a cell is represented using a tuple of its location
 coordinates.
 For example:
   the cell at the left-top corner is (0, 0);
   the cell at the first row and second column is (0, 1);
   the cell at the second row and first column is (1, 0).
 You need put these tuples into the open set or/and closed set properly
 during searching.
"""

# ACTIONS defines how to reach an adjacent cell from a given cell
# Important: please check that a cell within the bounds of the grid
# when try to access it.
ACTIONS = (
    (-1,  0), # go up
    ( 0, -1), # go left
    ( 1,  0), # go down
    ( 0,  1)  # go right
)

from utils.search_app import OrderedSet, Stack, Queue, PriorityQueue
"""
 Four different structures are provided as the containers of the open set
 and closed set.

 OrderedSet is an ordered collections of unique elements.

 Stack is an LIFO container whose `pop()` method always pops out the last
 added element.

 Queue is an FIFO container whose `pop()` method always pops out the first
 added element.

 PriorityQueue is a key-value container whose `pop()` method always pops out
 the element whose value has the highest priority.

 All of these containers are iterable but not all of them are ordered. Use
 their pop() methods to ensure elements are popped out as expected.


 Common operations of OrderedSet, Stack, Queue, PriorityQueue
   len(s): number of elements in the container s
   x in s: test x for membership in s
   x not in s: text x for non-membership in s
   s.clear(): clear s
   s.remove(x): remove the element x from the set s;
                nothing will be done if x is not in s


 Unique operations of OrderedSet:
   s.add(x): add the element x into the set s;
             nothing will be done if x is already in s
   s.pop(): return and remove the LAST added element in s;
            raise IndexError if s is empty
   s.pop(last=False): return and remove the FIRST added element in s;
            raise IndexError if s is empty
 Example:
   s = Set()
   s.add((1,2))    # add a tuple element (1,2) into the set
   s.remove((1,2)) # remove the tuple element (1,2) from the set
   s.add((1,1))
   s.add((2,2))
   s.add((3,3))
   x = s.pop()
   assert(x == (3,3))
   assert((1,1) in s and (2,2) in s)
   assert((3,3) not in s)
   x = s.pop(last=False)
   assert(x == (1,1))
   assert((2,2) in s)
   assert((1,1) not in s)


 Unique operations of Stack:
   s.append(x): add the element x into the back of the stack s
   s.pop(): return and remove the LAST added element in the stack s;
            raise IndexError if s is empty
 Example:
   s = Stack()
   s.add((1,1))
   s.add((2,2))
   x = s.pop()
   assert(x == (2,2))
   assert((1,1) in s)
   assert((2,2) not in s)


 Unique operations of Queue:
   s.append(x): add the element x into the back of the queue s
   s.pop(): return and remove the FIRST added element in the queue s;
            raise IndexError if s is empty
 Example:
   s = Queue()
   s.add((1,1))
   s.add((2,2))
   x = s.pop()
   assert(x == (1,1))
   assert((2,2) in s)
   assert((1,1) not in s)


 Unique operations of PriorityQueue:
   PriorityQueue(order="min", f=lambda v: v): build up a priority queue
       using the function f to compute the priority based on the value
       of an element
   s.put(x, v): add the element x with value v into the queue
                update the value of x if x is already in the queue
   s.get(x): get the value of the element x
            raise KeyError if x is not in s
   s.pop(): return and remove the element with highest priority in s;
            raise IndexError if s is empty
            if order is "min", the element with minimum f(v) will be popped;
            if order is "max", the element with maximum f(v) will be popped.
 Example:
   s = PriorityQueue(order="max", f=lambda v: abs(v))
   s.put((1,1), -1)
   s.put((2,2), -20)
   s.put((3,3), 10)
   x, v = s.pop()  # the element with maximum value of abs(v) will be popped
   assert(x == (2,2) and v == -20)
   assert(x not in s)
   assert(x.get((1,1)) == -1)
   assert(x.get((3,3)) == 10)
"""


# use math library if needed
import math

###################################

"""
  Code Based on:
  Author: Ioannis Karamouzas
  Date of Retrieval: 09/17/2019
  Availibility: https://89fa4bd4-a-1e6e9713-s-sites.googlegroups.com/a/g.clemson.edu/cpsc-ai/schedule/lec04.pdf?attachauth=ANoY7cq4bmGQsYIulNLZvZxka5zCruNV9z0Px1mBndHBu8evrdxX8JM4WCMP--Vh4kN-vY9EVqYvZQBynW7eehaKy7g7HNe-P2rJaITYUPO6aOu8DTfnsLzw3RXf-pF6L2NwPfWTANjxeQw0_DiD90yQx8JdHB2JfbAml5XGahOiihHZEKPcCXL-ywHNPgzY4C1-X_sYo9pJbm1QLlU0sTbDu2WAOyxgOQ%3D%3D&attredirects=0
"""
###################################

def depth_first_search(grid_size, start, goal, obstacles, costFn, logger):
    """
    DFS algorithm finds the path from the start cell to the
    goal cell in the 2D grid world.

    Parameters
    ----------
    grid_size: tuple, (n_rows, n_cols)
        (number of rows of the grid, number of cols of the grid)
    start: tuple, (row, col)
        location of the start cell;
        row and col are counted from 0, i.e. the 1st row is 0
    goal: tuple, (row, col)
        location of the goal cell
    obstacles: tuple, ((row, col), (row, col), ...)
        locations of obstacles in the grid
        the cells where obstacles are located are not allowed to access
    costFn: a function that returns the cost of landing to a cell (x,y)
         after taking an action. The default cost is 1, regardless of the
         action and the landing cell, i.e. every time the agent moves NSEW
         it costs one unit.
    logger: a logger to visualize the search process.
         Do not do anything to it.



    Returns
    -------
    movement along the path from the start to goal cell: list of actions
        The first returned value is the movement list found by the search
        algorithm from the start cell to the end cell.
        The movement list should be a list object who is composed of actions
        that should made moving from the start to goal cell along the path
        found the algorithm.
        For example, if nodes in the path from the start to end cell are:
            (0, 0) (start) -> (0, 1) -> (1, 1) -> (1, 0) (goal)
        then, the returned movement list should be
            [(0,1), (1,0), (0, -1)]
        which means: move right, down, left.

        Return an EMPTY list if the search algorithm fails finding any
        available path.

    closed set: list of location tuple (row, col)
        The second returned value is the closed set, namely, the cells are expanded during search.
    """
    n_rows, n_cols = grid_size
    start_row, start_col = start
    goal_row, goal_col = goal

    ##########################################
    # Choose a proper container yourself from
    # OrderedSet, Stack, Queue, PriorityQueue
    # for the open set and closed set.
    open_set = Stack()
    closed_set = OrderedSet()
    ##########################################

    ##########################################
    # Set up visualization logger hook
    # Please do not modify these four lines
    closed_set.logger = logger
    logger.closed_set = closed_set
    open_set.logger = logger
    logger.open_set = open_set
    ##########################################
    movement = []

    # ----------------------------------------
    # finish the code below

    ######################
    """
    In this part, we pop the node from Stack and add it to the closed Set
    All the children from the poped node are added to the Open Set provided that they are not already in the Open or Closed set
    (Provided that they aren't in the closed list, they arent obstacles and fit in the grid.)
    The nodes opened are noted into the list - myList to be used further for showing the movement.
    """
    ######################
    open_set.add(start)
    mylist = []
    while open_set:
        node = open_set.pop()
        closed_set.add(node)
        mylist.append(node)
        if(node == goal):
            break;
        if((node[0]-1,node[1]) not in obstacles and node[0]-1 >= 0 and (node[0]-1,node[1]) not in closed_set ):
            open_set.add((node[0]-1,node[1]))
        if((node[0],node[1]-1) not in obstacles and node[1]-1 >= 0 and (node[0],node[1]-1) not in closed_set ):
            open_set.add((node[0],node[1]-1))
        if((node[0]+1,node[1]) not in obstacles and node[0]+1 < grid_size[0] and (node[0]+1,node[1]) not in closed_set):
            open_set.add((node[0]+1,node[1]))
        if((node[0],node[1]+1) not in obstacles and node[1]+1 < grid_size[1] and (node[0],node[1]+1) not in closed_set ):
            open_set.add((node[0],node[1]+1))

    ######################
    """
    Here, we backtrack the list and compare the current and previous nodes.
    The difference of their Row and Column values help us normalize the values into Actions from Line 59
    """
    ######################

    x = len(mylist)-1
    cur_path = mylist[x]
    for i in range (x,-1,-1):
        x2 = mylist[i-1][0]
        y2 = mylist[i-1][1]
        if((cur_path[0]-x2 ,cur_path[1]-y2) in ACTIONS):
            movement.insert(0,(cur_path[0]-x2 ,cur_path[1]-y2))
            cur_path = mylist[i-1]


    # ----------------------------------------
#############################################################################

#############################################################################
    return movement, closed_set

#Here we return the movement along the path from start state to goal state and and the nodes expanded in the search.

###################################
"""
  Code Based on:
  Author: Ioannis Karamouzas
  Date of Retrieval: 09/17/2019
  Availibility: https://89fa4bd4-a-1e6e9713-s-sites.googlegroups.com/a/g.clemson.edu/cpsc-ai/schedule/lec04.pdf?attachauth=ANoY7cq4bmGQsYIulNLZvZxka5zCruNV9z0Px1mBndHBu8evrdxX8JM4WCMP--Vh4kN-vY9EVqYvZQBynW7eehaKy7g7HNe-P2rJaITYUPO6aOu8DTfnsLzw3RXf-pF6L2NwPfWTANjxeQw0_DiD90yQx8JdHB2JfbAml5XGahOiihHZEKPcCXL-ywHNPgzY4C1-X_sYo9pJbm1QLlU0sTbDu2WAOyxgOQ%3D%3D&attredirects=0
"""
###################################


def uniform_cost_search(grid_size, start, goal, obstacles, costFn, logger):
    """
    Uniform-cost search algorithm finds the optimal path from
    the start cell to the goal cell in the 2D grid world.

    After expanding a node, to retrieve the cost of a child node at location (x,y),
    please call costFn((x,y)). In all of the grid maps, the cost is always 1.

    See depth_first_search() for details.
    """
    n_rows, n_cols = grid_size
    start_row, start_col = start
    goal_row, goal_col = goal

    ##########################################
    # Choose a proper container yourself from
    # OrderedSet, Stack, Queue, PriorityQueue
    # for the open set and closed set.
    open_set = PriorityQueue(order="min", f=lambda v: abs(v))
    closed_set = OrderedSet()
    ##########################################

    ##########################################
    # Set up visualization logger hook
    # Please do not modify these four lines
    closed_set.logger = logger
    logger.closed_set = closed_set
    open_set.logger = logger
    logger.open_set = open_set
    ##########################################

    movement = []

    # ----------------------------------------
    # finish the code below

    ######################
    """
        The function add_update is used to add the child node to open list.
        If it is already in the open list but has a higher cost than its current value, then we replace the former parent and set it's parent to the current node.
    """
    def add_update(child_node,parent_node,value):
        value = value + costFn(child_node)
        if(child_node not in open_set):
            open_set.put(child_node,value)
            mylist[child_node] = (parent_node,value)
        elif(value < mylist[child_node][1]):
            open_set.remove(child_node)
            open_set.put(child_node,value)
            mylist[child_node] = (parent_node,value)

    ######################
        """
        We implement the main logic of Uniform cost search
        We pop the cell with the lowest cost.
        We then add the adjacent cells to the open_set provided that they aren't in the closed list, they arent obstacles and fit in the grid.
        """
    ######################
    mylist = {}
    open_set.put(start,0)
    mylist[start] = ([None],0)
    while open_set:
        node, value = open_set.pop()
        closed_set.add(node)
        if(node == goal):
            break;
        else:
            if((node[0]-1,node[1]) not in obstacles and node[0]-1 >= 0 and (node[0]-1,node[1]) not in closed_set):
                add_update((node[0]-1,node[1]),node,value)
            if((node[0],node[1]-1) not in obstacles and node[1]-1 >= 0 and (node[0],node[1]-1) not in closed_set):
                add_update((node[0],node[1]-1),node,value)
            if((node[0]+1,node[1]) not in obstacles and node[0]+1 < grid_size[0] and (node[0]+1,node[1]) not in closed_set):
                add_update((node[0]+1,node[1]),node,value)
            if((node[0],node[1]+1) not in obstacles and node[1]+1 < grid_size[1] and (node[0],node[1]+1) not in closed_set):
                add_update((node[0],node[1]+1),node,value)
######################
    """
    The path is found by going back from the goal node to the start node by comparing it with ACTIONS from Line 59.
    """
######################
    path = []
    path_node = goal
    while path_node!=start:
        curr_node = path_node
        path.append(mylist[path_node][0])
        path_node = (mylist[path_node][0])
        if((curr_node[0] - path_node[0],curr_node[1] - path_node[1]) in ACTIONS):
            movement.insert(0,(curr_node[0] - path_node[0],curr_node[1] - path_node[1]))

    # ----------------------------------------
#############################################################################

#############################################################################
    return movement, closed_set
#We return the movement along the path from start state to goal state and and the nodes expanded in the search.


###################################
"""
  Code Based on:
  Author: Ioannis Karamouzas
  Date of Retrieval: 09/17/2019
  Availibility: https://89fa4bd4-a-1e6e9713-s-sites.googlegroups.com/a/g.clemson.edu/cpsc-ai/schedule/lec05.pdf?attachauth=ANoY7cpjcxMzrtkbL6sPKEMo0MJ1Ylhe9HaZ_P_gmiTAdttcTwJ9kNNrs-DLLVktBrArhpdkLGJzQ5-KX6jzlC0xI0xdUibgUIH0UKTU6nioSo1fV3RsxA3fUgchhOljg1wuiX2ALo0xrzIQodT3EP9DdKjRmoQZj6hVi4G69SgiwxnKlMlLGgofWBMvWGSVDaeK5ltuyEmTUDiNVHW9lqGOeglAT2XmHA%3D%3D&attredirects=0
"""
###################################


def astar_search(grid_size, start, goal, obstacles, costFn, logger):
    """
    A* search algorithm finds the optimal path from the start cell to the
    goal cell in the 2D grid world.

    After expanding a node, to retrieve the cost of a child node at location (x,y),
    please call costFn((x,y)). In all of the grid maps, the cost is always 1.

    See depth_first_search() for details.
    """
    n_rows, n_cols = grid_size
    start_row, start_col = start
    goal_row, goal_col = goal

    ##########################################
    # Choose a proper container yourself from
    # OrderedSet, Stack, Queue, PriorityQueue
    # for the open set and closed set.
    open_set = PriorityQueue(order="min", f=lambda v: abs(v))
    closed_set = OrderedSet()
    ##########################################

    ##########################################
    # Set up visualization logger hook
    # Please do not modify these four lines
    closed_set.logger = logger
    logger.closed_set = closed_set
    open_set.logger = logger
    logger.open_set = open_set
    ##########################################

    ######################
    """
        The heuristic value is calculated by using the Manhattan distance from the cell to the goal cell.
        Manhattan distance : |x2-x1|+|y2-y1|

        Code Based on:
        Author: Paul E. Black
        Date of Retrieval: 09/17/2019
        Availibility: https://xlinux.nist.gov/dads/HTML/manhattanDistance.html

    """
    ######################
    def heuristic(row, col):
        return abs(row - goal[0])+abs(col-goal[1])

    movement = []

    # ----------------------------------------
    # finish the code below to implement a Manhattan distance heuristic
    ######################
    """
            The function add_update is used to calculate the F-Value and add the child node to open list.
            The F-Value is calculated by adding the step cost and the heuristic value.
            f(n)=g(n)+h(n) where
            f(n) is the funcion to calculate the F-value
            g(n) is the true cost from the start node to the current node
            h(n) is the heuristic value.
            If it is already in the open list but has a higher F-Value than its current value, then we replace the former parent and set it's parent to the current node.
    """
    ######################
    def add_update(child_node,parent_node,value):
        value = value + costFn(child_node)
        if(child_node not in open_set):
            open_set.put(child_node,value + heuristic(child_node[0],child_node[1]))
            cost_set[child_node] = (parent_node,value)
        elif(value < cost_set[child_node][1]):
            open_set.remove(child_node)
            open_set.put(child_node,value + heuristic(child_node[0],child_node[1]))
            cost_set[child_node] = (parent_node,value)
    ######################
        """
        We implement the main logic of A star algorithm
        We pop the cell with the lowest F-Value.
        We then add the adjacent cells to the open_set provided that they aren't in the closed list, they arent obstacles and fit in the grid.
        """
    ######################
    open_set.put(start,heuristic(start[0],start[1]))
    cost_set = {}
    cost_set[start]=([None],0)
    while open_set:
        node, value = open_set.pop()
        closed_set.add(node)
        if(node == goal):
            break;
        else:
            if((node[0]-1,node[1]) not in obstacles and node[0]-1 >= 0 and (node[0]-1,node[1]) not in closed_set):
               add_update((node[0]-1,node[1]),node,cost_set[node][1])
            if((node[0],node[1]-1) not in obstacles and node[1]-1 >= 0 and (node[0],node[1]-1) not in closed_set):
               add_update((node[0],node[1]-1),node,cost_set[node][1])
            if((node[0]+1,node[1]) not in obstacles and node[0]+1 < grid_size[0] and (node[0]+1,node[1]) not in closed_set):
               add_update((node[0]+1,node[1]),node,cost_set[node][1])
            if((node[0],node[1]+1) not in obstacles and node[1]+1 < grid_size[1] and (node[0],node[1]+1) not in closed_set):
               add_update((node[0],node[1]+1),node,cost_set[node][1])

######################
    """
    The path is found by going back from the goal node to the start node by comparing it with ACTIONS from Line 59.
    """
######################
    path = []
    path_node = goal
    while path_node!=start:
        curr_node = path_node
        path.append(cost_set[path_node][0])
        path_node = (cost_set[path_node][0])
        if((curr_node[0] - path_node[0],curr_node[1] - path_node[1]) in ACTIONS):
            movement.insert(0,(curr_node[0] - path_node[0],curr_node[1] - path_node[1]))
    # ----------------------------------------



#############################################################################

#############################################################################
    return movement, closed_set

#Here we return the movement along the path from start state to goal state and and the nodes expanded in the search.


###################################
# Extra Code.
###################################

###################################
"""
  Code Based on:
  Author: Ioannis Karamouzas
  Date of Retrieval: 09/17/2019
  Availibility: https://89fa4bd4-a-1e6e9713-s-sites.googlegroups.com/a/g.clemson.edu/cpsc-ai/schedule/lec05.pdf?attachauth=ANoY7cpjcxMzrtkbL6sPKEMo0MJ1Ylhe9HaZ_P_gmiTAdttcTwJ9kNNrs-DLLVktBrArhpdkLGJzQ5-KX6jzlC0xI0xdUibgUIH0UKTU6nioSo1fV3RsxA3fUgchhOljg1wuiX2ALo0xrzIQodT3EP9DdKjRmoQZj6hVi4G69SgiwxnKlMlLGgofWBMvWGSVDaeK5ltuyEmTUDiNVHW9lqGOeglAT2XmHA%3D%3D&attredirects=0
"""
###################################

def my_search(grid_size, start, goal, obstacles, costFn, logger):
    """
    Greedy Best First Search
    It expands the nodes with the lowest heuristic distance (in this case Manhattan distance : |x2-x1|+|y2-y1|).
    """
    n_rows, n_cols = grid_size
    start_row, start_col = start
    goal_row, goal_col = goal

    ##########################################
    # Choose a proper container yourself from
    # OrderedSet, Stack, Queue, PriorityQueue
    # for the open set and closed set.
    open_set = PriorityQueue(order="min", f=lambda v: abs(v))
    closed_set = OrderedSet()
    ##########################################

    ##########################################
    # Set up visualization logger hook
    # Please do not modify these four lines
    closed_set.logger = logger
    logger.closed_set = closed_set
    open_set.logger = logger
    logger.open_set = open_set

    ######################
    """
            The heuristic value is calculated by using the Manhattan distance from the cell to the goal cell.
            Manhattan distance : |x2-x1|+|y2-y1|

            Code Based on:
            Author: Paul E. Black
            Date of Retrieval: 09/17/2019
            Availibility: https://xlinux.nist.gov/dads/HTML/manhattanDistance.html

    """
    ######################
    ##########################################
    def heuristic(row, col):
        return abs(row - goal[0])+abs(col-goal[1])


    # ----------------------------------------
    # finish the code below to implement a Manhattan distance heuristic


    ######################
    """
            The Best First Search Greedy Algorithm is implemented by popping the cell with the lowest heuristic value.
            We then add the adjacent cells to open list.
    """
    ######################
    open_set.put(start,heuristic(start[0],start[1]))
    movement = []
    mylist = {}
    mylist[start] = ([None],0)
    while open_set:
        node, value = open_set.pop()
        closed_set.add(node)
        if(node == goal):
            break;
        else:
            if((node[0]-1,node[1]) not in obstacles and node[0]-1 >= 0 and (node[0]-1,node[1]) not in closed_set and (node[0]-1,node[1]) not in open_set ):
                mylist[(node[0]-1,node[1])] = ([node],heuristic(node[0]-1,node[1]))
                open_set.put((node[0]-1,node[1]),heuristic(node[0]-1,node[1]))

            if((node[0],node[1]-1) not in obstacles and node[1]-1 >= 0 and (node[0],node[1]-1) not in closed_set and (node[0],node[1]-1) not in open_set ):
                mylist[(node[0],node[1]-1)] = ([node],heuristic(node[0],node[1]-1))
                open_set.put((node[0],node[1]-1),heuristic(node[0],node[1]-1))

            if((node[0]+1,node[1]) not in obstacles and node[0]+1 < grid_size[0] and (node[0]+1,node[1]) not in closed_set and (node[0]+1,node[1]) not in open_set):
                mylist[(node[0]+1,node[1]) ] = ([node],heuristic(node[0]+1,node[1]))
                open_set.put((node[0]+1,node[1]),heuristic(node[0]+1,node[1]))

            if((node[0],node[1]+1) not in obstacles and node[1]+1 < grid_size[1] and (node[0],node[1]+1) not in closed_set and (node[0],node[1]+1) not in open_set ):
                mylist[(node[0],node[1]+1)] = ([node],heuristic(node[0],node[1]+1))
                open_set.put((node[0],node[1]+1),heuristic(node[0],node[1]+1))

######################
    """
    The path is found by going back from the goal node to the start node by comparing it with ACTIONS from Line 59.
    """
######################
    path = []
    path_node = goal
    while path_node!=start:
        curr_node = path_node
        path.append(mylist[path_node][0][0])
        path_node = (mylist[path_node][0][0])
        if((curr_node[0] - path_node[0],curr_node[1] - path_node[1]) in ACTIONS):
            movement.insert(0,(curr_node[0] - path_node[0],curr_node[1] - path_node[1]))
    # ----------------------------------------



#############################################################################

#############################################################################
    return movement, closed_set

######################
    """
    This is the Main method which calls all the algorithms as per the user's needs.
    """
######################

if __name__ == "__main__":
    # make sure actions and cost are defined correctly
    from utils.search_app import App
    assert(ACTIONS == App.ACTIONS)

    import tkinter as tk

    algs = {
        "Depth-First": depth_first_search,
        "Uniform Cost Search": uniform_cost_search,
        "A*": astar_search,
        "Greedy Best First Search": my_search
    }

    root = tk.Tk()
    App(algs, root)
    root.mainloop()
