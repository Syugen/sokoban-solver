#Look for #IMPLEMENT tags in this file. These tags indicate what has
#to be implemented to complete the Sokoban warehouse domain.

#   You may add only standard python imports---i.e., ones that are automatically
#   available on TEACH.CS
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

# import os for time functions
import os
from search import * #for search engines
from sokoban import SokobanState, Direction, PROBLEMS, sokoban_goal_state #for Sokoban specific classes and problems
import numpy as np
from scipy.optimize import linear_sum_assignment

#SOKOBAN HEURISTICS
def heur_displaced(state):
  '''trivial admissible sokoban heuristic'''
  '''INPUT: a sokoban state'''
  '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''       
  count = 0
  for box in state.boxes:
    if box not in state.storage:
      count += 1
  return count

def heur_manhattan_distance(state):
#IMPLEMENT
    '''admissible sokoban heuristic: manhattan distance'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''      
    #We want an admissible heuristic, which is an optimistic heuristic. 
    #It must always underestimate the cost to get from the current state to the goal.
    #The sum Manhattan distance of the boxes to their closest storage spaces is such a heuristic.  
    #When calculating distances, assume there are no obstacles on the grid and that several boxes can fit in one storage bin.
    #You should implement this heuristic function exactly, even if it is tempting to improve it.
    #Your function should return a numeric value; this is the estimate of the distance to the goal.
    distance_sum = 0
    for box in state.boxes:
      closest = None
      for storage in state.storage:
        if state.restrictions and storage not in state.restrictions[state.boxes[box]]:
          continue
        distance = abs(box[0] - storage[0]) + abs(box[1] - storage[1])
        if closest is None or distance < closest:
          closest = distance
      if closest is None:
        closest = 0
      distance_sum += closest
    return distance_sum

def heur_alternate(state):
#IMPLEMENT
    '''a better sokoban heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''        
    #heur_manhattan_distance has flaws.   
    #Write a heuristic function that improves upon heur_manhattan_distance to estimate distance between the current state and the goal.
    #Your function should return a numeric value for the estimate of the distance to the goal.
    
    #print(state.print_state())
    obs = state.obstacles
    boxes = list(state.boxes.keys())
    tars = list(state.storage.keys())
    res = state.restrictions
    
    # Unmoveable boxes
    for box in boxes:
      if res is None and box in tars or res is not None and box in res[state.boxes[box]]: continue
      
      # Box near wall
      if box[0] == 0 and all(storage[0] != 0 for storage in tars) or\
         box[0] == state.width - 1 and all(storage[0] != state.width - 1 for storage in tars) or\
         box[1] == 0 and all(storage[1] != 0 for storage in tars) or\
         box[1] == state.height - 1 and all(storage[1] != state.height - 1 for storage in tars):
        return 10e8
      
      # Box in corner
      L, R, U, D = surrounding_obstacles(state, box[0], box[1])
      if (L and U) or (U and R) or (R and D) or (D and L):
        return 10e8
      
      # Problem 35
      if (L or R) and box[1] > 0 and (box[0], box[1] - 1) not in tars and\
         sum(surrounding_obstacles(state, box[0], box[1] - 1)) == 3:
        return 10e8
    
    # Unreachable storage
    '''for tar in tars:
      # Storage blocked (only unreachable in MOST CASES, NOT ALL)
      L, R, U, D = surrounding_obstacles(state, tar[0], tar[1])
      #print(tar, L, R, U, D)
      if tar[1] + 3 < state.height and (tar[0], tar[1] + 1) in state.boxes and (tar[0], tar[1] + 2) in state.boxes and\
         (tar[0] == 0 or tar[0] == state.width - 1 or (tar[0] - 1, tar[1] + 1) in obs or (tar[0] + 1, tar[1] + 1) in obs) and\
         (tar[0] == 0 or tar[0] == state.width - 1 or (tar[0] - 1, tar[1] + 2) in obs or (tar[0] + 1, tar[1] + 2) in obs):
        if L and U or U and R: return 10e8
      if tar[0] + 3 < state.width and (tar[0] + 1, tar[1]) in state.boxes and (tar[0] + 2, tar[1]) in state.boxes and\
         (tar[1] == 0 or tar[1] == state.height - 1 or (tar[0] + 1, tar[1] - 1) in obs or (tar[0] + 1, tar[1] + 1) in obs) and\
         (tar[1] == 0 or tar[1] == state.height - 1 or (tar[0] + 2, tar[1] - 1) in obs or (tar[0] + 2, tar[1] + 1) in obs):
        if L and U or L and D: return 10e8
      if tar[0] - 3 >= 0 and (tar[0] - 1, tar[1]) in state.boxes and (tar[0] - 2, tar[1]) in state.boxes and\
         (tar[1] == 0 or tar[1] == state.height - 1 or (tar[0] - 1, tar[1] - 1) in obs or (tar[0] - 1, tar[1] + 1) in obs) and\
         (tar[1] == 0 or tar[1] == state.height - 1 or (tar[0] - 2, tar[1] - 1) in obs or (tar[0] - 2, tar[1] + 1) in obs):      
        if U and R or R and D: return 10e8
      if tar[1] - 3 >= 0 and (tar[0], tar[1] - 1) in state.boxes and (tar[0], tar[1] - 2) in state.boxes and\
         (tar[0] == 0 or tar[0] == state.width - 1 or (tar[0] - 1, tar[1] - 1) in obs or (tar[0] + 1, tar[1] - 1) in obs) and\
         (tar[0] == 0 or tar[0] == state.width - 1 or (tar[0] - 1, tar[1] - 2) in obs or (tar[0] + 1, tar[1] - 2) in obs):
        if L and D or D and R: return 10e8
       ''' 
    
    # 1. Manhattan distance of robot to closest displaced box
    closest = None
    for box in boxes:
      if res is None and box in tars or res is not None and box in res[state.boxes[box]]:
        continue
      distance = abs(box[0] - state.robot[0]) + abs(box[1] - state.robot[1])
      if closest is None or distance < closest:
        closest = distance
    if closest is None:
      closest = 1

    # 2. Sum of Manhattan distance of boxes to targets
    distance_sum = 0
    costs = np.zeros((len(boxes), len(tars)))
    i = 0
    for box in boxes:
      j = 0
      for tar in tars:
        if res and tar not in res[state.boxes[box]]:
          costs[i][j] = 10e8
        else:
          costs[i][j] = get_box_storage_distance(state, box, tar, [])
        j += 1
      i += 1
    row_ind, col_ind = linear_sum_assignment(costs)
    box_distance = costs[row_ind, col_ind].sum()
    
    # 3. Bonus. If storage in the corner filled, reduce cost
    bonus = 0
    num_obs = [sum(surrounding_obstacles(state, tar[0], tar[1])) for tar in tars]
    j = 0
    for tar in tars:
      # Bonus for moving closer to storage with highest surronding obstacles
      bonus += max(0, len(tars) - min(costs[:, j])) * max(0, num_obs[j] - max(num_obs) + 1)
      '''
      # If storage has box
      if tar in boxes and (res is None or res is not None and tar in res[state.boxes[tar]]):
        bonus += len(tars) * max(0, num_obs[j] - max(num_obs) + 1)
      else:
        d = (tar[0] - state.robot[0]) ** 2 + (tar[1] - state.robot[1]) ** 2
        bonus += max(0, len(tars) - d) * max(0, num_obs[j] - max(num_obs) + 1)
      '''
      j += 1
      '''
      # Storage blocked (only unreachable in MOST CASES, NOT ALL)
      if tar in state.boxes and (res is None or res is not None and tar in res[state.boxes[tar]]):
        L, R, U, D = surrounding_obstacles(state, tar[0], tar[1])
        bonus += sum([L, R, U, D]) + len(tars)
        print("here",bonus)
        print(L and R , tar[1] > 0 , (tar[0], tar[1] - 1) == state.robot)
        if L and R and tar[1] > 0 and (tar[0], tar[1] - 1) == state.robot:
          i = 1
          while tar[1] + i < state.height and (tar[0], tar[1] + i) in tars: 
            bonus -= 1
            i += 1
        print("!!",bonus)
      '''
          
          
      
    #print(costs)
    
    #print(costs[row_ind, col_ind].sum(), int(closest ** 0.5), bonus)
    #print("?", costs[row_ind, col_ind].sum() + int(closest ** 0.5) - bonus)
    return box_distance + int(closest ** 0.5) - bonus
    #'''


def surrounding_obstacles(state, col, row):
  return (col == 0 or (col - 1, row) in state.obstacles, 
          col == state.width - 1 or (col + 1, row) in state.obstacles,
          row == 0 or (col, row - 1) in state.obstacles,
          row == state.height - 1 or (col, row + 1) in state.obstacles) 
  
def get_box_storage_distance(state, pos, target, been):
  #print(pos)
  been.append(pos)
  pos_col, pos_row = pos
  tar_col, tar_row = target
  if pos_col == tar_col and pos_row == tar_row:
    return 0
  LEFT = (pos_col - 1, pos_row)
  RIGHT = (pos_col + 1, pos_row)
  UP = (pos_col, pos_row - 1)
  DOWN = (pos_col, pos_row + 1)
  if abs(tar_row - pos_row) <= abs(pos_col - tar_col):
    if tar_col <= pos_col and tar_row <= pos_row:
      seq = [LEFT, UP, DOWN, RIGHT]
    elif tar_col <= pos_col:
      seq = [LEFT, DOWN, UP, RIGHT]
    elif tar_row <= pos_row:
      seq = [RIGHT, UP, DOWN, LEFT] 
    else:
      seq = [RIGHT, DOWN, UP, LEFT] 
  else:
    if tar_row <= pos_row and tar_col <= pos_col:
      seq = [UP, LEFT, RIGHT, DOWN] 
    elif tar_row <= pos_row:
      seq = [UP, RIGHT, LEFT, DOWN] 
    elif tar_col <= pos_col:
      seq = [DOWN, LEFT, RIGHT, DOWN] 
    else:
      seq = [DOWN, RIGHT, LEFT, DOWN]
  for succ in seq:
    if succ in been: continue
    if succ == LEFT and (succ[0] < 0 or succ in state.obstacles or RIGHT[0] >= state.width or RIGHT in state.obstacles) or\
       succ == RIGHT and (succ[0] >= state.width or succ in state.obstacles or LEFT[0] < 0 or LEFT in state.obstacles) or\
       succ == UP and (succ[1] < 0 or succ in state.obstacles or DOWN[1] >= state.height or DOWN in state.obstacles) or\
       succ == DOWN and (succ[1] >= state.height or succ in state.obstacles or UP[1] < 0 or UP in state.obstacles):
      continue
    sub = get_box_storage_distance(state, succ, target, been)
    if sub < 10e8: return 1 + sub
  return 10e8

def fval_function(sN, weight):
#IMPLEMENT
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.

    @param sNode sN: A search node (containing a SokobanState)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """
  
    #Many searches will explore nodes (or states) that are ordered by their f-value.
    #For UCS, the fvalue is the same as the gval of the state. For best-first search, the fvalue is the hval of the state.
    #You can use this function to create an alternate f-value for states; this must be a function of the state and the weight.
    #The function must return a numeric f-value.
    #The value will determine your state's position on the Frontier list during a 'custom' search.
    #You must initialize your search engine object as a 'custom' search engine if you supply a custom fval function.
    return sN.gval + weight * sN.hval

def anytime_gbfs (initial_state, heur_fn, timebound = 10):
#IMPLEMENT
    '''Provides an implementation of anytime greedy best-first search, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False''' 
    start_time = os.times()[0]
    end_time = start_time + timebound
    se = SearchEngine('best_first', 'full')
    se.init_search(initial_state, goal_fn=sokoban_goal_state, heur_fn=heur_fn)
    final = False
    costbound = None
    while os.times()[0] < end_time:
      timebound = end_time - os.times()[0]
      result = se.search(timebound, costbound)
      if result:
        final = result
      else:
        break
      costbound = (final.gval - 10e-8, 10e8, 10e8)
    return final

def anytime_weighted_astar(initial_state, heur_fn, weight=1., timebound = 10):
#IMPLEMENT
    '''Provides an implementation of anytime weighted a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False''' 
    start_time = os.times()[0]
    end_time = start_time + timebound
    se = SearchEngine('custom', 'full')
    wrapped_fval_function = (lambda sN: fval_function(sN, weight))
    se.init_search(initial_state, goal_fn=sokoban_goal_state, 
                   heur_fn=heur_fn, fval_function=wrapped_fval_function)   
    final = False
    costbound = None
    while os.times()[0] < end_time:
      timebound = end_time - os.times()[0]
      result = se.search(timebound, costbound)
      if result:
        final = result
      else:
        break
      costbound = (10e8, 10e8, final.gval - 10e-8)
    return final
'''
if __name__ == "__main__":
  #TEST CODE
  solved = 0; unsolved = []; counter = 0; percent = 0; timebound = 2; #2 second time limit for each problem
  print("*************************************")  
  print("Running A-star")     

  for i in range(0, 10): #note that there are 40 problems in the set that has been provided.  We just run through 10 here for illustration.

    print("*************************************")  
    print("PROBLEM {}".format(i))
    
    s0 = PROBLEMS[i] #Problems will get harder as i gets bigger

    se = SearchEngine('astar', 'full')
    se.init_search(s0, goal_fn=sokoban_goal_state, heur_fn=heur_displaced)
    final = se.search(timebound)

    if final:
      final.print_path()
      solved += 1
    else:
      unsolved.append(i)    
    counter += 1

  if counter > 0:  
    percent = (solved/counter)*100

  print("*************************************")  
  print("{} of {} problems ({} %) solved in less than {} seconds.".format(solved, counter, percent, timebound))  
  print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))      
  print("*************************************") 

  solved = 0; unsolved = []; counter = 0; percent = 0; timebound = 8; #8 second time limit 
  print("Running Anytime Weighted A-star")   

  for i in range(0, 10):
    print("*************************************")  
    print("PROBLEM {}".format(i))

    s0 = PROBLEMS[i] #Problems get harder as i gets bigger
    weight = 10
    final = anytime_weighted_astar(s0, heur_fn=heur_displaced, weight=weight, timebound=timebound)

    if final:
      final.print_path()   
      solved += 1 
    else:
      unsolved.append(i)
    counter += 1      

  if counter > 0:  
    percent = (solved/counter)*100   
      
  print("*************************************")  
  print("{} of {} problems ({} %) solved in less than {} seconds.".format(solved, counter, percent, timebound))  
  print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))      
  print("*************************************") 



'''