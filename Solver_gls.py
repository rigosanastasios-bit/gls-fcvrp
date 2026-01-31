"""
Guided Local Search (GLS) Solver for Family-CVRP

This module implements a Guided Local Search metaheuristic for solving the
Family Capacitated Vehicle Routing Problem. GLS escapes local optima by
penalizing frequently-used solution features (arcs/edges).

Algorithm Overview:
    1. Construct initial solution using Nearest Neighbor heuristic
    2. Apply local search operators (Relocation, Swap, 2-Opt, FamilySwap)
    3. When stuck, penalize costly arcs to guide search to new regions
    4. Reset penalties upon finding new best solutions (intensification)
    5. Polish final solution with Variable Neighborhood Descent (VND)

Key Components:
    - Solution: Container for routes and total cost
    - Move Classes: RelocationMove, SwapMove, TwoOptMove, FamilySwapMove
    - Solver: Main GLS solver with all operators and search logic

References:
    - Voudouris, C., & Tsang, E. (1999). Guided local search and its 
      application to the traveling salesman problem.
    - Mendoza, J. E., et al. (2010). A memetic algorithm for the 
      multi-compartment vehicle routing problem with stochastic demands.

Author: Anastasios Rigos
"""

from F_CVRP_Model import *
import random


class Solution:
    """
    Represents a complete solution to the F-CVRP.
    
    Attributes:
        cost (float): Total cost of all routes in the solution
        routes (list[Route]): List of vehicle routes
    """
    
    def __init__(self):
        self.cost = 0.0
        self.routes = []


class RelocationMove:
    """
    Represents a relocation move: moving a single node to a different position.
    
    The node can be relocated within the same route or to a different route.
    """
    
    def __init__(self):
        self.originRoutePosition = None
        self.targetRoutePosition = None
        self.originNodePosition = None
        self.targetNodePosition = None
        self.costChangeOriginRt = None
        self.costChangeTargetRt = None
        self.moveCost = None
        self.moveCost_penalized = None

    def Initialize(self):
        """Reset move to initial state for new evaluation."""
        self.originRoutePosition = None
        self.targetRoutePosition = None
        self.originNodePosition = None
        self.targetNodePosition = None
        self.costChangeOriginRt = None
        self.costChangeTargetRt = None
        self.moveCost = 10 ** 9
        self.moveCost_penalized = 10 ** 9


class SwapMove:
    """
    Represents a swap move: exchanging positions of two nodes.
    
    The nodes can be in the same route or in different routes.
    """
    
    def __init__(self):
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.costChangeFirstRt = None
        self.costChangeSecondRt = None
        self.moveCost = None
        self.moveCost_penalized = None

    def Initialize(self):
        """Reset move to initial state for new evaluation."""
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.costChangeFirstRt = None
        self.costChangeSecondRt = None
        self.moveCost = 10 ** 9
        self.moveCost_penalized = 10 ** 9


class TwoOptMove:
    """
    Represents a 2-opt move: reversing a segment of the route or 
    exchanging tails between two routes.
    """
    
    def __init__(self):
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.moveCost = None
        self.moveCost_penalized = None

    def Initialize(self):
        """Reset move to initial state for new evaluation."""
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.moveCost = 10 ** 9
        self.moveCost_penalized = 10 ** 9


class FamilySwapMove:
    """
    Represents a family swap move: swapping a visited node with an 
    unvisited node from the same family.
    
    This move is specific to F-CVRP and allows exploring alternative 
    family member selections while maintaining family requirements.
    """
    
    def __init__(self):
        self.unvisitedNode = None
        self.visitedNode = None
        self.insertRouteIndex = None
        self.insertPosition = None
        self.removeRouteIndex = None
        self.removePosition = None
        self.moveCost = None

    def Initialize(self):
        """Reset move to initial state for new evaluation."""
        self.unvisitedNode = None
        self.visitedNode = None
        self.insertRouteIndex = None
        self.insertPosition = None
        self.removeRouteIndex = None
        self.removePosition = None
        self.moveCost = 10 ** 9


class CustomerInsertionAllPositions:
    """Represents a customer insertion at a specific position in a route."""
    def __init__(self):
        self.customer = None
        self.route = None
        self.insertionPosition = None
        self.cost = 10 ** 9

class Solver:
    def __init__(self, m, 
                 # === GLS TUNING PARAMETERS ===
                 penalty_weight=0.03,       # Lambda: penalty weight (α≈0.1 for 200 nodes)
                 max_iterations=12000,     # Maximum GLS iterations
                 random_seed=2,            # Random seed for reproducibility
                 k_neighbors=40,           # Neighbors to check in FamilySwap
                 # === CONSTRUCTION HEURISTIC PARAMETERS ===
                 top_k=1):                 # Number of top candidates to consider in construction
        
        # Store tuning parameters
        self.penalty_weight = penalty_weight
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.k_neighbors = k_neighbors
        self.top_k = top_k
        
        # F_CVRP Model attributes
        self.num_nodes = m.num_nodes
        self.num_fam = m.num_fam
        self.num_req = m.num_req
        self.capacity = m.capacity
        self.vehicles = m.vehicles
        self.fam_members = m.fam_members
        self.fam_req = m.fam_req
        self.fam_visits = [0] * m.num_fam
        self.fam_demand = m.fam_demand
        self.cost_matrix = m.cost_matrix
        self.depot = m.depot
        self.nodes = m.nodes
        self.customers = m.customers
        self.families = m.families
        
        self.sol = None
        self.bestSolution = None
        
        # Penalized matrix for GLS
        rows = len(self.cost_matrix)
        self.cost_matrix_penalized = [[self.cost_matrix[i][j] for j in range(rows)] for i in range(rows)]
        self.times_penalized = [[0 for j in range(rows)] for i in range(rows)]
        self.penalized_n1_id = -1
        self.penalized_n2_id = -1
        
        # Pre-compute sorted neighbors for each node (for FamilySwap)
        self.sorted_adjacencies = []
        for i in range(rows):
            neighbors = []
            for j in range(rows):
                if i != j:
                    neighbors.append((j, self.cost_matrix[i][j]))
            neighbors.sort(key=lambda x: x[1])
            self.sorted_adjacencies.append([n[0] for n in neighbors])

    def solve(self):
        # Reset state
        for node in self.nodes:
            node.is_routed = False
        self.fam_visits = [0] * self.num_fam
        
        # Build initial solution - choose one:
        self.Nearest_Neighbor()
        #self.MinimumInsertions()  # Alternative construction heuristic
        print(f"Initial solution cost: {self.sol.cost:.2f}")
        
        # Apply Guided Local Search
        self.GuidedLocalSearch()
        
        print(f"Final solution cost: {self.bestSolution.cost:.2f}")
        self.ReportSolution()
        self.CreateFile("solution_gls.txt")
        return self.bestSolution

    def Nearest_Neighbor(self):
        """Creates initial solution using nearest neighbor heuristic with family constraints.
        
        If top_k > 1, randomly selects from top_k nearest nodes using random_seed.
        """
        random.seed(self.random_seed)
        self.sol = self.create_routes_from_depot()
        
        for i in range(self.num_req):
            self.insert_nearest_req_feasible_node()
        
        self.finish_routes_to_depot()

    def create_routes_from_depot(self):
        """Initialize empty routes starting from depot."""
        s = Solution()
        s.routes = [
            Route(i, [self.depot], self.capacity, 0, 0)
            for i in range(self.vehicles)
        ]
        return s

    def insert_nearest_req_feasible_node(self):
        """Find and insert a feasible node from top_k nearest candidates."""
        # Collect all feasible insertions with their costs
        candidate_insertions = []

        for rt in self.sol.routes:
            last_node_in_route = rt.sequence_of_nodes[-1]
            
            for i in range(len(last_node_in_route.costs)):
                cost = last_node_in_route.costs[i]
                node = self.nodes[i]
                
                if node.is_depot:
                    continue
                if node.is_routed:
                    continue
                if self.fam_visits[node.family.id] >= node.family.required:
                    continue
                if node.demand + rt.load > rt.capacity:
                    continue
                
                candidate_insertions.append((cost, node, rt))
        
        if len(candidate_insertions) > 0:
            # Sort by cost and select from top_k
            candidate_insertions.sort(key=lambda x: x[0])
            top_candidates = candidate_insertions[:min(self.top_k, len(candidate_insertions))]
            
            # Randomly select one from top_k
            selected_idx = random.randint(0, len(top_candidates) - 1)
            nearest_node_cost, nearest_node, rt_with_nearest_node = top_candidates[selected_idx]
            
            self.fam_visits[nearest_node.family.id] += 1
            rt_with_nearest_node.sequence_of_nodes.append(nearest_node)
            self.sol.cost += nearest_node_cost
            nearest_node.is_routed = True
            rt_with_nearest_node.cost += nearest_node_cost
            rt_with_nearest_node.load += nearest_node.demand

    def finish_routes_to_depot(self):
        """Add depot as final node to all routes."""
        for rt in self.sol.routes:
            if len(rt.sequence_of_nodes) > 1:  # Only if route has customers
                last_node = rt.sequence_of_nodes[-1]
                cost_to_depot = self.cost_matrix[last_node.id][self.depot.id]
                self.sol.cost += cost_to_depot
                rt.cost += cost_to_depot
            rt.sequence_of_nodes.append(self.depot)

    def MinimumInsertions(self):
        """Creates initial solution using minimum insertion heuristic.
        
        Uses self.top_k and self.random_seed from class attributes.
        """
        random.seed(self.random_seed)
        
        model_is_feasible = True
        self.sol = Solution()
        insertions = 0

        # Family CVRP: stop at num_req nodes
        while insertions < self.num_req:
            candidate_insertions = []
            self.Always_keep_an_empty_route()
            self.IdentifyMinimumCostInsertions(candidate_insertions)

            if len(candidate_insertions) > 0:
                # Select from top_k candidates
                selected_idx = random.randint(0, len(candidate_insertions) - 1)
                self.ApplyCustomerInsertionAllPositions(candidate_insertions[selected_idx])
                insertions += 1
            else:
                print('FeasibilityIssue')
                model_is_feasible = False
                break

        # Remove empty routes (depot->depot only)
        self.sol.routes = [rt for rt in self.sol.routes if len(rt.sequence_of_nodes) > 2]

    def Always_keep_an_empty_route(self):
        """Ensure there's always an empty route available for insertions."""
        if len(self.sol.routes) == 0:
            rt = Route(len(self.sol.routes), [self.depot, self.depot], self.capacity, 0, 0)
            self.sol.routes.append(rt)
        else:
            rt = self.sol.routes[-1]
            if len(rt.sequence_of_nodes) > 2:
                rt = Route(len(self.sol.routes), [self.depot, self.depot], self.capacity, 0, 0)
                self.sol.routes.append(rt)

    def IdentifyMinimumCostInsertions(self, candidate_insertions):
        """Identify top_k best insertions across all routes and positions.
        
        Uses self.top_k from class attributes.
        
        Args:
            candidate_insertions: List to populate with top_k best insertions
        """
        all_insertions = []
        for i in range(0, len(self.customers)):
            candidateCust = self.customers[i]
            if candidateCust.is_routed is False:
                # Family CVRP: check if family still needs nodes
                fam_id = candidateCust.family.id
                if self.fam_visits[fam_id] >= self.fam_req[fam_id]:
                    continue  # Skip - family already has enough nodes
                
                for rt in self.sol.routes:
                    if rt.load + candidateCust.demand <= rt.capacity:
                        for j in range(0, len(rt.sequence_of_nodes) - 1):
                            A = rt.sequence_of_nodes[j]
                            B = rt.sequence_of_nodes[j + 1]
                            costAdded = self.cost_matrix[A.id][candidateCust.id] + \
                                        self.cost_matrix[candidateCust.id][B.id]
                            # Handle depot->depot case (empty route)
                            if A.id == B.id:
                                costRemoved = 0  # No arc to remove when A == B
                            else:
                                costRemoved = self.cost_matrix[A.id][B.id]
                            trialCost = costAdded - costRemoved
                            
                            insertion = CustomerInsertionAllPositions()
                            insertion.customer = candidateCust
                            insertion.route = rt
                            insertion.insertionPosition = j
                            insertion.cost = trialCost
                            all_insertions.append(insertion)
        
        # Sort by cost and return top_k
        all_insertions.sort(key=lambda x: x.cost)
        candidate_insertions.extend(all_insertions[:self.top_k])

    def ApplyCustomerInsertionAllPositions(self, insertion):
        """Apply a customer insertion at a specific position."""
        insCustomer = insertion.customer
        rt = insertion.route
        # Insert at the specified position
        insIndex = insertion.insertionPosition
        rt.sequence_of_nodes.insert(insIndex + 1, insCustomer)
        rt.cost += insertion.cost
        self.sol.cost += insertion.cost
        rt.load += insCustomer.demand
        insCustomer.is_routed = True
        # Family CVRP: track family visits
        self.fam_visits[insCustomer.family.id] += 1


    def GuidedLocalSearch(self):
        """Guided Local Search with penalized distances."""
        random.seed(self.random_seed)
        self.bestSolution = self.cloneSolution(self.sol)
        terminationCondition = False
        localSearchIterator = 0
        last_printed_iter = -1

        rm = RelocationMove()
        sm = SwapMove()
        top = TwoOptMove()
        fsm = FamilySwapMove()

        while terminationCondition is False:
            # Uniform random operator selection (0-3: Relocation, Swap, 2-Opt, FamilySwap)
            operator = random.randint(0, 3)
            
            self.InitializeOperators(rm, sm, top, fsm)
            move_applied = False

            # Relocations
            if operator == 0:
                self.FindBestRelocationMove(rm)
                if rm.originRoutePosition is not None:
                    if rm.moveCost_penalized < 0:
                        self.ApplyRelocationMove(rm)
                        move_applied = True
                    else:
                        self.penalize_arcs()
                        localSearchIterator -= 1
            # Swaps
            elif operator == 1:
                self.FindBestSwapMove(sm)
                if sm.positionOfFirstRoute is not None:
                    if sm.moveCost_penalized < 0:
                        self.ApplySwapMove(sm)
                        move_applied = True
                    else:
                        self.penalize_arcs()
                        localSearchIterator -= 1
            # 2-Opt
            elif operator == 2:
                self.FindBestTwoOptMove(top)
                if top.positionOfFirstRoute is not None:
                    if top.moveCost_penalized < 0:
                        self.ApplyTwoOptMove(top)
                        move_applied = True
                    else:
                        self.penalize_arcs()
                        localSearchIterator -= 1
            # Family Swap
            elif operator == 3:
                self.FindBestFamilySwapMove(fsm)
                if fsm.unvisitedNode is not None:
                    if fsm.moveCost < 0:
                        self.ApplyFamilySwapMove(fsm)
                        move_applied = True
                    else:
                        self.penalize_arcs()
                        localSearchIterator -= 1


            if self.sol.cost < self.bestSolution.cost:
                self.bestSolution = self.cloneSolution(self.sol)
                print(f"Iter {localSearchIterator}: New best = {self.bestSolution.cost:.2f}")
                
                # Penalty Reset (Intensification): Forget old penalties to focus on the new best neighborhood
                rows = len(self.cost_matrix)
                for i in range(rows):
                    for j in range(rows):
                        self.times_penalized[i][j] = 0
                        self.cost_matrix_penalized[i][j] = self.cost_matrix[i][j]

            localSearchIterator += 1
            
            if localSearchIterator % 500 == 0 and localSearchIterator != last_printed_iter:
                print(f"Iter {localSearchIterator}: Current = {self.sol.cost:.2f}, Best = {self.bestSolution.cost:.2f}")
                last_printed_iter = localSearchIterator

            # Termination conditions
            if localSearchIterator >= self.max_iterations:
                terminationCondition = True
        

        # Final VND intensification on best solution
        self.sol = self.bestSolution
        print("Running final VND intensification...")
        self.VND()
        if self.sol.cost < self.bestSolution.cost:
            self.bestSolution = self.cloneSolution(self.sol)
            print(f"VND improved final solution to: {self.bestSolution.cost:.2f}")

    def VND(self):
        """
        Variable Neighborhood Descent - intensification procedure.
        Cycles through neighborhoods in order until no improvement is found.
        Order: Relocation -> Swap -> 2-Opt -> FamilySwap
        """
        improved = True
        rm = RelocationMove()
        sm = SwapMove()
        top = TwoOptMove()
        fsm = FamilySwapMove()
        
        while improved:
            improved = False
            
            # Neighborhood 1: Relocation
            while True:
                rm.Initialize()
                self.FindBestRelocationMove(rm)
                if rm.originRoutePosition is not None and rm.moveCost < 0:
                    self.ApplyRelocationMove(rm)
                    improved = True
                else:
                    break
            
            # Neighborhood 2: Swap
            while True:
                sm.Initialize()
                self.FindBestSwapMove(sm)
                if sm.positionOfFirstRoute is not None and sm.moveCost < 0:
                    self.ApplySwapMove(sm)
                    improved = True
                else:
                    break
            
            # Neighborhood 3: 2-Opt
            while True:
                top.Initialize()
                self.FindBestTwoOptMove(top)
                if top.positionOfFirstRoute is not None and top.moveCost < 0:
                    self.ApplyTwoOptMove(top)
                    improved = True
                else:
                    break
            
            # Neighborhood 4: Family Swap
            while True:
                fsm.Initialize()
                self.FindBestFamilySwapMove(fsm)
                if fsm.unvisitedNode is not None and fsm.moveCost < 0:
                    self.ApplyFamilySwapMove(fsm)
                    improved = True
                else:
                    break

    def InitializeOperators(self, rm, sm, top, fsm):
        rm.Initialize()
        sm.Initialize()
        top.Initialize()
        fsm.Initialize()

    def FindBestRelocationMove(self, rm):
        for originRouteIndex in range(len(self.sol.routes)):
            rt1 = self.sol.routes[originRouteIndex]
            for originNodeIndex in range(1, len(rt1.sequence_of_nodes) - 1):
                for targetRouteIndex in range(len(self.sol.routes)):
                    rt2 = self.sol.routes[targetRouteIndex]
                    for targetNodeIndex in range(len(rt2.sequence_of_nodes) - 1):
                        if originRouteIndex == targetRouteIndex and \
                           (targetNodeIndex == originNodeIndex or targetNodeIndex == originNodeIndex - 1):
                            continue

                        A = rt1.sequence_of_nodes[originNodeIndex - 1]
                        B = rt1.sequence_of_nodes[originNodeIndex]
                        C = rt1.sequence_of_nodes[originNodeIndex + 1]
                        F = rt2.sequence_of_nodes[targetNodeIndex]
                        G = rt2.sequence_of_nodes[targetNodeIndex + 1]

                        if rt1 != rt2:
                            if rt2.load + B.demand > rt2.capacity:
                                continue

                        costAdded = self.cost_matrix[A.id][C.id] + self.cost_matrix[F.id][B.id] + self.cost_matrix[B.id][G.id]
                        costRemoved = self.cost_matrix[A.id][B.id] + self.cost_matrix[B.id][C.id] + self.cost_matrix[F.id][G.id]
                        moveCost = costAdded - costRemoved

                        costAdded_pen = self.cost_matrix_penalized[A.id][C.id] + self.cost_matrix_penalized[F.id][B.id] + self.cost_matrix_penalized[B.id][G.id]
                        costRemoved_pen = self.cost_matrix_penalized[A.id][B.id] + self.cost_matrix_penalized[B.id][C.id] + self.cost_matrix_penalized[F.id][G.id]
                        moveCost_pen = costAdded_pen - costRemoved_pen

                        if moveCost_pen < rm.moveCost_penalized:
                            rm.originRoutePosition = originRouteIndex
                            rm.originNodePosition = originNodeIndex
                            rm.targetRoutePosition = targetRouteIndex
                            rm.targetNodePosition = targetNodeIndex
                            rm.costChangeOriginRt = self.cost_matrix[A.id][C.id] - self.cost_matrix[A.id][B.id] - self.cost_matrix[B.id][C.id]
                            rm.costChangeTargetRt = self.cost_matrix[F.id][B.id] + self.cost_matrix[B.id][G.id] - self.cost_matrix[F.id][G.id]
                            rm.moveCost = moveCost
                            rm.moveCost_penalized = moveCost_pen



    def FindBestSwapMove(self, sm):
        for firstRouteIndex in range(len(self.sol.routes)):
            rt1 = self.sol.routes[firstRouteIndex]
            for secondRouteIndex in range(firstRouteIndex, len(self.sol.routes)):
                rt2 = self.sol.routes[secondRouteIndex]
                for firstNodeIndex in range(1, len(rt1.sequence_of_nodes) - 1):
                    startOfSecondNodeIndex = 1
                    if rt1 == rt2:
                        startOfSecondNodeIndex = firstNodeIndex + 1
                    for secondNodeIndex in range(startOfSecondNodeIndex, len(rt2.sequence_of_nodes) - 1):
                        a1 = rt1.sequence_of_nodes[firstNodeIndex - 1]
                        b1 = rt1.sequence_of_nodes[firstNodeIndex]
                        c1 = rt1.sequence_of_nodes[firstNodeIndex + 1]
                        a2 = rt2.sequence_of_nodes[secondNodeIndex - 1]
                        b2 = rt2.sequence_of_nodes[secondNodeIndex]
                        c2 = rt2.sequence_of_nodes[secondNodeIndex + 1]

                        moveCost = None
                        moveCost_pen = None
                        costChangeFirstRoute = None
                        costChangeSecondRoute = None

                    if rt1 == rt2:
                        if firstNodeIndex == secondNodeIndex - 1:
                            costAdded = self.cost_matrix[a1.id][b2.id] + self.cost_matrix[b1.id][c2.id]
                            costRemoved = self.cost_matrix[a1.id][b1.id] + self.cost_matrix[b2.id][c2.id]
                            costAdded_pen = self.cost_matrix_penalized[a1.id][b2.id] + self.cost_matrix_penalized[b1.id][c2.id]
                            costRemoved_pen = self.cost_matrix_penalized[a1.id][b1.id] + self.cost_matrix_penalized[b2.id][c2.id]
                        elif firstNodeIndex == secondNodeIndex + 1:
                            costAdded = self.cost_matrix[a2.id][b1.id] + self.cost_matrix[b2.id][c1.id]
                            costRemoved = self.cost_matrix[a2.id][b2.id] + self.cost_matrix[b1.id][c1.id]
                            costAdded_pen = self.cost_matrix_penalized[a2.id][b1.id] + self.cost_matrix_penalized[b2.id][c1.id]
                            costRemoved_pen = self.cost_matrix_penalized[a2.id][b2.id] + self.cost_matrix_penalized[b1.id][c1.id]
                        else:
                            costAdded = self.cost_matrix[a1.id][b2.id] + self.cost_matrix[b2.id][c1.id] + self.cost_matrix[a2.id][b1.id] + self.cost_matrix[b1.id][c2.id]
                            costRemoved = self.cost_matrix[a1.id][b1.id] + self.cost_matrix[b1.id][c1.id] + self.cost_matrix[a2.id][b2.id] + self.cost_matrix[b2.id][c2.id]
                            costAdded_pen = self.cost_matrix_penalized[a1.id][b2.id] + self.cost_matrix_penalized[b2.id][c1.id] + self.cost_matrix_penalized[a2.id][b1.id] + self.cost_matrix_penalized[b1.id][c2.id]
                            costRemoved_pen = self.cost_matrix_penalized[a1.id][b1.id] + self.cost_matrix_penalized[b1.id][c1.id] + self.cost_matrix_penalized[a2.id][b2.id] + self.cost_matrix_penalized[b2.id][c2.id]
                    else:
                        costAdded = self.cost_matrix[a1.id][b2.id] + self.cost_matrix[b2.id][c1.id] + self.cost_matrix[a2.id][b1.id] + self.cost_matrix[b1.id][c2.id]
                        costRemoved = self.cost_matrix[a1.id][b1.id] + self.cost_matrix[b1.id][c1.id] + self.cost_matrix[a2.id][b2.id] + self.cost_matrix[b2.id][c2.id]
                        costAdded_pen = self.cost_matrix_penalized[a1.id][b2.id] + self.cost_matrix_penalized[b2.id][c1.id] + self.cost_matrix_penalized[a2.id][b1.id] + self.cost_matrix_penalized[b1.id][c2.id]
                        costRemoved_pen = self.cost_matrix_penalized[a1.id][b1.id] + self.cost_matrix_penalized[b1.id][c1.id] + self.cost_matrix_penalized[a2.id][b2.id] + self.cost_matrix_penalized[b2.id][c2.id]

                    moveCost = costAdded - costRemoved
                    moveCost_pen = costAdded_pen - costRemoved_pen

                    if moveCost_pen < sm.moveCost_penalized:
                        sm.positionOfFirstRoute = firstRouteIndex
                        sm.positionOfSecondRoute = secondRouteIndex
                        sm.positionOfFirstNode = firstNodeIndex
                        sm.positionOfSecondNode = secondNodeIndex
                        sm.costChangeFirstRt = self.cost_matrix[a1.id][b2.id] + self.cost_matrix[b2.id][c1.id] - self.cost_matrix[a1.id][b1.id] - self.cost_matrix[b1.id][c1.id]
                        sm.costChangeSecondRt = self.cost_matrix[a2.id][b1.id] + self.cost_matrix[b1.id][c2.id] - self.cost_matrix[a2.id][b2.id] - self.cost_matrix[b2.id][c2.id]
                        sm.moveCost = moveCost
                        sm.moveCost_penalized = moveCost_pen

    def FindBestTwoOptMove(self, top):
        for rtInd1 in range(len(self.sol.routes)):
            rt1 = self.sol.routes[rtInd1]
            for rtInd2 in range(rtInd1, len(self.sol.routes)):
                rt2 = self.sol.routes[rtInd2]
                for nodeInd1 in range(len(rt1.sequence_of_nodes) - 1):
                    start2 = 0
                    if rt1 == rt2:
                        start2 = nodeInd1 + 2

                    for nodeInd2 in range(start2, len(rt2.sequence_of_nodes) - 1):
                        moveCost = 10 ** 9
                        moveCost_pen = 10 ** 9

                        A = rt1.sequence_of_nodes[nodeInd1]
                        B = rt1.sequence_of_nodes[nodeInd1 + 1]
                        K = rt2.sequence_of_nodes[nodeInd2]
                        L = rt2.sequence_of_nodes[nodeInd2 + 1]

                        if rt1 == rt2:
                            if nodeInd1 == 0 and nodeInd2 == len(rt1.sequence_of_nodes) - 2:
                                continue
                            costAdded = self.cost_matrix[A.id][K.id] + self.cost_matrix[B.id][L.id]
                            costRemoved = self.cost_matrix[A.id][B.id] + self.cost_matrix[K.id][L.id]
                            costAdded_pen = self.cost_matrix_penalized[A.id][K.id] + self.cost_matrix_penalized[B.id][L.id]
                            costRemoved_pen = self.cost_matrix_penalized[A.id][B.id] + self.cost_matrix_penalized[K.id][L.id]
                            moveCost = costAdded - costRemoved
                            moveCost_pen = costAdded_pen - costRemoved_pen
                        else:
                            if nodeInd1 == 0 and nodeInd2 == 0:
                                continue
                            if nodeInd1 == len(rt1.sequence_of_nodes) - 2 and nodeInd2 == len(rt2.sequence_of_nodes) - 2:
                                continue

                            if self.CapacityIsViolated(rt1, nodeInd1, rt2, nodeInd2):
                                continue
                            costAdded = self.cost_matrix[A.id][L.id] + self.cost_matrix[B.id][K.id]
                            costRemoved = self.cost_matrix[A.id][B.id] + self.cost_matrix[K.id][L.id]
                            costAdded_pen = self.cost_matrix_penalized[A.id][L.id] + self.cost_matrix_penalized[B.id][K.id]
                            costRemoved_pen = self.cost_matrix_penalized[A.id][B.id] + self.cost_matrix_penalized[K.id][L.id]
                            moveCost = costAdded - costRemoved
                            moveCost_pen = costAdded_pen - costRemoved_pen

                        if moveCost_pen < top.moveCost_penalized:
                            top.positionOfFirstRoute = rtInd1
                            top.positionOfSecondRoute = rtInd2
                            top.positionOfFirstNode = nodeInd1
                            top.positionOfSecondNode = nodeInd2
                            top.moveCost = moveCost
                            top.moveCost_penalized = moveCost_pen

    def CapacityIsViolated(self, rt1, nodeInd1, rt2, nodeInd2):
        rt1FirstSegmentLoad = sum(n.demand for n in rt1.sequence_of_nodes[:nodeInd1 + 1])
        rt1SecondSegmentLoad = rt1.load - rt1FirstSegmentLoad
        rt2FirstSegmentLoad = sum(n.demand for n in rt2.sequence_of_nodes[:nodeInd2 + 1])
        rt2SecondSegmentLoad = rt2.load - rt2FirstSegmentLoad

        if rt1FirstSegmentLoad + rt2SecondSegmentLoad > rt1.capacity:
            return True
        if rt2FirstSegmentLoad + rt1SecondSegmentLoad > rt2.capacity:
            return True
        return False

    def ApplyRelocationMove(self, rm):
        originRt = self.sol.routes[rm.originRoutePosition]
        targetRt = self.sol.routes[rm.targetRoutePosition]
        B = originRt.sequence_of_nodes[rm.originNodePosition]

        if originRt == targetRt:
            del originRt.sequence_of_nodes[rm.originNodePosition]
            if rm.originNodePosition < rm.targetNodePosition:
                targetRt.sequence_of_nodes.insert(rm.targetNodePosition, B)
            else:
                targetRt.sequence_of_nodes.insert(rm.targetNodePosition + 1, B)
            originRt.cost += rm.moveCost
        else:
            del originRt.sequence_of_nodes[rm.originNodePosition]
            targetRt.sequence_of_nodes.insert(rm.targetNodePosition + 1, B)
            originRt.cost += rm.costChangeOriginRt
            targetRt.cost += rm.costChangeTargetRt
            originRt.load -= B.demand
            targetRt.load += B.demand

        self.sol.cost += rm.moveCost



    def ApplySwapMove(self, sm):
        rt1 = self.sol.routes[sm.positionOfFirstRoute]
        rt2 = self.sol.routes[sm.positionOfSecondRoute]
        b1 = rt1.sequence_of_nodes[sm.positionOfFirstNode]
        b2 = rt2.sequence_of_nodes[sm.positionOfSecondNode]
        rt1.sequence_of_nodes[sm.positionOfFirstNode] = b2
        rt2.sequence_of_nodes[sm.positionOfSecondNode] = b1

        if rt1 == rt2:
            rt1.cost += sm.moveCost
        else:
            rt1.cost += sm.costChangeFirstRt
            rt2.cost += sm.costChangeSecondRt
            rt1.load = rt1.load - b1.demand + b2.demand
            rt2.load = rt2.load + b1.demand - b2.demand

        self.sol.cost += sm.moveCost

    def ApplyTwoOptMove(self, top):
        rt1 = self.sol.routes[top.positionOfFirstRoute]
        rt2 = self.sol.routes[top.positionOfSecondRoute]

        if rt1 == rt2:
            reversedSegment = list(reversed(rt1.sequence_of_nodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1]))
            rt1.sequence_of_nodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1] = reversedSegment
            rt1.cost += top.moveCost
        else:
            relocatedSegmentOfRt1 = rt1.sequence_of_nodes[top.positionOfFirstNode + 1:]
            relocatedSegmentOfRt2 = rt2.sequence_of_nodes[top.positionOfSecondNode + 1:]

            del rt1.sequence_of_nodes[top.positionOfFirstNode + 1:]
            del rt2.sequence_of_nodes[top.positionOfSecondNode + 1:]

            rt1.sequence_of_nodes.extend(relocatedSegmentOfRt2)
            rt2.sequence_of_nodes.extend(relocatedSegmentOfRt1)

            self.UpdateRouteCostAndLoad(rt1)
            self.UpdateRouteCostAndLoad(rt2)

        self.sol.cost += top.moveCost

    def UpdateRouteCostAndLoad(self, rt):
        tc = 0
        tl = 0
        for i in range(len(rt.sequence_of_nodes) - 1):
            A = rt.sequence_of_nodes[i]
            B = rt.sequence_of_nodes[i + 1]
            tc += self.cost_matrix[A.id][B.id]
            tl += A.demand
        rt.load = tl
        rt.cost = tc

    def penalize_arcs(self):
        """Penalize the most costly arc that hasn't been penalized much."""
        max_criterion = 0
        pen_1 = -1
        pen_2 = -1
        
        for rt in self.sol.routes:
            for j in range(len(rt.sequence_of_nodes) - 1):
                id1 = rt.sequence_of_nodes[j].id
                id2 = rt.sequence_of_nodes[j + 1].id
                criterion = self.cost_matrix[id1][id2] / (1 + self.times_penalized[id1][id2])
                if criterion > max_criterion:
                    max_criterion = criterion
                    pen_1 = id1
                    pen_2 = id2
        
        if pen_1 >= 0 and pen_2 >= 0:
            self.times_penalized[pen_1][pen_2] += 1
            self.times_penalized[pen_2][pen_1] += 1

            self.cost_matrix_penalized[pen_1][pen_2] = (1 + self.penalty_weight * self.times_penalized[pen_1][pen_2]) * self.cost_matrix[pen_1][pen_2]
            self.cost_matrix_penalized[pen_2][pen_1] = (1 + self.penalty_weight * self.times_penalized[pen_2][pen_1]) * self.cost_matrix[pen_2][pen_1]


    # ============================================================
    # FAMILY SWAP LOGIC
    # ============================================================

    def FindBestFamilySwapMove(self, fsm):
        """
        Find the best Family Swap move: swap a visited node with an unvisited 
        node from the same family.
        """
        visited_nodes_info = {}
        node_location = {}
        
        for rt_idx, route in enumerate(self.sol.routes):
            for pos, node in enumerate(route.sequence_of_nodes):
                if not node.is_depot:
                    visited_nodes_info[node.id] = (rt_idx, pos)
                node_location[node.id] = (rt_idx, pos)

        visited_ids = set(visited_nodes_info.keys())
        unvisited_per_family = {}
        for family in self.families:
            unvisited_per_family[family.id] = [n for n in family.nodes if n.id not in visited_ids]

        u_best_insertions = {}
        for fam_id, u_nodes in unvisited_per_family.items():
            for U in u_nodes:
                best_u_cost = float('inf')
                best_u_rt = -1
                best_u_pos = -1
                neighbors = self.sorted_adjacencies[U.id]
                checked_positions = set()
                count = 0
                for neighbor_id in neighbors:
                    if count >= self.k_neighbors:
                        break
                    if neighbor_id not in node_location:
                        continue
                    
                    n_rt_idx, n_pos = node_location[neighbor_id]
                    route = self.sol.routes[n_rt_idx]
                    
                    # Try inserting BEFORE and AFTER the neighbor
                    for pos in [n_pos, n_pos + 1]:
                        if pos < 1 or pos >= len(route.sequence_of_nodes):
                            continue
                        if (n_rt_idx, pos) in checked_positions:
                            continue
                        checked_positions.add((n_rt_idx, pos))
                        
                        if route.load + U.demand > self.capacity:
                            continue
                        
                        cost = self._calculate_insertion_cost(route, pos, U)
                        if cost < best_u_cost:
                            best_u_cost = cost
                            best_u_rt = n_rt_idx
                            best_u_pos = pos
                    count += 1


                if best_u_rt >= 0:
                    u_best_insertions[U.id] = (best_u_cost, best_u_rt, best_u_pos)

        # Evaluate Moves
        for family in self.families:
            visited_in_family = [n for n in family.nodes if n.id in visited_ids]
            unvisited_in_family = unvisited_per_family.get(family.id, [])

            if not unvisited_in_family or not visited_in_family:
                continue

            for V in visited_in_family:
                v_rt_idx, v_pos = visited_nodes_info[V.id]
                v_route = self.sol.routes[v_rt_idx]

                P = v_route.sequence_of_nodes[v_pos - 1]
                Q = v_route.sequence_of_nodes[v_pos + 1]
                cost_removed_V = (self.cost_matrix[P.id][V.id] + 
                                  self.cost_matrix[V.id][Q.id] - 
                                  self.cost_matrix[P.id][Q.id])

                for U in unvisited_in_family:
                    # Different Route case
                    if U.id in u_best_insertions:
                        best_u_cost, best_u_rt, best_u_pos = u_best_insertions[U.id]
                        if best_u_rt != v_rt_idx:
                            move_cost = best_u_cost - cost_removed_V

                            if move_cost < fsm.moveCost:
                                fsm.unvisitedNode = U
                                fsm.visitedNode = V
                                fsm.insertRouteIndex = best_u_rt
                                fsm.insertPosition = best_u_pos
                                fsm.removeRouteIndex = v_rt_idx
                                fsm.removePosition = v_pos
                                fsm.moveCost = move_cost

                    # Same Route case
                    if v_route.load - V.demand + U.demand <= self.capacity:
                        cost_u_same_route = self._calculate_same_route_swap_cost(v_route, v_pos, U)
                        move_cost_same = cost_u_same_route - cost_removed_V

                        if move_cost_same < fsm.moveCost:
                            best_pos = self._find_best_pos_in_modified_route(v_route, v_pos, U)
                            fsm.unvisitedNode = U
                            fsm.visitedNode = V
                            fsm.insertRouteIndex = v_rt_idx
                            fsm.insertPosition = best_pos
                            fsm.removeRouteIndex = v_rt_idx
                            fsm.removePosition = v_pos
                            fsm.moveCost = move_cost_same

    def _calculate_insertion_cost(self, route, pos, U):
        """Calculate cost of inserting U at position pos in route."""
        A = route.sequence_of_nodes[pos - 1]
        B = route.sequence_of_nodes[pos]
        return (self.cost_matrix[A.id][U.id] + 
                self.cost_matrix[U.id][B.id] - 
                self.cost_matrix[A.id][B.id])

    def _find_best_pos_in_modified_route(self, route, v_pos, U):
        """Find best position to insert U in route after V is removed."""
        best_cost = float('inf')
        best_pos = -1
        seq = route.sequence_of_nodes
        effective_nodes = seq[:v_pos] + seq[v_pos+1:]
        
        for i in range(1, len(effective_nodes)):
            A = effective_nodes[i-1]
            B = effective_nodes[i]
            cost_added = (self.cost_matrix[A.id][U.id] + 
                          self.cost_matrix[U.id][B.id] - 
                          self.cost_matrix[A.id][B.id])
            
            if cost_added < best_cost:
                best_cost = cost_added
                best_pos = i
        return best_pos

    def _calculate_same_route_swap_cost(self, route, v_pos, U):
        """Calculate cost of swapping V with U in same route."""
        pos = self._find_best_pos_in_modified_route(route, v_pos, U)
        if pos == -1:
            return float('inf')
        seq = route.sequence_of_nodes
        effective_nodes = seq[:v_pos] + seq[v_pos+1:]
        A = effective_nodes[pos-1]
        B = effective_nodes[pos]
        return (self.cost_matrix[A.id][U.id] + 
                self.cost_matrix[U.id][B.id] - 
                self.cost_matrix[A.id][B.id])

    def ApplyFamilySwapMove(self, fsm):
        """Apply the Family Swap move."""
        U = fsm.unvisitedNode
        V = fsm.visitedNode
        insert_route = self.sol.routes[fsm.insertRouteIndex]
        remove_route = self.sol.routes[fsm.removeRouteIndex]

        same_route = (fsm.insertRouteIndex == fsm.removeRouteIndex)

        if same_route:
            if fsm.insertPosition == fsm.removePosition or fsm.insertPosition == fsm.removePosition + 1:
                insert_route.sequence_of_nodes[fsm.removePosition] = U
            else:
                insert_route.sequence_of_nodes.pop(fsm.removePosition)
                adj_pos = fsm.insertPosition - 1 if fsm.insertPosition > fsm.removePosition else fsm.insertPosition
                insert_route.sequence_of_nodes.insert(adj_pos, U)
            self.UpdateRouteCostAndLoad(insert_route)
        else:
            remove_route.sequence_of_nodes.pop(fsm.removePosition)
            self.UpdateRouteCostAndLoad(remove_route)
            insert_route.sequence_of_nodes.insert(fsm.insertPosition, U)
            self.UpdateRouteCostAndLoad(insert_route)

        # Recalculate solution cost
        self.sol.cost = sum(rt.cost for rt in self.sol.routes)
        U.is_routed = True
        V.is_routed = False



    def cloneRoute(self, rt):
        sequence_copy = rt.sequence_of_nodes[:]
        cloned = Route(rt.id, sequence_copy, rt.capacity, rt.cost, rt.load)
        return cloned

    def cloneSolution(self, sol):
        cloned = Solution()
        for rt in sol.routes:
            clonedRoute = self.cloneRoute(rt)
            cloned.routes.append(clonedRoute)
        cloned.cost = sol.cost
        return cloned

    def CalculateTotalCost(self):
        c = 0
        for rt in self.sol.routes:
            for j in range(len(rt.sequence_of_nodes) - 1):
                a = rt.sequence_of_nodes[j]
                b = rt.sequence_of_nodes[j + 1]
                c += self.cost_matrix[a.id][b.id]
        return c

    def ReportSolution(self):
        print("\n" + "="*60)
        print("SOLUTION REPORT")
        print("="*60)
        for i, rt in enumerate(self.bestSolution.routes):
            if len(rt.sequence_of_nodes) > 2:  # Only print non-empty routes
                route_str = " -> ".join(str(n.id) for n in rt.sequence_of_nodes)
                print(f"Route {i}: {route_str} (Cost: {rt.cost:.2f}, Load: {rt.load})")
        print(f"\nTotal Cost: {self.bestSolution.cost:.2f}")
        print("="*60)

    def CreateFile(self, filename):
        with open(filename, "w") as f:
            f.write(f"Cost: {self.bestSolution.cost}\n")
            for rt in self.bestSolution.routes:
                seq = [node.id for node in rt.sequence_of_nodes]
                f.write("-".join(str(x) for x in seq) + "\n")
