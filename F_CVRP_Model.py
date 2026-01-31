"""
Family-CVRP Data Model

This module defines the core data structures for the Family Capacitated 
Vehicle Routing Problem (F-CVRP), including nodes, families, routes, 
and the problem model.

Classes:
    Node: Represents a customer or depot location
    Family: Represents a group of customers with shared requirements
    Route: Represents a vehicle route with its sequence and constraints
    Model: Represents the complete F-CVRP instance

Functions:
    create_model: Factory function to load a model from an instance file
"""


class Node:
    """
    Represents a single location (customer or depot) in the F-CVRP.
    
    Attributes:
        id (int): Unique identifier for the node
        family (Family): The family this node belongs to (None for depot)
        costs (list): Distance/cost to reach every other node
        demand (int): Demand to be fulfilled at this node
        dist_from_depot (float): Distance from the depot (costs[0])
        mean_dist (float): Sum of distances to all nodes
        is_depot (bool): Whether this node is the depot
        is_routed (bool): Whether this node is currently in a route
        route (Route): Reference to the route containing this node
        is_customer (bool): Whether this node is a customer (not depot)
        position_in_route (int): Index position within the route
        isTabuTillIterator (int): Tabu tenure (for Tabu Search variants)
    """
    
    def __init__(self, node_id, family, costs, demand):
        self.id = node_id
        self.family = family
        self.costs = costs
        self.demand = demand
        self.dist_from_depot = costs[0]
        self.mean_dist = sum(costs)
        self.is_depot = False
        self.is_routed = False
        self.route = None
        self.is_customer = True
        self.position_in_route = None
        self.isTabuTillIterator = 0


class Family:
    """
    Represents a family of customers in the F-CVRP.
    
    In F-CVRP, customers are grouped into families, and each family has a 
    required number of visits. The solver must select which family members 
    to visit to satisfy the requirement.
    
    Attributes:
        id (int): Unique identifier for the family
        nodes (list[Node]): List of customer nodes belonging to this family
        demand (int): Demand per visit (same for all family members)
        required (int): Number of required visits to satisfy this family
    """
    
    def __init__(self, family_id, nodes, demand, required):
        self.id = family_id
        self.nodes = nodes
        self.demand = demand
        self.required = required


class Route:
    """
    Represents a vehicle route in the solution.
    
    A route is a sequence of nodes starting and ending at the depot,
    with a total load not exceeding the vehicle capacity.
    
    Attributes:
        id (int): Unique identifier for the route
        sequence_of_nodes (list[Node]): Ordered list of nodes to visit
        capacity (int): Maximum load capacity for this route
        cost (float): Total travel cost of this route
        load (int): Current total demand served by this route
    """
    
    def __init__(self, route_id, sequence_of_nodes, capacity, cost, load):
        self.id = route_id
        self.sequence_of_nodes = sequence_of_nodes
        self.capacity = capacity
        self.cost = cost
        self.load = load


class Model:
    """
    Represents a complete Family-CVRP problem instance.
    
    Attributes:
        num_nodes (int): Total number of nodes including depot
        num_fam (int): Number of customer families
        num_req (int): Total number of required visits
        capacity (int): Vehicle capacity
        vehicles (int): Number of available vehicles
        fam_members (list[int]): Number of members in each family
        fam_req (list[int]): Required visits for each family
        fam_demand (list[int]): Demand for each family
        cost_matrix (list[list[int]]): Distance/cost matrix between nodes
        depot (Node): The depot node
        nodes (list[Node]): All nodes (depot + customers)
        customers (list[Node]): Customer nodes only
        families (list[Family]): All family objects
    """
    
    def __init__(self):
        self.num_nodes = 0
        self.num_fam = 0
        self.num_req = 0
        self.capacity = 0
        self.vehicles = 0
        self.fam_members = []
        self.fam_req = []
        self.fam_demand = []
        self.cost_matrix = []
        self.depot = None
        self.nodes = []
        self.customers = []
        self.families = []


def create_model(file_path):
    model = Model()
    try:
        with open(file_path, 'r') as sr:
            # 1st line
            line = sr.readline()
            parts = line.split()
            model.num_nodes = int(parts[0])
            model.num_fam = int(parts[1])
            model.num_req = int(parts[2])
            model.capacity = int(parts[3])
            model.vehicles = int(parts[4])

            # 2nd line
            line = sr.readline()
            parts = [p for p in line.split() if p]
            model.fam_members = [int(part) for part in parts]

            # 3rd line
            line = sr.readline()
            parts = [p for p in line.split() if p]
            model.fam_req = [int(part) for part in parts]

            # 4th line
            line = sr.readline()
            parts = [p for p in line.split() if p]
            model.fam_demand = [int(part) for part in parts]

            # 5th line onwards (cost matrix)
            cost_matrix = []
            for line in sr:
                if not line.strip():
                    continue
                node_costs = []
                parts = [p for p in line.split() if p]
                for part in parts:
                    cost = int(part)
                    if cost < 0:
                        node_costs.append(10000)
                    else:
                        node_costs.append(cost)
                cost_matrix.append(node_costs)
            model.cost_matrix = cost_matrix

        return _create_nodes_families(model)

    except Exception as e:
        print(f"Exception: {e}")
        return Model()


def _create_nodes_families(model):
    families = []
    nodes = []
    customers = []

    # Family initialization
    for i in range(model.num_fam):
        family = Family(i, [], model.fam_demand[i], model.fam_req[i])
        families.append(family)

    # Depot initialization
    depot = Node(0, None, model.cost_matrix[0], 0)
    depot.is_depot = True
    depot.is_customer = False  # Depot is not a customer
    nodes.append(depot)
    model.depot = depot

    # Nodes and customers initialization
    for i in range(1, len(model.cost_matrix)):
        fam_index = _find_node_family(model, i)
        node = Node(i, families[fam_index], model.cost_matrix[i], families[fam_index].demand)
        nodes.append(node)
        customers.append(node)

    # Add customer nodes to families
    for customer in customers:
        families[customer.family.id].nodes.append(customer)

    model.families = families
    model.nodes = nodes
    model.customers = customers

    return model


def _find_node_family(model, node_id):
    c = 0
    prev = 0
    for i in model.fam_members:
        if node_id <= i + prev:
            return c
        else:
            prev = prev + i
            c += 1
    return c