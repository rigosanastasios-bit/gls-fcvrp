"""
Guided Local Search (GLS) Solver for Family-CVRP

This script runs the GLS algorithm on the F-CVRP instance
and generates a visualization of the solution.

Usage:
    python main_gls.py
"""

from F_CVRP_Model import create_model
from Solver_gls import Solver
from visualization import visualize_solution


def main():
    # Load problem instance
    model = create_model("instance.txt")
    
    # Initialize and run solver
    solver = Solver(model)
    solution = solver.solve()
    
    # Visualize the solution
    visualize_solution(
        model, 
        solution, 
        title="GLS F-CVRP Solution", 
        show_arrows=True, 
        save_path="my_routes_gls.png"
    )


if __name__ == "__main__":
    main()
