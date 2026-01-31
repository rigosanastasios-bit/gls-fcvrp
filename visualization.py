"""
Solution Visualization for Family-CVRP

This module provides visualization functions for F-CVRP solutions using
matplotlib. It projects the distance matrix into 2D coordinates using 
Multidimensional Scaling (MDS) and renders routes with color-coded families.

Functions:
    visualize_solution: Main visualization with routes and family coloring
    visualize_families_summary: Bar chart of family visit requirements

Author: Anastasios Rigos
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
from sklearn.manifold import MDS

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def visualize_solution(model, sol, title="F-CVRP Vehicle Routing Solution", show_arrows=True, save_path=None):
    """
    Visualizes the VRP solution with family-based coloring.
    - Uses MDS to project the cost matrix into 2D space.
    - Draws routes with distinct colors.
    - Draws VISITED nodes with their family color (solid).
    - Draws UNVISITED nodes with their family color (faded/hollow).
    """

    # 1. Setup the figure - IMPROVED DIMENSIONS
    fig, ax = plt.subplots(figsize=(20, 14))

    # Style configurations
    depot_color = 'black'
    depot_size = 400
    node_size = 60
    visited_node_size = 100
    text_offset = 0.02  # Reduced offset for better scaling

    # 2. Create coordinate positions from the cost matrix using MDS
    matrix = np.array(model.cost_matrix)
    num_actual_nodes = len(matrix)

    # Pre-process matrix: Replace large penalties with reasonable values for visualization
    valid_mask = matrix < 1000000
    if np.any(valid_mask):
        max_val = np.max(matrix[valid_mask])
        matrix_vis = np.where(matrix >= 1000000, max_val * 1.5, matrix)
    else:
        matrix_vis = matrix

    # Ensure symmetry for MDS
    matrix_vis = (matrix_vis + matrix_vis.T) / 2

    mds = MDS(n_components=2, metric=True, random_state=42, 
              max_iter=2000, eps=1e-9, n_init=8, dissimilarity='precomputed')
    coords = mds.fit_transform(matrix_vis)

    # Normalize coordinates
    coords = coords - coords.mean(axis=0)
    max_range = np.abs(coords).max()
    if max_range > 0:
        coords = coords / max_range

    # Map Node ID to (x, y)
    node_positions = {i: (coords[i, 0], coords[i, 1]) for i in range(num_actual_nodes)}

    # 3. Identify Visited Nodes
    visited_node_ids = {model.depot.id}
    for route in sol.routes:
        for node in route.sequence_of_nodes:
            visited_node_ids.add(node.id)

    # 4. Create family color palette
    num_families = model.num_fam
    if num_families <= 10:
        family_cmap = matplotlib.colormaps['tab10']
    elif num_families <= 20:
        family_cmap = matplotlib.colormaps['tab20']
    else:
        family_cmap = matplotlib.colormaps['hsv']

    family_colors = {i: family_cmap(i / max(1, num_families - 1)) for i in range(num_families)}

    # 5. Plot Depot
    dx, dy = node_positions[model.depot.id]
    ax.scatter(dx, dy, c=depot_color, s=depot_size, marker='s', edgecolors='gold',
               linewidths=3, label='DEPOT', zorder=10)
    ax.text(dx, dy + text_offset * 3, "DEPOT", fontweight='bold', ha='center',
            va='bottom', fontsize=12, zorder=10)

    # 6. Plot all nodes by family
    for node in model.nodes:
        if node.is_depot:
            continue
            
        node_id = node.id
        family_id = node.family.id
        color = family_colors[family_id]
        x, y = node_positions[node_id]

        if node_id in visited_node_ids:
            # Visited node: solid color, black edge
            ax.scatter(x, y, c=[color], s=visited_node_size, edgecolors='black',
                       linewidths=1.5, zorder=5, alpha=1.0)
            ax.text(x, y + text_offset, str(node_id), fontweight='bold',
                    ha='center', va='bottom', fontsize=7, zorder=6)
        else:
            # Unvisited node: faded color
            ax.scatter(x, y, c=[color], s=node_size, edgecolors='gray',
                       linewidths=0.5, zorder=2, alpha=0.15)

    # 7. Plot Routes
    route_colors = ['#E63946', '#2A9D8F', '#E9C46A', '#264653', '#F4A261', '#9B59B6', '#3498DB', '#27AE60', '#E74C3C']

    for r_idx, route in enumerate(sol.routes):
        route_color = route_colors[r_idx % len(route_colors)]
        
        # Build path
        path_ids = [n.id for n in route.sequence_of_nodes]
        path_coords = np.array([node_positions[pid] for pid in path_ids])

        # Draw Line
        ax.plot(path_coords[:, 0], path_coords[:, 1], color=route_color, 
                linewidth=2.5, alpha=0.8, zorder=4)

        if show_arrows:
            for i in range(len(path_coords) - 1):
                start = path_coords[i]
                end = path_coords[i+1]
                if np.linalg.norm(end - start) > 0.02:
                    ax.annotate('', xy=end, xytext=start,
                                arrowprops=dict(arrowstyle='-|>', color=route_color, 
                                lw=1.5, mutation_scale=12), zorder=4)

        # Legend entry for route
        ax.plot([], [], color=route_color, linewidth=3, 
                label=f"Route {route.id}: Cost {route.cost:.0f}, Load {route.load}")

    # 8. Create Family Legend
    family_handles = []
    for fam_id in range(num_families):
        color = family_colors[fam_id]
        visited = sum(1 for n in model.families[fam_id].nodes if n.id in visited_node_ids)
        req = model.families[fam_id].required
        total = len(model.families[fam_id].nodes)
        label = f"Fam {fam_id}: {visited}/{req} (of {total})"
        family_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=color, markersize=10, label=label))

    # 9. Legends and Formatting
    # Route legend (left)
    leg1 = ax.legend(loc='upper left', title="Routes", fontsize=10, framealpha=0.9,
                     title_fontsize=12)
    
    # Family legend (right) - in 2 columns for space
    leg_fam = ax.legend(handles=family_handles, loc='upper right', title="Families", 
                        fontsize=8, ncol=2, framealpha=0.9, title_fontsize=10)
    ax.add_artist(leg1)

    ax.set_title(f"{title}\nTotal Cost: {sol.cost:.2f} | Visited: {len(visited_node_ids)-1}/{model.num_req} required", 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Clean up axes
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_families_summary(model, sol, save_path=None):
    """Simple bar chart summary of family requirements."""
    visited_ids = {n.id for r in sol.routes for n in r.sequence_of_nodes}
    
    fams = model.families
    ids = [f.id for f in fams]
    visited = [sum(1 for n in f.nodes if n.id in visited_ids) for f in fams]
    required = [f.required for f in fams]
    
    x = np.arange(len(ids))
    width = 0.4
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, visited, width, label='Visited', color='forestgreen', alpha=0.8)
    ax.bar(x + width/2, required, width, label='Required', color='royalblue', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.set_title("Family Visit Requirements vs Actual", fontsize=14)
    ax.set_xlabel("Family ID")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()