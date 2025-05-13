import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from ortools.sat.python import cp_model  # For CP-SAT
from ortools.linear_solver import pywraplp # For MIP
import time
import numpy as np
import os

class Cell3D:
    def __init__(self, x, y, z=0): # z is the primary height value
        self.x = x
        self.y = y
        self.z = z # This will hold the final merged Z-value assigned by the solver
        self.value = 0 # Temporary value from simulation used as target/bias in solver
        self.links = set() # Set of linked neighbor Cell3D objects
        self.neighbors = [] # List of adjacent Cell3D objects (N,S,E,W)

    def link(self, other, bidirectional=True):
        self.links.add(other)
        if bidirectional:
            other.link(self, False)

    def is_linked(self, other):
        return other in self.links

    def __repr__(self):
        return f"Cell({self.x},{self.y}, z={self.z}, val={self.value})" # Show both for debug

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if not isinstance(other, Cell3D):
            return NotImplemented
        return (self.x, self.y) == (other.x, other.y)

class Grid3D:
    def __init__(self, width, height, depth=1):
        self.width = width
        self.height = height
        self.depth = depth # Still mostly works in 2D (z=0) for links/sim
        self.grid = self.prepare_grid()
        self.configure_neighbors()
        self.last_solver_type_used = None # Track which solver was last called

    def prepare_grid(self):
        # Creates cells, z will be overwritten by solver
        return [[[Cell3D(x, y, z=0) for x in range(self.width)]
                 for y in range(self.height)]
                 for z in range(self.depth)] # Note: Current logic focuses on z=0

    def configure_neighbors(self):
        # Configures N, S, E, W neighbors for the layer z=0
        if self.depth > 0:
            z = 0 # Assuming operations are primarily on the base layer
            for y in range(self.height):
                for x in range(self.width):
                    cell = self.grid[z][y][x]
                    potential_neighbors = []
                    # North
                    if y > 0: potential_neighbors.append(self.grid[z][y - 1][x])
                    # South
                    if y < self.height - 1: potential_neighbors.append(self.grid[z][y + 1][x])
                    # West
                    if x > 0: potential_neighbors.append(self.grid[z][y][x - 1])
                    # East
                    if x < self.width - 1: potential_neighbors.append(self.grid[z][y][x + 1])
                    cell.neighbors = potential_neighbors

    def each_cell(self):
        # Iterates over cells in the base layer (z=0)
        if self.depth > 0:
            z = 0
            for row in self.grid[z]:
                for cell in row:
                    yield cell

    def get_cell(self, x, y, z=0):
        # Gets a cell, ensuring it's within bounds (primarily for z=0)
        if 0 <= z < self.depth and 0 <= y < self.height and 0 <= x < self.width:
            return self.grid[z][y][x]
        return None

    # --- Simulation Methods ---
    def initialize_values(self, initial_value=0):
        for cell in self.each_cell():
            cell.value = initial_value

    def run_simulation_step(self):
        increments = {cell: 0 for cell in self.each_cell()}
        changed = False
        for cell in self.each_cell():
            # Propagate value *only* through actual maze links
            for neighbor in cell.links:
                 if neighbor in increments: # Check if neighbor is part of the simulation domain
                    increments[neighbor] += 1 # Increment the neighbor's count
        # Apply increments
        for cell in self.each_cell():
             increment_amount = increments.get(cell, 0) # Use .get for safety
             if increment_amount > 0:
                new_value = cell.value + increment_amount
                if cell.value != new_value: # Check if value actually changes
                    cell.value = new_value
                    changed = True
        return changed

    def run_full_simulation(self, max_iterations=100, print_steps=False):
        print("\n--- Running Value Propagation Simulation (to generate targets) ---")
        self.initialize_values(0)
        corner_cell = self.get_cell(0, 0)
        if corner_cell:
            corner_cell.value = 1
            print("Seeding simulation value=1 at cell (0,0)")

        start_total_time = time.time()
        for i in range(max_iterations):
            step_start_time = time.time()
            changed = self.run_simulation_step()
            step_end_time = time.time()
            if print_steps:
                print(f" Iteration {i+1}/{max_iterations} - Changed: {changed} (Time: {step_end_time - step_start_time:.4f}s)")
            if not changed:
                print(f"Simulation stabilized after {i+1} iterations.")
                break
        else:
            print(f"Simulation stopped after reaching max iterations ({max_iterations}).")
        end_total_time = time.time()
        print(f"--- Simulation Finished (Total Time: {end_total_time - start_total_time:.4f}s) ---")
        self.display_values()

    def display_values(self):
        print("Grid Simulation Target Values (Layer 0):")
        print("-" * (self.width * 5 + 1))
        if self.depth > 0:
            z = 0
            for y in range(self.height):
                # Format numbers to fit: use f-string alignment
                row_str = "| "
                row_str += " | ".join(f"{self.grid[z][y][x].value:^3}" for x in range(self.width))
                row_str += " |"
                print(row_str)
                print("-" * (self.width * 5 + 1)) # Separator line


    # --- Solver Methods (Integrated into Grid3D) ---

    def assign_z_values(self, solver_type='CP-SAT', threshold=1, max_z=10, deviation_weight=1,
                        log_progress=False, solver_id='SCIP'):
        """
        Assigns 'z' values to grid cells using the specified solver.

        Args:
            solver_type (str): 'CP-SAT' or 'MIP'.
            threshold (int): Max Z difference allowed for linked cells.
            max_z (int): Max Z value allowed for any cell.
            deviation_weight (int/float): Penalty multiplier for deviation from simulation 'value'.
                                        Set to 0 to ignore simulation values in objective.
            log_progress (bool): For CP-SAT, whether to log search progress.
            solver_id (str): For MIP, the solver backend to use (e.g., 'SCIP', 'CBC', 'GLPK').
        """
        print(f"\n--- Preparing to Assign Z-Values using {solver_type} ---")
        self.last_solver_type_used = solver_type # Store for potential use later (e.g., filenames)
        start_time = time.time()

        if solver_type.upper() == 'CP-SAT':
            self._assign_z_values_cp_sat_internal(threshold=threshold,
                                                  max_z=max_z,
                                                  deviation_weight=deviation_weight,
                                                  log_progress=log_progress)
        elif solver_type.upper() == 'MIP':
            self._assign_z_values_mip_internal(threshold=threshold,
                                               max_z=max_z,
                                               deviation_weight=deviation_weight,
                                               solver_id=solver_id)
        else:
            print(f"Error: Unknown solver_type '{solver_type}'. Valid options: 'CP-SAT', 'MIP'")
            print("Assigning default Z=0 to all cells due to invalid solver type.")
            for cell in self.each_cell(): cell.z = 0 # Assign default

        end_time = time.time()
        print(f"Total Z-assignment process time ({solver_type}): {end_time - start_time:.4f} seconds")


    def _assign_z_values_cp_sat_internal(self, threshold, max_z, deviation_weight, log_progress):
        """Internal CP-SAT implementation."""
        print(f"Starting CP-SAT Assignment (threshold={threshold}, max_z={max_z}, weight={deviation_weight})")
        start_setup_time = time.time()
        model = cp_model.CpModel()
        z_vars = {}
        abs_deviation_vars = []

        sim_values = {cell: cell.value for cell in self.each_cell()}
        max_sim_value = max(sim_values.values()) if sim_values else 1
        target_scale_max = max_z

        # Create Variables
        for cell in self.each_cell():
            cell.z = None # Reset final Z value
            z_vars[cell] = model.NewIntVar(0, max_z, f"z_{cell.x}_{cell.y}")

            if deviation_weight > 0:
                original_sim_value = sim_values.get(cell, 0)
                scaled_target = int(round((original_sim_value / max_sim_value) * target_scale_max)) if max_sim_value > 0 else 0
                target_value = min(max(0, scaled_target), max_z)

                dev_diff = model.NewIntVar(-max_z, max_z, f"dev_diff_{cell.x}_{cell.y}")
                model.Add(dev_diff == z_vars[cell] - target_value)
                abs_dev = model.NewIntVar(0, max_z, f"abs_dev_{cell.x}_{cell.y}")
                model.AddAbsEquality(abs_dev, dev_diff)
                abs_deviation_vars.append(abs_dev)

        # Add Neighbor Difference Constraints
        already_checked = set()
        constraints_added = 0
        for cell in self.each_cell():
            for neighbor in cell.neighbors: # Use the pre-configured neighbors
                # Ensure pair is processed only once
                pair = tuple(sorted(((cell.x, cell.y), (neighbor.x, neighbor.y))))
                if pair in already_checked: continue
                already_checked.add(pair)

                # Check if neighbor exists in our variable map (should always be true here)
                if neighbor not in z_vars: continue

                # Difference = z_cell - z_neighbor
                diff = model.NewIntVar(-max_z, max_z, f"diff_{pair[0]}_{pair[1]}")
                model.Add(diff == z_vars[cell] - z_vars[neighbor])

                # Absolute difference: abs_diff = |z_cell - z_neighbor|
                abs_diff = model.NewIntVar(0, max_z, f"absdiff_{pair[0]}_{pair[1]}")
                model.AddAbsEquality(abs_diff, diff)

                # Constraint based on link status (using cell.is_linked)
                if cell.is_linked(neighbor):
                    model.Add(abs_diff <= threshold)
                else:
                    model.Add(abs_diff >= threshold + 1) # Strict inequality handled
                constraints_added += 1
        print(f"Added {constraints_added} CP-SAT neighbor difference constraints.")

        # Define Objective Function
        objective_terms = [z_vars[c] for c in z_vars] # Minimize sum of Z
        if deviation_weight > 0 and abs_deviation_vars:
             objective_terms.extend([deviation_weight * dev_var for dev_var in abs_deviation_vars]) # Add deviation penalty
             print(f"Defined CP-SAT objective: Minimize Sum(Z) + {deviation_weight} * Sum(Deviation)")
        else:
             print("Defined CP-SAT objective: Minimize Sum(Z)")
        model.Minimize(sum(objective_terms))

        end_setup_time = time.time()
        print(f"CP-SAT Model setup time: {end_setup_time - start_setup_time:.4f} seconds.")

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = log_progress
        print("Starting CP-SAT solver...")
        start_solve_time = time.time()
        status = solver.Solve(model)
        end_solve_time = time.time()
        print(f"CP-SAT Solver finished in {end_solve_time - start_solve_time:.4f} seconds.")
        if log_progress:
            print(f"  Solver status: {solver.StatusName(status)}")
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                 print(f"  Objective value: {solver.ObjectiveValue()}")

        # Process Results
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            status_str = 'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE'
            print(f"CP-SAT solution found (Status: {status_str}). Assigning final cell.z values...")
            max_assigned_z = 0
            total_deviation = 0
            for cell in z_vars:
                assigned_val = solver.Value(z_vars[cell])
                cell.z = assigned_val # Assign directly to the cell object
                max_assigned_z = max(max_assigned_z, assigned_val)
            print(f"Assigned Z values. Max Z reached = {max_assigned_z}.")
            if deviation_weight > 0 and abs_deviation_vars:
                 total_deviation = sum(solver.Value(dev_var) for dev_var in abs_deviation_vars)
                 avg_deviation = total_deviation / len(abs_deviation_vars) if abs_deviation_vars else 0
                 print(f"  Total absolute deviation from target: {total_deviation:.2f}")
                 print(f"  Avg absolute deviation from target: {avg_deviation:.2f}")
        else:
            print(f"No feasible solution found by CP-SAT solver. Status: {solver.StatusName(status)}")
            for cell in self.each_cell(): cell.z = 0 # Assign default
            print("Assigned default Z=0 to all cells.")
        print("--- CP-SAT Assignment Finished ---")


    def _assign_z_values_mip_internal(self, threshold, max_z, deviation_weight, solver_id):
        """Internal MIP implementation."""
        print(f"Starting MIP Assignment ({solver_id}, threshold={threshold}, max_z={max_z}, weight={deviation_weight})")
        start_setup_time = time.time()

        # Create Solver
        try:
            solver = pywraplp.Solver.CreateSolver(solver_id)
            if not solver: # Fallback if creation returns None
                 print(f"Warning: MIP Solver '{solver_id}' creation returned None. Trying CBC...")
                 solver_id = 'CBC'
                 solver = pywraplp.Solver.CreateSolver(solver_id)
            if not solver: # Final check if even CBC failed
                print("Error: Could not create MIP solver (tried specified and CBC). Cannot run MIP.")
                for cell in self.each_cell(): cell.z = 0
                print("Assigned default Z=0 to all cells.")
                print("--- MIP Assignment Failed ---")
                return
        except Exception as e:
            print(f"An error occurred creating the MIP solver ({solver_id}): {e}")
            for cell in self.each_cell(): cell.z = 0
            print("Assigned default Z=0 to all cells.")
            print("--- MIP Assignment Failed ---")
            return

        infinity = solver.infinity()
        z_vars = {}
        abs_diff_vars = {}
        abs_dev_vars = {}

        sim_values = {cell: cell.value for cell in self.each_cell()}
        max_sim_value = max(sim_values.values()) if sim_values else 1
        target_scale_max = max_z

        # Create Variables
        for cell in self.each_cell():
            cell.z = None # Reset final Z value
            z_vars[cell] = solver.IntVar(0, max_z, f"z_{cell.x}_{cell.y}")

            if deviation_weight > 0:
                original_sim_value = sim_values.get(cell, 0)
                scaled_target = int(round((original_sim_value / max_sim_value) * target_scale_max)) if max_sim_value > 0 else 0
                target_value = min(max(0, scaled_target), max_z)

                # Linearization: abs_dev >= |z_cell - target_value|
                abs_dev_vars[cell] = solver.IntVar(0, max_z, f"abs_dev_{cell.x}_{cell.y}")
                solver.Add(abs_dev_vars[cell] >= z_vars[cell] - target_value, f"dev_lin1_{cell.x}_{cell.y}")
                solver.Add(abs_dev_vars[cell] >= target_value - z_vars[cell], f"dev_lin2_{cell.x}_{cell.y}")

        # Add Neighbor Difference Constraints
        already_checked = set()
        constraints_added = 0
        for cell in self.each_cell():
            for neighbor in cell.neighbors:
                pair = tuple(sorted(((cell.x, cell.y), (neighbor.x, neighbor.y))))
                if pair in already_checked: continue
                already_checked.add(pair)
                if neighbor not in z_vars: continue

                # Linearization: abs_diff >= |z_cell - z_neighbor|
                abs_diff_var = solver.IntVar(0, max_z, f"absdiff_{pair[0]}_{pair[1]}")
                abs_diff_vars[pair] = abs_diff_var
                solver.Add(abs_diff_var >= z_vars[cell] - z_vars[neighbor], f"absdiff_lin1_{pair}")
                solver.Add(abs_diff_var >= z_vars[neighbor] - z_vars[cell], f"absdiff_lin2_{pair}")

                # Constraint based on link status
                if cell.is_linked(neighbor):
                    solver.Add(abs_diff_var <= threshold, f"linked_{pair}")
                else:
                    solver.Add(abs_diff_var >= threshold + 1, f"notlinked_{pair}")
                constraints_added += 1
        print(f"Added {constraints_added} MIP neighbor difference constraints (linearized).")

        # Define Objective Function
        objective = solver.Objective()
        # Part 1: Minimize Sum(Z)
        for cell in z_vars:
            objective.SetCoefficient(z_vars[cell], 1)
        # Part 2: Minimize Sum(Deviation) weighted
        if deviation_weight > 0:
            for cell in abs_dev_vars:
                objective.SetCoefficient(abs_dev_vars[cell], deviation_weight)
            print(f"Defined MIP objective: Minimize Sum(Z) + {deviation_weight} * Sum(Linearized Deviation)")
        else:
             print("Defined MIP objective: Minimize Sum(Z)")
        # Part 3: Small penalty on abs_diff variables (optional tightening)
        epsilon = 0.001
        for pair in abs_diff_vars:
             objective.SetCoefficient(abs_diff_vars[pair], epsilon)
        if abs_diff_vars: print(f"Added small penalty ({epsilon}) for linearized absolute differences to MIP objective.")

        objective.SetMinimization()
        end_setup_time = time.time()
        print(f"MIP Model setup time: {end_setup_time - start_setup_time:.4f} seconds.")

        # Solve
        print(f"Starting MIP solver ({solver_id})...")
        start_solve_time = time.time()
        status = solver.Solve()
        end_solve_time = time.time()
        print(f"MIP Solver finished in {end_solve_time - start_solve_time:.4f} seconds.")

        # Process Results
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            status_str = 'OPTIMAL' if status == pywraplp.Solver.OPTIMAL else 'FEASIBLE'
            print(f"MIP solution found (Status: {status_str}). Objective value: {solver.Objective().Value():.4f}")
            print("Assigning final cell.z values...")
            max_assigned_z = 0
            total_deviation = 0
            num_dev_vars = 0
            for cell in z_vars:
                assigned_val = int(round(z_vars[cell].solution_value()))
                cell.z = assigned_val # Assign directly to cell
                max_assigned_z = max(max_assigned_z, assigned_val)
                if cell in abs_dev_vars:
                     total_deviation += abs_dev_vars[cell].solution_value()
                     num_dev_vars += 1

            print(f"Assigned Z values. Max Z reached = {max_assigned_z}.")
            if deviation_weight > 0 and num_dev_vars > 0:
                 avg_deviation = total_deviation / num_dev_vars
                 print(f"  Sum of MIP deviation variables: {total_deviation:.2f}")
                 print(f"  Avg MIP deviation variable value: {avg_deviation:.2f}")
        else:
            status_map = { pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
                           pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
                           pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
                           pywraplp.Solver.ABNORMAL: "ABNORMAL",
                           pywraplp.Solver.MODEL_INVALID: "MODEL_INVALID" }
            print(f"No optimal/feasible solution found by MIP solver ({solver_id}). Status: {status_map.get(status, 'UNKNOWN')}")
            for cell in self.each_cell(): cell.z = 0 # Assign default
            print("Assigned default Z=0 to all cells.")
        print(f"--- MIP ({solver_id}) Assignment Finished ---")


# --- Union-Find Class ---
class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, cell):
        # Path compression
        if self.parent.get(cell, cell) != cell:
            self.parent[cell] = self.find(self.parent[cell])
        return self.parent.get(cell, cell)
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            # Union by rank/size could be added here for further optimization
            self.parent[root_b] = root_a
            return True # Return True if a union occurred
        return False # Return False if they were already in the same set

# --- Kruskal Maze Generation ---
def kruskal_with_cycle_tracking(grid):
    """Generates a maze on the grid using Kruskal's algorithm."""
    print("\n--- Generating Maze using Kruskal's ---")
    uf = UnionFind()
    edges = []
    # Create a list of all possible edges between adjacent cells (N/S, E/W) on layer 0
    for cell in grid.each_cell():
        for neighbor in cell.neighbors:
             # Add edge only once (e.g., based on coordinates to avoid duplicates)
             if (cell.x, cell.y) < (neighbor.x, neighbor.y):
                 # Store as tuple of Cell3D objects
                 edges.append(tuple(sorted((cell, neighbor), key=lambda c: (c.x, c.y))))

    random.shuffle(edges)
    link_count = 0
    # Iterate through shuffled edges
    for a, b in edges:
        # If cells 'a' and 'b' are not already connected (different sets in UnionFind)
        if uf.union(a, b):
            # Link them in the grid (creates the passage)
            a.link(b) # link() is bidirectional by default
            link_count += 1
            # Optional: Stop early if a perfect maze is desired (width*height - 1 links)
            # if link_count == (grid.width * grid.height - 1): break

    print(f"--- Maze Generation Finished ({link_count} links created) ---")

# --- Genome Extraction ---
def extract_genome(grid):
    """Extracts the Z-values from layer 0 into a 2D list."""
    genome = []
    if grid.depth > 0:
        z_layer = 0 # Assume we want the base layer
        genome = [[grid.get_cell(x, y, z_layer).z if grid.get_cell(x, y, z_layer) else 0
                   for x in range(grid.width)]
                  for y in range(grid.height)]
    return genome

# --- Visualization Functions  ---

def draw_3d_maze(grid, filename="3d_maze_matplotlib_merged.png"):
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)
    print(f"\n--- Generating 3D Matplotlib plot ({filename}) ---")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"3D Maze (Height from Z, Solver: {grid.last_solver_type_used})")

    max_z_val = 0; min_z_val = 0
    all_zs = [c.z for c in grid.each_cell() if c.z is not None]
    if all_zs: max_z_val = max(all_zs); min_z_val = min(all_zs)

    for cell in grid.each_cell():
        x, y, z = cell.x, cell.y, cell.z if cell.z is not None else 0
        for neighbor in cell.links:
             if (neighbor.x, neighbor.y) > (cell.x, cell.y):
                 x2, y2 = neighbor.x, neighbor.y
                 z2 = neighbor.z if neighbor.z is not None else 0
                 ax.plot([x + 0.5, x2 + 0.5], [y + 0.5, y2 + 0.5], [z, z2], color='black', linewidth=1.5, zorder=1)
        ax.scatter(x + 0.5, y + 0.5, z, color='purple', s=35, depthshade=True, zorder=2)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z (Assigned)')
    ax.set_xlim(0, grid.width); ax.set_ylim(0, grid.height);
    z_range = max(1, max_z_val - min_z_val)
    ax.set_zlim(min_z_val - 0.1 * z_range, max_z_val + 0.1 * z_range)
    if max_z_val == 0 and min_z_val == 0: ax.set_zlim(-0.5, 1)

    plt.tight_layout()
    try: plt.savefig(full_path); print(f" Saved Matplotlib Z-plot '{full_path}'")
    except Exception as e: print(f"Error saving Matplotlib plot: {e}")
    plt.close(fig)

def link_matrix(grid):
    """Generates matrices representing horizontal and vertical links."""
    matrix = []; bottom_links = []
    if grid.depth > 0:
        z = 0 # Assume base layer
        # Horizontal links (check cell to the right)
        for y in range(grid.height):
            row = []
            for x in range(grid.width):
                cell = grid.grid[z][y][x]
                # Check link to the East (x+1)
                if x < grid.width - 1:
                    neighbor_east = grid.grid[z][y][x + 1]
                    row.append(1 if cell.is_linked(neighbor_east) else 0)
                else:
                    row.append(-1) # Indicate boundary edge
            matrix.append(row)
        # Vertical links (check cell below)
        for y in range(grid.height):
            row = []
            for x in range(grid.width):
                cell = grid.grid[z][y][x]
                # Check link to the South (y+1)
                if y < grid.height - 1:
                    neighbor_south = grid.grid[z][y + 1][x]
                    row.append(1 if cell.is_linked(neighbor_south) else 0)
                else:
                    row.append(-1) # Indicate boundary edge
            bottom_links.append(row)
    return matrix, bottom_links

def plot_link_matrix(grid, filename="link_matrix_plot.png"):
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)
    print(f"\n--- Generating Link Matrix plot ({filename}) ---")
    horiz, vert = link_matrix(grid) # Get link data
    width = grid.width; height = grid.height
    if not horiz or not vert: print("No link data to plot."); return

    fig, ax = plt.subplots(figsize=(width*0.6 + 1, height*0.6 + 1))
    ax.set_title("Maze Link Matrix"); ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5); ax.set_aspect('equal') # Ensure square cells

    # Draw grid points and links
    for y in range(height):
        for x in range(width):
            ax.plot(x, y, 'ko', markersize=5) # Draw cell center node
            # Draw horizontal link (to the right) if exists
            if x < width - 1 and horiz[y][x] == 1:
                 ax.plot([x, x + 1], [y, y], 'b-', linewidth=2)
            # Draw vertical link (downwards) if exists
            if y < height - 1 and vert[y][x] == 1:
                 ax.plot([x, x], [y, y + 1], 'r-', linewidth=2)

    ax.set_xticks(range(width)); ax.set_yticks(range(height))
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    try: plt.savefig(full_path); print(f" Saved Link Matrix plot '{full_path}'")
    except Exception as e: print(f"Error saving Link Matrix plot: {e}")
    plt.close(fig)

def plot_z_values_as_bars(grid, filename="z_bar_plot_interactive_merged.html"):
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)
    print(f"\n--- Generating Interactive Plotly plot ({filename}) ---")
    bar_data = []; link_data = []; max_z_val = 0; min_z_val = 0
    cell_zs = {}
    valid_zs = [c.z for c in grid.each_cell() if c.z is not None]
    if valid_zs: max_z_val = max(valid_zs); min_z_val = min(valid_zs)
    for cell in grid.each_cell(): cell_zs[cell] = cell.z if cell.z is not None else 0

    for cell in grid.each_cell():
        x1, y1 = cell.x + 0.5, cell.y + 0.5; z1 = cell_zs[cell]
        bar_base = min(0, min_z_val)
        bar_data.append(go.Scatter3d(
            x=[x1, x1], y=[y1, y1], z=[bar_base, z1], mode="lines",
            line=dict(color='rgba(128, 0, 128, 0.7)', width=max(4, 12/grid.width)),
            hoverinfo='text', text=f'Cell ({cell.x},{cell.y})<br>Z: {z1}', showlegend=False))
        for neighbor in cell.links:
            if (neighbor.x, neighbor.y) > (cell.x, cell.y):
                x2, y2 = neighbor.x + 0.5, neighbor.y + 0.5
                z2 = cell_zs.get(neighbor, 0)
                link_data.append(go.Scatter3d(
                    x=[x1, x2], y=[y1, y2], z=[z1, z2], mode="lines",
                    line=dict(color='black', width=max(1, 3/grid.width)),
                    hoverinfo='none', showlegend=False ))

    fig = go.Figure(data=bar_data + link_data)
    z_range = max(1, max_z_val - min_z_val)
    z_axis_range = [min_z_val - 0.1 * z_range, max_z_val + 0.1 * z_range]
    if max_z_val == 0 and min_z_val == 0: z_axis_range = [-0.5, 1]

    fig.update_layout(
        title=f"3D Maze Z 'Height' ({grid.width}x{grid.height}, Solver: {grid.last_solver_type_used})",
        scene=dict( xaxis_title="X", yaxis_title="Y", zaxis_title="Z (Assigned)",
            xaxis=dict(nticks=max(5, grid.width//2), range=[0,grid.width]),
            yaxis=dict(nticks=max(5, grid.height//2), range=[0,grid.height]),
            zaxis=dict(nticks=max(5, int(z_range+1)//2), range=z_axis_range),
            aspectmode='cube'),
        margin=dict(l=10, r=10, b=10, t=50), showlegend=False )
    try: fig.write_html(full_path); print(f" Saved Interactive Z-plot '{full_path}'")
    except Exception as e: print(f"Error saving Interactive plot: {e}")

def plot_genome_2d(grid, filename="genome_2d_plot.png", show_values=True, cmap_name='viridis'):
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)
    print(f"\n--- Generating 2D Genome Heatmap ({filename}) ---")
    width = grid.width; height = grid.height
    genome_data = np.full((height, width), np.nan)
    valid_zs = []
    for y in range(height):
        for x in range(width):
            cell = grid.get_cell(x, y)
            if cell and cell.z is not None:
                z_val = cell.z
                genome_data[y, x] = z_val
                valid_zs.append(z_val)

    min_z = min(valid_zs) if valid_zs else 0
    max_z = max(valid_zs) if valid_zs else 1
    if min_z == max_z: max_z += 1

    fig, ax = plt.subplots(figsize=(width * 0.8 + 1, height * 0.8 + 1))
    ax.set_title(f"2D Genome Layout (Z-Value Heatmap, Solver: {grid.last_solver_type_used})")
    try: cmap = plt.get_cmap(cmap_name)
    except ValueError: print(f"Warning: Colormap '{cmap_name}' not found. Using 'viridis'."); cmap = plt.get_cmap('viridis')

    im = ax.imshow(genome_data, cmap=cmap, origin='upper',
                   extent=[-0.5, width - 0.5, height - 0.5, -0.5],
                   vmin=min_z, vmax=max_z, interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('Z-Value', rotation=270, labelpad=15)

    for y in range(height):
        for x in range(width):
            cell = grid.get_cell(x, y)
            if not cell: continue
            for neighbor in cell.links:
                if (neighbor.x > cell.x) or (neighbor.y > cell.y):
                    ax.plot([cell.x, neighbor.x], [cell.y, neighbor.y], color='white', linestyle='-', linewidth=1.5, alpha=0.8)

    if show_values:
        for y in range(height):
            for x in range(width):
                z_val = genome_data[y, x]
                if not np.isnan(z_val):
                    norm_val = (z_val - min_z) / (max_z - min_z + 1e-9)
                    bg_color_val = cmap(norm_val)
                    luminance = 0.299 * bg_color_val[0] + 0.587 * bg_color_val[1] + 0.114 * bg_color_val[2]
                    text_color = 'white' if luminance < 0.5 else 'black'
                    ax.text(x, y, f"{int(z_val)}", ha='center', va='center', color=text_color, fontsize=max(6, 10 - grid.width // 3))

    ax.set_xticks(np.arange(width)); ax.set_yticks(np.arange(height))
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True); ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    ax.tick_params(which='minor', size=0); ax.tick_params(axis='both', which='major', length=0)
    plt.tight_layout(pad=1.5)
    try: plt.savefig(full_path); print(f" Saved 2D Genome plot '{full_path}'")
    except Exception as e: print(f"Error saving 2D Genome plot: {e}")
    plt.close(fig)
