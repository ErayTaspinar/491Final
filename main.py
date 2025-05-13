import gridBaseMaze as gbm
import pprint
import os

if __name__ == "__main__":
    # --- Configuration ---
    GRID_WIDTH = 5
    GRID_HEIGHT = 5
    SIMULATION_ITERATIONS = 50
    THRESHOLD = 5
    MAX_Z = 20
    DEVIATION_WEIGHT = 3

    # --- CHOOSE SOLVER TYPE AND SETTINGS ---
    # You set the desired solver type when calling the method
    #solver_to_use = 'CP-SAT'
    solver_to_use = 'MIP'

    # --- Solver Specific Settings (passed during the call) ---
    mip_solver_id = 'SCIP'      # e.g., 'SCIP', 'CBC', 'GLPK'
    cp_sat_log_progress = False # Verbose output for CP-SAT

    os.makedirs("outputs", exist_ok=True) #output directory

    print(f"Creating Grid ({GRID_WIDTH}x{GRID_HEIGHT})...")
    grid = gbm.Grid3D(width=GRID_WIDTH, height=GRID_HEIGHT, depth=1)

    gbm.kruskal_with_cycle_tracking(grid)
    gbm.plot_link_matrix(grid, filename=f"link_matrix_{GRID_WIDTH}x{GRID_HEIGHT}.png")

    if DEVIATION_WEIGHT > 0:
        # Simulation seeding is now handled inside run_full_simulation in the class
        grid.run_full_simulation(max_iterations=SIMULATION_ITERATIONS)
    else:
        print("\nSkipping simulation run as DEVIATION_WEIGHT is 0.")
        for cell in grid.each_cell(): cell.value = 0 # Still ensure value exists

    # Call the integrated method and pass the solver choice and relevant params
    grid.assign_z_values(
        solver_type=solver_to_use,
        threshold=THRESHOLD,
        max_z=MAX_Z,
        deviation_weight=DEVIATION_WEIGHT,
        solver_id=mip_solver_id,         # Pass MIP setting
        log_progress=cp_sat_log_progress # Pass CP-SAT setting
    )

    # Filenames now use grid.last_solver_type_used if needed, or pass solver_to_use
    gbm.draw_3d_maze(grid, filename=f"3d_maze_{GRID_WIDTH}x{GRID_HEIGHT}_{solver_to_use}.png")
    gbm.plot_z_values_as_bars(grid, filename=f"z_bars_{GRID_WIDTH}x{GRID_HEIGHT}_{solver_to_use}.html")

    genome_layout = gbm.extract_genome(grid)
    print("\n--- Z-Value Genome Layout (Text) ---")
    pprint.pprint(genome_layout, indent=2, width=(GRID_WIDTH * 4 + 10))

    gbm.plot_genome_2d(grid,
                       filename=f"genome_heatmap_{GRID_WIDTH}x{GRID_HEIGHT}_{solver_to_use}.png",
                       show_values=True,
                       cmap_name='viridis')

    print("\n--- Main Script Finished ---")