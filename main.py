import gridBaseMaze as gbm

if __name__ == "__main__":
    # Step 1: Create grid
    grid = gbm.Grid3D(width=4, height=4, depth=1)

    # Step 2: Generate links using Kruskal's algorithm
    gbm.kruskal_with_cycle_tracking(grid)

    # Step 2.5: Visualize link matrix
    gbm.plot_link_matrix(grid)

    # Step 3: Solve for z-values using OR-Tools with higher variation
    gbm.assign_z_values(grid, threshold=2, max_z=10)
    gbm.plot_z_values_as_bars(grid)

    # Debug print: See actual assigned z-values
    print("\n Final Z-values after solving:")
    for cell in grid.each_cell():
        print(f"Cell ({cell.x}, {cell.y}) → z = {cell.z}")

    # Step 4: Draw the 3D maze using updated z values
    gbm.draw_3d_maze(grid, filename="3d_maze_with_assigned_z.png")
