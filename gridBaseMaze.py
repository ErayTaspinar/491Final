import random
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from ortools.sat.python import cp_model

# --- Cell3D Class ---
class Cell3D:
    def __init__(self, x, y, z=None):
        self.x = x
        self.y = y
        self.z = z
        self.links = set()
        self.neighbors = []

    def link(self, other, bidirectional=True):
        self.links.add(other)
        if bidirectional:
            other.link(self, False)

    def is_linked(self, other):
        return other in self.links

# --- Grid3D Class ---
class Grid3D:
    def __init__(self, width, height, depth=1):
        self.width = width
        self.height = height
        self.depth = depth
        self.grid = self.prepare_grid()
        self.configure_neighbors()

    def prepare_grid(self):
        return [[[Cell3D(x, y, z) for x in range(self.width)]
                 for y in range(self.height)]
                 for z in range(self.depth)]

    def configure_neighbors(self):
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    cell = self.grid[z][y][x]
                    if x > 0:
                        cell.neighbors.append(self.grid[z][y][x - 1])
                    if x < self.width - 1:
                        cell.neighbors.append(self.grid[z][y][x + 1])
                    if y > 0:
                        cell.neighbors.append(self.grid[z][y - 1][x])
                    if y < self.height - 1:
                        cell.neighbors.append(self.grid[z][y + 1][x])

    def each_cell(self):
        for z_layer in self.grid:
            for row in z_layer:
                for cell in row:
                    yield cell

# --- Union-Find Class ---
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, cell):
        if self.parent.get(cell, cell) != cell:
            self.parent[cell] = self.find(self.parent[cell])
        return self.parent.get(cell, cell)

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a
            return True
        return False

# --- Kruskal Maze Generation ---
def kruskal_with_cycle_tracking(grid):
    uf = UnionFind()
    edges = []

    for cell in grid.each_cell():
        for neighbor in cell.neighbors:
            if (cell.x, cell.y) <= (neighbor.x, neighbor.y):
                edges.append((cell, neighbor))

    random.shuffle(edges)

    for a, b in edges:
        if uf.union(a, b):
            a.link(b)

# --- Assign Z-Values with Constraints ---
def assign_z_values(grid, threshold=1, max_z=10):
    model = cp_model.CpModel()
    z_vars = {}
    max_height = max_z

    for cell in grid.each_cell():
        cell.z = None
        z_vars[cell] = model.NewIntVar(0, max_height, f"z_{cell.x}_{cell.y}_{cell.z}")

    already_checked = set()
    for cell in grid.each_cell():
        for neighbor in cell.neighbors:
            if (cell, neighbor) in already_checked or (neighbor, cell) in already_checked:
                continue
            already_checked.add((cell, neighbor))

            diff = model.NewIntVar(-max_height, max_height, f"diff_{cell.x}_{cell.y}_{cell.z}_{neighbor.x}_{neighbor.y}_{neighbor.z}")
            abs_diff = model.NewIntVar(0, max_height, f"absdiff_{cell.x}_{cell.y}_{cell.z}_{neighbor.x}_{neighbor.y}_{neighbor.z}")
            model.Add(diff == z_vars[cell] - z_vars[neighbor])
            model.AddAbsEquality(abs_diff, diff)

            if cell.is_linked(neighbor):
                model.Add(abs_diff <= threshold)
            else:
                model.Add(abs_diff > threshold)

    model.Minimize(sum(z_vars[c] for c in z_vars))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for cell in z_vars:
            cell.z = solver.Value(z_vars[cell])
            print(f" Cell ({cell.x}, {cell.y}) assigned z = {cell.z}")
    else:
        print(" No feasible solution found.")

# --- Draw 3D Maze ---
def draw_3d_maze(grid, filename="3d_maze_with_assigned_z.png"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Maze")

    for cell in grid.each_cell():
        x, y, z = cell.x, cell.y, cell.z
        for neighbor in cell.links:
            x2, y2, z2 = neighbor.x, neighbor.y, neighbor.z
            ax.plot([x + 0.5, x2 + 0.5], [y + 0.5, y2 + 0.5], [z + 0.5, z2 + 0.5], color='black')
        ax.scatter(x + 0.5, y + 0.5, z + 0.5, color='blue', s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)
    ax.set_zlim(0, max(cell.z for cell in grid.each_cell()) + 1)
    plt.tight_layout()
    plt.savefig(filename)
    print(f" 3D maze image saved as '{filename}'")

# --- Link Matrix ---
def link_matrix(grid):
    matrix = []
    for y in range(grid.height):
        row = []
        for x in range(grid.width):
            cell = grid.grid[0][y][x]
            if x < grid.width - 1:
                right = grid.grid[0][y][x + 1]
                row.append(1 if cell.is_linked(right) else 0)
            else:
                row.append(-1)
        matrix.append(row)

    bottom_links = []
    for y in range(grid.height):
        row = []
        for x in range(grid.width):
            cell = grid.grid[0][y][x]
            if y < grid.height - 1:
                below = grid.grid[0][y + 1][x]
                row.append(1 if cell.is_linked(below) else 0)
            else:
                row.append(-1)
        bottom_links.append(row)

    return matrix, bottom_links

# --- Plot Link Matrix ---
def plot_link_matrix(grid):
    horiz, vert = link_matrix(grid)
    width = grid.width
    height = grid.height

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Maze Link Matrix")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    for y in range(height):
        for x in range(width):
            ax.plot(x + 0.5, y + 0.5, 'ko', markersize=5)
            if x < width - 1 and horiz[y][x] == 1:
                ax.plot([x + 0.5, x + 1.5], [y + 0.5, y + 0.5], 'b-', linewidth=2)
            if y < height - 1 and vert[y][x] == 1:
                ax.plot([x + 0.5, x + 0.5], [y + 0.5, y + 1.5], 'r-', linewidth=2)

    ax.set_xticks(range(width + 1))
    ax.set_yticks(range(height + 1))
    ax.grid(True, which='both')
    plt.tight_layout()
    plt.savefig("link_matrix_plot.png")
    print(" Link matrix visualization saved as 'link_matrix_plot.png'")

def plot_z_values_as_bars(grid):
    bar_data = []
    link_data = []

    for cell in grid.each_cell():
        x1, y1, z1 = cell.x + 0.5, cell.y + 0.5, cell.z if cell.z is not None else 0

        # Bar from base to Z
        bar_data.append(go.Scatter3d(
            x=[x1, x1],
            y=[y1, y1],
            z=[0, z1],
            mode="lines",
            line=dict(color='blue', width=8),
            showlegend=False
        ))

        # Add link lines to neighbors
        for neighbor in cell.links:
            x2, y2, z2 = neighbor.x + 0.5, neighbor.y + 0.5, neighbor.z if neighbor.z is not None else 0
            # Avoid double-drawing links
            if (neighbor.x > cell.x) or (neighbor.y > cell.y):
                link_data.append(go.Scatter3d(
                    x=[x1, x2],
                    y=[y1, y2],
                    z=[z1, z2],
                    mode="lines",
                    line=dict(color='black', width=3),
                    showlegend=False
                ))

    # Combine bar and link data
    fig = go.Figure(data=bar_data + link_data)

    fig.update_layout(
        title="3D Maze with Elevation and Links (Interactive)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z (Height)",
            xaxis=dict(nticks=grid.width),
            yaxis=dict(nticks=grid.height),
            zaxis=dict(nticks=10),
        ),
        margin=dict(l=10, r=10, b=10, t=40),
        showlegend=False
    )

    fig.write_html("z_bar_plot_with_links.html")
    print("✅ Interactive plot with links saved as 'z_bar_plot_with_links.html'")
    fig.show()
