import random
import matplotlib.pyplot as plt
import numpy as np

# --- Cell3D Class ---
class Cell3D:
    def __init__(self, x, y, z=0):
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

    def __repr__(self):
        return f"Cell({self.x}, {self.y}, {self.z})"


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
                    if z > 0:
                        cell.neighbors.append(self.grid[z - 1][y][x])
                    if z < self.depth - 1:
                        cell.neighbors.append(self.grid[z + 1][y][x])

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


# --- Kruskal with Cycle Detection ---
def kruskal_with_cycle_tracking(grid, z=0):
    uf = UnionFind()
    edges = []

    for cell in grid.each_cell():
        if cell.z != z:
            continue
        for neighbor in cell.neighbors:
            if neighbor.z == z and cell.x <= neighbor.x and cell.y <= neighbor.y:
                edges.append((cell, neighbor))

    random.shuffle(edges)
    cycle_links = []

    print("\n--- Running Kruskal's Algorithm ---")

    for a, b in edges:
        if uf.union(a, b):
            a.link(b)
        else:
            print(f"Cycle detected: ({a.x},{a.y}) <-> ({b.x},{b.y})")
            a.link(b)
            cycle_links.append((a, b))

    return cycle_links


# --- Draw Maze Evolution ---
def draw_maze_evolution(grid, z, cycle_steps):
    total_steps = len(cycle_steps) + 1
    cols = min(5, total_steps)
    rows = (total_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    def draw_grid_state(ax, title=""):
        for cell in grid.each_cell():
            if cell.z != z:
                continue
            x = cell.x
            y = grid.height - cell.y - 1
            for linked in cell.links:
                if linked.z != z:
                    continue
                x2 = linked.x
                y2 = grid.height - linked.y - 1
                ax.plot([x + 0.5, x2 + 0.5], [y + 0.5, y2 + 0.5], color='black', linewidth=1)
            ax.plot(x + 0.5, y + 0.5, 'o', color='gray', markersize=3)
            ax.text(x + 0.5, y + 0.5, f"({x},{y})", ha='center', va='center', fontsize=6, color='blue')
        ax.set_xlim(0, grid.width)
        ax.set_ylim(0, grid.height)
        ax.set_title(title)
        ax.axis('off')

    draw_grid_state(axes[0], "Original (with cycles)")

    for i, (a, b) in enumerate(cycle_steps):
        a.links.discard(b)
        b.links.discard(a)
        draw_grid_state(axes[i + 1], f"Removed ({a.x},{a.y}) <-> ({b.x},{b.y})")

    for i in range(total_steps, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# --- Main Program ---
grid = Grid3D(width=5, height=5, depth=1)
cycle_links = kruskal_with_cycle_tracking(grid, z=0)
draw_maze_evolution(grid, z=0, cycle_steps=cycle_links)
