import numpy as np
import random
import imageio

class MazeEnv:
    def __init__(self, width=21, height=21):
        self.width = width
        self.height = height
        # Ensure odd dimensions for wall/path logic
        if self.width % 2 == 0: self.width += 1
        if self.height % 2 == 0: self.height += 1

        self.grid = np.ones((self.height, self.width), dtype=int) # 1=wall, 0=path
        self.start = (1, 1)
        self.goal = (self.height - 2, self.width - 2)
        self.agent_pos = self.start

        self._generate_maze()

    def _generate_maze(self):
        # Recursive backtracking
        # Start with all walls (already done in init)
        self.grid[self.start] = 0

        visited = set([self.start])

        def get_neighbors(r, c):
            # Jump 2 steps
            directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            neighbors = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    neighbors.append(((nr, nc), (r + dr//2, c + dc//2)))
            random.shuffle(neighbors)
            return neighbors

        # Iterative DFS to avoid recursion depth issues
        stack = [self.start]
        while stack:
            current = stack[-1]
            neighbors = get_neighbors(*current)
            unvisited_neighbors = [n for n in neighbors if n[0] not in visited]

            if unvisited_neighbors:
                next_cell, wall = unvisited_neighbors[0]
                self.grid[next_cell] = 0
                self.grid[wall] = 0
                visited.add(next_cell)
                stack.append(next_cell)
            else:
                stack.pop()

        # Ensure goal is accessible and empty
        self.grid[self.goal] = 0

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        r, c = self.agent_pos
        dr, dc = 0, 0
        if action == 0: dr = -1
        elif action == 1: dr = 1
        elif action == 2: dc = -1
        elif action == 3: dc = 1

        nr, nc = r + dr, c + dc
        if 0 <= nr < self.height and 0 <= nc < self.width and self.grid[nr, nc] == 0:
            self.agent_pos = (nr, nc)

        done = (self.agent_pos == self.goal)
        return self.agent_pos, done

    def render(self, scale=20):
        # Create an RGB image
        img = np.zeros((self.height * scale, self.width * scale, 3), dtype=np.uint8)

        # Walls are black (0,0,0), Paths are white (255,255,255)
        # Using broadcasting for efficiency if possible, but loop is fine for small maze
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] == 0:
                    img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = [255, 255, 255]
                else:
                    img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = [0, 0, 0]

        # Goal is Green
        gr, gc = self.goal
        img[gr*scale:(gr+1)*scale, gc*scale:(gc+1)*scale] = [0, 255, 0]

        # Agent is Red
        ar, ac = self.agent_pos
        img[ar*scale:(ar+1)*scale, ac*scale:(ac+1)*scale] = [255, 0, 0]

        return img

def solve_maze_bfs(env):
    start = env.start
    goal = env.goal
    queue = [(start, [])]
    visited = set([start])

    while queue:
        (r, c), path = queue.pop(0)

        if (r, c) == goal:
            return path

        # Explore neighbors
        # Actions: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        directions = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)] # dr, dc, action
        for dr, dc, action in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width and env.grid[nr, nc] == 0:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [action]))
    return []

if __name__ == "__main__":
    # 1. Create Environment
    env = MazeEnv(width=21, height=21)
    print("Maze generated.")

    # 2. Solve Maze
    solution_actions = solve_maze_bfs(env)
    print(f"Maze solved in {len(solution_actions)} steps.")

    # 3. Render Solution
    frames = []
    env.reset()
    frames.append(env.render())

    for action in solution_actions:
        env.step(action)
        frames.append(env.render())

    # Add a few frames at the end to pause on the solution
    final_frame = frames[-1]
    for _ in range(10):
        frames.append(final_frame)

    # 4. Save Video
    output_filename = "maze_solution.mp4"
    imageio.mimsave(output_filename, frames, fps=10)
    print(f"Video saved to {output_filename}")
