import random
import heapq
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import os


def load_maze(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    maze = [list(line.strip()) for line in lines if
            line.strip() and not line.startswith("start") and not line.startswith("end")]

    start, end = None, None
    for line in lines:
        if line.startswith("start") or line.startswith("end"):
            label, x, y = line.strip().split(maxsplit=2)
            x = ''.join(filter(str.isdigit, x))
            y = ''.join(filter(str.isdigit, y))
            if label == "start":
                start = (int(y), int(x))
            elif label == "end":
                end = (int(y), int(x))

    return maze, start, end


def print_maze(maze, start, end, path=[]):
    maze_copy = [row[:] for row in maze]
    for x, y in path:
        maze_copy[x][y] = "o"

    sx, sy = start
    ex, ey = end
    maze_copy[sx][sy] = "S"
    maze_copy[ex][ey] = "E"
    for row in maze_copy:
        print("".join(row))
    print()


def visualize_maze(frames, algorithm_name):
    def preprocess_for_plot(maze):
        return [[
            0 if cell == "X" else
            1 if cell == " " else
            2 if cell == "#" else
            5 if cell == "o" else
            3 if cell == "S" else
            4 if cell == "E" else
            -1 for cell in row
        ] for row in maze]

    if "S" not in [cell for row in frames[0] for cell in row]:
        # print(maze[start[0]][start[1]], frames[0][start[0]][start[1]])
        raise ValueError("The start position 'S' is not present in the initial frame.")
    if "E" not in [cell for row in frames[0] for cell in row]:
        # print(maze[end[0]][end[1]], frames[0][end[0]][end[1]])
        raise ValueError("The end position 'E' is not present in the initial frame.")

    processed_frames = [preprocess_for_plot(frame) for frame in frames]

    cmap = ListedColormap([
        "black",
        "white",
        "green",
        "red",
        "yellow"
    ])

    fig, ax = plt.subplots()
    im = ax.imshow(processed_frames[0], cmap=cmap, interpolation="nearest")
    ax.set_title(algorithm_name, fontsize=16)
    ax.axis("off")
    def update(frame):
        im.set_array(processed_frames[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(processed_frames), interval=50, repeat=False, blit=True
    )
    plt.show()


def handle_edge_case(maze, start, end):
    """Handles edge case where start == end."""
    if start == end:
        print("Start and end positions are the same! No search needed.")
        maze[start[0]][start[1]] = "E"
        return [[row[:] for row in maze]]
    return None

def dfs_with_animation(maze, start, end):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    stack = [(start, [start])]
    visited = set()
    opened = -1
    maze_copy = [row[:] for row in maze]
    frames = [[row[:] for row in maze_copy]]

    while stack:
        (x, y), path = stack.pop()
        if (x, y) in visited:
            continue

        visited.add((x, y))
        maze_copy[x][y] = '#'
        opened += 1

        frames.append([row[:] for row in maze_copy])

        if (x, y) == end:
            for px, py in path:
                maze_copy[px][py] = "o" if (px, py) != start else "S"
            maze_copy[x][y] = 'E'
            frames.append([row[:] for row in maze_copy])
            print("Nodes expanded: ", opened)
            print("Path length: ", len(path) - 1)
            return frames

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                    0 <= nx < len(maze)
                    and 0 <= ny < len(maze[0])
                    and maze_copy[nx][ny] != "X"
                    and (nx, ny) not in visited
            ):
                stack.append(((nx, ny), path + [(nx, ny)]))

    return frames



def bfs_with_animation(maze, start, end):

    maze_copy = [row[:] for row in maze]

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    queue = deque([start])
    visited = [[False] * len(maze_copy[0]) for _ in range(len(maze_copy))]
    visited[start[0]][start[1]] = True
    parent = {}
    opened_nodes = 0
    frames = [[row[:] for row in maze_copy]]

    while queue:
        x, y = queue.popleft()

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                    0 <= nx < len(maze_copy) and
                    0 <= ny < len(maze_copy[0]) and
                    maze_copy[nx][ny] != "X" and
                    not visited[nx][ny]
            ):
                parent[(nx, ny)] = (x, y)
                visited[nx][ny] = True
                queue.append((nx, ny))
                maze_copy[nx][ny] = "#"
                opened_nodes += 1

                frames.append([row[:] for row in maze_copy])

                if (nx, ny) == end:

                    path = []
                    while (nx, ny) != start:
                        path.append((nx, ny))
                        nx, ny = parent[(nx, ny)]
                    path.reverse()

                    for px, py in path:
                        maze_copy[px][py] = "o"
                    maze_copy[start[0]][start[1]] = "S"
                    maze_copy[end[0]][end[1]] = "E"

                    frames.append([row[:] for row in maze_copy])
                    print("Nodes expanded:", opened_nodes)
                    print("Path length:", len(path))
                    return frames

    return frames



def greedy_search_with_animation(maze, start, end):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    pq = [(heuristic(start, end), start, [start])]
    visited = set()
    opened = 0

    maze_copy = [row[:] for row in maze]
    frames = [[row[:] for row in maze_copy]]

    while pq:
        _, (x, y), path = heapq.heappop(pq)
        if (x, y) in visited:
            continue

        visited.add((x, y))
        maze_copy[x][y] = "#"
        opened += 1
        frames.append([row[:] for row in maze_copy])

        if (x, y) == end:
            for px, py in path:
                maze_copy[px][py] = "o" if (px, py) != start else "S"
            maze_copy[x][y] = 'E'
            frames.append([row[:] for row in maze_copy])
            print("Nodes expanded: ", opened)
            print("Path length: ", len(path) - 1)
            return frames

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                    0 <= nx < len(maze)
                    and 0 <= ny < len(maze[0])
                    and maze_copy[nx][ny] != "X"
                    and (nx, ny) not in visited
            ):
                heapq.heappush(pq, (heuristic((nx, ny), end), (nx, ny), path + [(nx, ny)]))
    return frames


def a_star_with_animation(maze, start, end):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    pq = [(0, start, [start])]
    g_score = {start: 0}
    opened = 0

    maze_copy = [row[:] for row in maze]
    frames = [[row[:] for row in maze_copy]]

    while pq:
        _, (x, y), path = heapq.heappop(pq)
        maze_copy[x][y] = "#"
        opened += 1
        frames.append([row[:] for row in maze_copy])

        if (x, y) == end:
            for px, py in path:
                maze_copy[px][py] = "o" if (px, py) != start else "S"
            maze_copy[x][y] = 'E'
            frames.append([row[:] for row in maze_copy])
            print("Nodes expanded: ", opened)
            print("Path length: ", len(path) - 1)
            return frames

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                    0 <= nx < len(maze)
                    and 0 <= ny < len(maze[0])
                    and maze_copy[nx][ny] != "X"
            ):
                new_g = g_score[(x, y)] + 1
                if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = new_g
                    f_score = new_g + heuristic((nx, ny), end)
                    heapq.heappush(pq, (f_score, (nx, ny), path + [(nx, ny)]))
    return frames


def random_search_with_animation(maze, start, end):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    path = [start]
    visited = set()
    opened = 0
    visited.add(start)

    maze_copy = [row[:] for row in maze]
    frames = [[row[:] for row in maze_copy]]

    while path:
        x, y = path[-1]

        if (x, y) == end:
            for px, py in path:
                maze_copy[px][py] = "o" if (px, py) != start else "S"
            maze_copy[start[0]][start[1]] = "S"
            maze_copy[end[0]][end[1]] = "E"
            frames.append([row[:] for row in maze_copy])
            print("Nodes expanded: ", opened)
            print("Path length: ", len(path))
            break

        visited.add((x, y))
        maze_copy[x][y] = "#"
        maze_copy[start[0]][start[1]] = "S"
        opened += 1
        frames.append([row[:] for row in maze_copy])

        random.shuffle(directions)
        moved = False

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < len(maze)
                and 0 <= ny < len(maze[0])
                and maze_copy[nx][ny] != "X"
                and (nx, ny) not in visited
            ):
                path.append((nx, ny))
                moved = True
                break

        if not moved:
            path.pop()

        if not frames:
            frames.append([row[:] for row in maze_copy])

    return frames


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


if __name__ == "__main__":
    dataset_folder = "testovaci_data"
    maze_files = sorted(
        [f for f in os.listdir(dataset_folder) if f.endswith(".txt")],
        key=lambda x: int(x.split(".")[0])
    )

    algorithms = [
        ("DFS", dfs_with_animation),
        ("BFS", bfs_with_animation),
        ("Greedy", greedy_search_with_animation),
        ("A*", a_star_with_animation),
        ("Random Search", random_search_with_animation),
    ]

    for filename in maze_files:
        print(f"\nProcessing: {filename}")

        filepath = os.path.join(dataset_folder, filename)
        maze, start, end = load_maze(filepath)

        rows = len(maze)
        cols = len(maze[0])
        print(f"Maze size: {rows} rows x {cols} columns")
        sx, sy = start
        ex, ey = end
        maze[sx][sy] = "S"
        maze[ex][ey] = "E"

        if start == end:
            print("Start and end positions are the same! No search needed.")
        else:
            for algo_name, algo_func in algorithms:
                print(f"\n-----> {algo_name} on {filename} <-----")
                frames = algo_func(maze, start, end)
                visualize_maze(frames, f"{algo_name} - {filename}")
                print("-" * 30)
