import pygame
import heapq

pygame.init()

# --- Window and grid settings ---
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 40
GRID_COLS = WIDTH // CELL_SIZE
GRID_ROWS = HEIGHT // CELL_SIZE
GRID_LAYERS = 3  # altitude layers: z = 0 (low) up to z = 2 (high)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Replanner - Stage 5")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)

# --- Colors ---
BG_COLOR = (30, 30, 40)
GRID_COLOR = (60, 60, 75)
START_COLOR = (80, 200, 120)
GOAL_COLOR = (220, 90, 90)
DRONE_COLOR = (90, 160, 230)
DRONE_CLIMB_COLOR = (255, 210, 90)
PATH_COLOR = (120, 120, 160)
OBSTACLE_FULL_COLOR = (100, 60, 60)      # blocked at every altitude
OBSTACLE_PARTIAL_COLOR = (150, 110, 60)  # blocked at SOME altitudes only

VERTICAL_COST = 2  # climbing/descending costs more than moving sideways

# --- 3D obstacles: a set of (col, row, z) ---
obstacles_3d = set()

# Original obstacle clusters -> now blocked at EVERY altitude (still fully impassable)
full_obstacles_2d = [
    (5, 2), (5, 3), (5, 4), (5, 5),
    (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
    (14, 2), (15, 2), (16, 2),
]
for (col, row) in full_obstacles_2d:
    for z in range(GRID_LAYERS):
        obstacles_3d.add((col, row, z))

# New: a "wall" that only blocks LOW altitudes -> forces the drone to climb over it
WALL_COL = 12
for row in range(GRID_ROWS):
    obstacles_3d.add((WALL_COL, row, 0))
    obstacles_3d.add((WALL_COL, row, 1))
    # z = 2 is deliberately left open -> the only way through is to fly high

start_cell = (1, 1, 0)
goal_cell = (18, 13, 0)

# ============================================================
#                    A* PATHFINDING (3D)
# ============================================================

def heuristic(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dz = abs(a[2] - b[2])
    return dx + dy + VERTICAL_COST * dz

def get_neighbors(cell):
    col, row, z = cell
    candidates = [
        (col + 1, row, z), (col - 1, row, z),
        (col, row + 1, z), (col, row - 1, z),
        (col, row, z + 1), (col, row, z - 1),
    ]
    valid = []
    for c in candidates:
        cx, cy, cz = c
        if 0 <= cx < GRID_COLS and 0 <= cy < GRID_ROWS and 0 <= cz < GRID_LAYERS and c not in obstacles_3d:
            valid.append(c)
    return valid

def move_cost(a, b):
    """Cost of moving from cell a to adjacent cell b."""
    if a[2] != b[2]:
        return VERTICAL_COST
    return 1

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def a_star(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + move_cost(current, neighbor)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))
                came_from[neighbor] = current

    return None

path = a_star(start_cell, goal_cell)

if path is None:
    print("WARNING: No path found between start and goal!")
    path = [start_cell]
else:
    print(f"Path found with {len(path)} waypoints:")
    print(path)

# ============================================================
#                      VISUALIZATION
# ============================================================

def grid_to_pixel(cell):
    col, row, z = cell
    x = col * CELL_SIZE + CELL_SIZE // 2
    y = row * CELL_SIZE + CELL_SIZE // 2
    return (x, y)

pixel_path = [grid_to_pixel(cell) for cell in path]

def lerp(a, b, t):
    return a + (b - a) * t

def draw_grid():
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))

def draw_obstacles():
    # Group obstacle cells by (col, row) so we know which altitudes are blocked there
    columns = {}
    for (col, row, z) in obstacles_3d:
        columns.setdefault((col, row), set()).add(z)

    for (col, row), blocked_zs in columns.items():
        rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        if len(blocked_zs) == GRID_LAYERS:
            pygame.draw.rect(screen, OBSTACLE_FULL_COLOR, rect)      # blocked at every altitude
        else:
            pygame.draw.rect(screen, OBSTACLE_PARTIAL_COLOR, rect)   # blocked at some altitudes -> can fly over

def draw_path():
    if len(pixel_path) > 1:
        pygame.draw.lines(screen, PATH_COLOR, False, pixel_path, 3)

# --- Drone movement state ---
current_segment = 0
progress = 0.0
speed = 0.02

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if current_segment < len(path) - 1:
        progress += speed
        if progress >= 1.0:
            progress = 0.0
            current_segment += 1

    is_climbing = False
    if current_segment < len(path) - 1:
        start_point = pixel_path[current_segment]
        end_point = pixel_path[current_segment + 1]
        drone_x = lerp(start_point[0], end_point[0], progress)
        drone_y = lerp(start_point[1], end_point[1], progress)
        current_z = path[current_segment][2]
        is_climbing = path[current_segment][2] != path[current_segment + 1][2]
    else:
        drone_x, drone_y = pixel_path[-1]
        current_z = path[-1][2]

    # --- Drawing ---
    screen.fill(BG_COLOR)
    draw_grid()
    draw_obstacles()
    draw_path()

    pygame.draw.circle(screen, START_COLOR, pixel_path[0], 10)
    pygame.draw.circle(screen, GOAL_COLOR, pixel_path[-1], 10)

    drone_color = DRONE_CLIMB_COLOR if is_climbing else DRONE_COLOR
    pygame.draw.circle(screen, drone_color, (int(drone_x), int(drone_y)), 8)

    # Temporary altitude readout — Stage 6 replaces this with a real side-panel graph
    label = f"Altitude (z): {current_z}" + ("  [CLIMBING/DESCENDING]" if is_climbing else "")
    text_surface = font.render(label, True, (220, 220, 220))
    screen.blit(text_surface, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()