import pygame
import heapq

pygame.init()

# --- Window layout ---
GRID_WIDTH, GRID_HEIGHT = 800, 600
PANEL_WIDTH = 260
WIDTH = GRID_WIDTH + PANEL_WIDTH
HEIGHT = GRID_HEIGHT

CELL_SIZE = 40
GRID_COLS = GRID_WIDTH // CELL_SIZE
GRID_ROWS = GRID_HEIGHT // CELL_SIZE
GRID_LAYERS = 3

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Replanner - Stage 7")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
font_small = pygame.font.SysFont(None, 20)

# --- Colors ---
BG_COLOR = (30, 30, 40)
GRID_COLOR = (60, 60, 75)
START_COLOR = (80, 200, 120)
GOAL_COLOR = (220, 90, 90)
DRONE_COLOR = (90, 160, 230)
DRONE_CLIMB_COLOR = (255, 210, 90)
PATH_COLOR = (120, 120, 160)
OBSTACLE_FULL_COLOR = (100, 60, 60)
OBSTACLE_PARTIAL_COLOR = (150, 110, 60)
ASH_COLOR = (200, 160, 60, 110)  # translucent (has alpha)
PANEL_BG_COLOR = (22, 22, 30)
PANEL_LINE_COLOR = (100, 180, 255)
PANEL_AXIS_COLOR = (90, 90, 105)
PANEL_MARKER_COLOR = (255, 210, 90)
PANEL_ASH_BAND_COLOR = (200, 160, 60, 70)

VERTICAL_COST = 2

# --- Hard obstacles (fully impassable, same as Stage 5) ---
obstacles_3d = set()

full_obstacles_2d = [
    (5, 2), (5, 3), (5, 4), (5, 5),
    (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
    (14, 2), (15, 2), (16, 2),
]
for (col, row) in full_obstacles_2d:
    for z in range(GRID_LAYERS):
        obstacles_3d.add((col, row, z))

WALL_COL = 12
for row in range(GRID_ROWS):
    obstacles_3d.add((WALL_COL, row, 0))
    obstacles_3d.add((WALL_COL, row, 1))

# --- NEW: Volcanic ash zone (soft cost, not a hard obstacle) ---
# Occupies a rectangular column/row range, but ONLY at low altitude (z = 0).
ASH_COLS = range(6, 10)     # columns 6,7,8,9
ASH_ROWS = range(8, 14)     # rows 8 through 13
ASH_Z_MIN, ASH_Z_MAX = 0, 0  # only dangerous at z = 0; z = 1 and z = 2 are clear
ASH_COST_PER_STEP = 12       # extra cost added when entering an ash cell in-band

def in_ash(cell):
    col, row, z = cell
    return col in ASH_COLS and row in ASH_ROWS and ASH_Z_MIN <= z <= ASH_Z_MAX

start_cell = (1, 1, 0)
goal_cell = (18, 13, 0)

# ============================================================
#                    A* PATHFINDING (3D + cost map)
# ============================================================

def heuristic(a, b):
    # Still assumes the cheapest possible case (no ash) -> remains a valid lower bound.
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
        # Note: ash cells are NOT removed here -> they remain enterable, just costly.
        if 0 <= cx < GRID_COLS and 0 <= cy < GRID_ROWS and 0 <= cz < GRID_LAYERS and c not in obstacles_3d:
            valid.append(c)
    return valid

def move_cost(a, b):
    """Cost of moving from cell a into adjacent cell b: base movement cost + hazard cost of entering b."""
    base = VERTICAL_COST if a[2] != b[2] else 1
    hazard = ASH_COST_PER_STEP if in_ash(b) else 0
    return base + hazard

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
    total_cost = sum(move_cost(path[i], path[i + 1]) for i in range(len(path) - 1))
    ash_cells_crossed = sum(1 for cell in path if in_ash(cell))
    print(f"Path found with {len(path)} waypoints.")
    print(f"Total path cost: {total_cost}")
    print(f"Ash-zone cells crossed: {ash_cells_crossed}  (0 means the drone climbed above it)")

# ============================================================
#              ALTITUDE PROFILE DATA (side panel)
# ============================================================

def compute_cumulative_distances(path):
    distances = [0]
    for i in range(1, len(path)):
        distances.append(distances[-1] + move_cost(path[i - 1], path[i]))
    return distances

cumulative_distances = compute_cumulative_distances(path)
total_distance = cumulative_distances[-1] if cumulative_distances else 1
max_altitude = max(GRID_LAYERS - 1, 1)

PANEL_MARGIN = 20
panel_rect = pygame.Rect(
    GRID_WIDTH + PANEL_MARGIN,
    60,
    PANEL_WIDTH - PANEL_MARGIN * 2,
    HEIGHT - 140,
)

def profile_to_pixel(distance, altitude):
    tx = 0 if total_distance == 0 else distance / total_distance
    ty = altitude / max_altitude
    x = panel_rect.left + tx * panel_rect.width
    y = panel_rect.bottom - ty * panel_rect.height
    return (x, y)

profile_points = [
    profile_to_pixel(cumulative_distances[i], path[i][2])
    for i in range(len(path))
]

def draw_altitude_panel():
    pygame.draw.rect(screen, PANEL_BG_COLOR, (GRID_WIDTH, 0, PANEL_WIDTH, HEIGHT))

    title = font.render("Altitude Profile", True, (220, 220, 220))
    screen.blit(title, (GRID_WIDTH + PANEL_MARGIN, 25))

    # Shaded band showing WHERE the ash danger zone sits, altitude-wise
    _, y_top = profile_to_pixel(0, ASH_Z_MAX)
    _, y_bottom = profile_to_pixel(0, ASH_Z_MIN)
    band_height = max(y_bottom - y_top, 4)
    band_surface = pygame.Surface((panel_rect.width, int(band_height)), pygame.SRCALPHA)
    band_surface.fill(PANEL_ASH_BAND_COLOR)
    screen.blit(band_surface, (panel_rect.left, int(y_top)))
    band_label = font_small.render("ash band", True, (200, 170, 100))
    screen.blit(band_label, (panel_rect.left + 4, int(y_top) + 2))

    pygame.draw.line(screen, PANEL_AXIS_COLOR, panel_rect.bottomleft, panel_rect.topleft, 2)
    pygame.draw.line(screen, PANEL_AXIS_COLOR, panel_rect.bottomleft, panel_rect.bottomright, 2)

    for z in range(GRID_LAYERS):
        _, y = profile_to_pixel(0, z)
        label = font_small.render(f"z={z}", True, (150, 150, 165))
        screen.blit(label, (panel_rect.left - 32, y - 8))
        pygame.draw.line(screen, (45, 45, 55), (panel_rect.left, y), (panel_rect.right, y), 1)

    x_label = font_small.render("distance traveled ->", True, (150, 150, 165))
    screen.blit(x_label, (panel_rect.left, panel_rect.bottom + 8))

    if len(profile_points) > 1:
        pygame.draw.lines(screen, PANEL_LINE_COLOR, False, profile_points, 3)
    for point in profile_points:
        pygame.draw.circle(screen, PANEL_LINE_COLOR, (int(point[0]), int(point[1])), 3)

def draw_altitude_marker(segment_index, progress, current_z):
    if segment_index >= len(cumulative_distances) - 1:
        dist_now = cumulative_distances[-1]
    else:
        d_start = cumulative_distances[segment_index]
        d_end = cumulative_distances[segment_index + 1]
        dist_now = d_start + (d_end - d_start) * progress

    x, y = profile_to_pixel(dist_now, current_z)
    pygame.draw.circle(screen, PANEL_MARKER_COLOR, (int(x), int(y)), 6)
    pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 6, 1)

# ============================================================
#                      MAIN VISUALIZATION
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
    for x in range(0, GRID_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, GRID_HEIGHT))
    for y in range(0, GRID_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (GRID_WIDTH, y))

def draw_obstacles():
    columns = {}
    for (col, row, z) in obstacles_3d:
        columns.setdefault((col, row), set()).add(z)
    for (col, row), blocked_zs in columns.items():
        rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        if len(blocked_zs) == GRID_LAYERS:
            pygame.draw.rect(screen, OBSTACLE_FULL_COLOR, rect)
        else:
            pygame.draw.rect(screen, OBSTACLE_PARTIAL_COLOR, rect)

def draw_ash():
    """Draw a translucent overlay over ash-zone cells (top-down; remember it's only dangerous at z=0)."""
    ash_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    ash_surface.fill(ASH_COLOR)
    for col in ASH_COLS:
        for row in ASH_ROWS:
            screen.blit(ash_surface, (col * CELL_SIZE, row * CELL_SIZE))

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

    # --- Drawing: main grid area ---
    screen.fill(BG_COLOR)
    draw_grid()
    draw_ash()
    draw_obstacles()
    draw_path()

    pygame.draw.circle(screen, START_COLOR, pixel_path[0], 10)
    pygame.draw.circle(screen, GOAL_COLOR, pixel_path[-1], 10)

    drone_color = DRONE_CLIMB_COLOR if is_climbing else DRONE_COLOR
    pygame.draw.circle(screen, drone_color, (int(drone_x), int(drone_y)), 8)

    label = f"Altitude (z): {current_z}" + ("  [CLIMBING/DESCENDING]" if is_climbing else "")
    text_surface = font.render(label, True, (220, 220, 220))
    screen.blit(text_surface, (10, 10))

    # --- Drawing: side panel ---
    draw_altitude_panel()
    draw_altitude_marker(current_segment, progress, current_z)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()