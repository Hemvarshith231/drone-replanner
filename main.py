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
pygame.display.set_caption("Drone Replanner - Stage 9")
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
ASH_COLOR = (200, 160, 60, 110)
TURBULENCE_COLOR = (150, 90, 200, 110)
PANEL_BG_COLOR = (22, 22, 30)
PANEL_LINE_COLOR = (100, 180, 255)
PANEL_AXIS_COLOR = (90, 90, 105)
PANEL_MARKER_COLOR = (255, 210, 90)
PANEL_ASH_BAND_COLOR = (200, 160, 60, 70)
PANEL_TURB_BAND_COLOR = (150, 90, 200, 70)
WIND_ARROW_COLOR = (180, 220, 255)
WARNING_COLOR = (255, 90, 90)

VERTICAL_COST = 2

# --- Hard obstacles ---
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

# --- Ash: NOW MOVES OVER TIME ---
ASH_WIDTH = 4   # number of columns wide
ASH_ROWS = range(8, 14)
ASH_Z_MIN, ASH_Z_MAX = 0, 0
ASH_COST_PER_STEP = 12

ash_start_col = 6         # current left edge of the ash rectangle (this DRIFTS)
ash_drift_direction = 1   # +1 = drifting right, -1 = drifting left
ASH_DRIFT_MIN_COL = 2
ASH_DRIFT_MAX_COL = GRID_COLS - ASH_WIDTH - 1

ASH_TICK_INTERVAL = 45  # ash shifts by 1 column every this many frames
tick_counter = 0

def get_ash_cols():
    return range(ash_start_col, ash_start_col + ASH_WIDTH)

def in_ash(cell):
    col, row, z = cell
    return col in get_ash_cols() and row in ASH_ROWS and ASH_Z_MIN <= z <= ASH_Z_MAX

def update_ash_drift():
    """Called once per tick interval: shifts the ash region by 1 column, bouncing at bounds."""
    global ash_start_col, ash_drift_direction
    ash_start_col += ash_drift_direction
    if ash_start_col >= ASH_DRIFT_MAX_COL:
        ash_start_col = ASH_DRIFT_MAX_COL
        ash_drift_direction = -1
    elif ash_start_col <= ASH_DRIFT_MIN_COL:
        ash_start_col = ASH_DRIFT_MIN_COL
        ash_drift_direction = 1

# --- Turbulence (unchanged, still static for now) ---
TURBULENCE_COLS = range(13, 19)
TURBULENCE_ROWS = range(2, 6)
TURB_Z_MIN, TURB_Z_MAX = 2, 2
TURBULENCE_COST_PER_STEP = 10

def in_turbulence(cell):
    col, row, z = cell
    return col in TURBULENCE_COLS and row in TURBULENCE_ROWS and TURB_Z_MIN <= z <= TURB_Z_MAX

# --- Wind ---
WIND_DIRECTION = (1, 0)
WIND_PENALTY = 3

def wind_cost(a, b):
    if a[2] != b[2]:
        return 0
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dot = dx * WIND_DIRECTION[0] + dy * WIND_DIRECTION[1]
    return WIND_PENALTY if dot < 0 else 0

start_cell = (1, 1, 0)
goal_cell = (18, 13, 0)

# ============================================================
#                    A* PATHFINDING
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
    base = VERTICAL_COST if a[2] != b[2] else 1
    ash = ASH_COST_PER_STEP if in_ash(b) else 0
    turb = TURBULENCE_COST_PER_STEP if in_turbulence(b) else 0
    wind = wind_cost(a, b)
    return base + ash + turb + wind

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

# --- Compute the path ONCE at startup (ash will drift after this, path stays stale on purpose) ---
path = a_star(start_cell, goal_cell)

if path is None:
    print("WARNING: No path found between start and goal!")
    path = [start_cell]
else:
    print(f"Path found with {len(path)} waypoints (computed at t=0, will NOT update as ash drifts).")

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

def draw_hazard_band(z_min, z_max, color, label):
    _, y_top = profile_to_pixel(0, z_max)
    _, y_bottom = profile_to_pixel(0, z_min)
    band_height = max(y_bottom - y_top, 4)
    band_surface = pygame.Surface((panel_rect.width, int(band_height)), pygame.SRCALPHA)
    band_surface.fill(color)
    screen.blit(band_surface, (panel_rect.left, int(y_top)))
    label_surface = font_small.render(label, True, (230, 230, 230))
    screen.blit(label_surface, (panel_rect.left + 4, int(y_top) + 2))

def draw_altitude_panel():
    pygame.draw.rect(screen, PANEL_BG_COLOR, (GRID_WIDTH, 0, PANEL_WIDTH, HEIGHT))

    title = font.render("Altitude Profile", True, (220, 220, 220))
    screen.blit(title, (GRID_WIDTH + PANEL_MARGIN, 25))

    draw_hazard_band(ASH_Z_MIN, ASH_Z_MAX, PANEL_ASH_BAND_COLOR, "ash band")
    draw_hazard_band(TURB_Z_MIN, TURB_Z_MAX, PANEL_TURB_BAND_COLOR, "turbulence band")

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
    ash_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    ash_surface.fill(ASH_COLOR)
    for col in get_ash_cols():
        for row in ASH_ROWS:
            screen.blit(ash_surface, (col * CELL_SIZE, row * CELL_SIZE))

def draw_turbulence():
    turb_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    turb_surface.fill(TURBULENCE_COLOR)
    for col in TURBULENCE_COLS:
        for row in TURBULENCE_ROWS:
            screen.blit(turb_surface, (col * CELL_SIZE, row * CELL_SIZE))

def draw_wind_indicator():
    ax, ay = 90, GRID_HEIGHT - 40
    dx, dy = WIND_DIRECTION
    tip = (ax + dx * 40, ay + dy * 40)
    pygame.draw.line(screen, WIND_ARROW_COLOR, (ax, ay), tip, 3)
    pygame.draw.circle(screen, WIND_ARROW_COLOR, tip, 5)
    label = font_small.render("WIND", True, WIND_ARROW_COLOR)
    screen.blit(label, (ax - 15, ay + 12))

def draw_path():
    if len(pixel_path) > 1:
        pygame.draw.lines(screen, PATH_COLOR, False, pixel_path, 3)

def path_crosses_ash_now():
    """Check if ANY waypoint in the (stale) planned path currently sits inside the drifted ash."""
    return any(in_ash(cell) for cell in path)

# --- Drone movement state ---
current_segment = 0
progress = 0.0
speed = 0.02

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Advance the ash drift on a timer ---
    tick_counter += 1
    if tick_counter >= ASH_TICK_INTERVAL:
        tick_counter = 0
        update_ash_drift()

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
    draw_turbulence()
    draw_obstacles()
    draw_path()
    draw_wind_indicator()

    pygame.draw.circle(screen, START_COLOR, pixel_path[0], 10)
    pygame.draw.circle(screen, GOAL_COLOR, pixel_path[-1], 10)

    drone_color = DRONE_CLIMB_COLOR if is_climbing else DRONE_COLOR
    pygame.draw.circle(screen, drone_color, (int(drone_x), int(drone_y)), 8)

    label = f"Altitude (z): {current_z}" + ("  [CLIMBING/DESCENDING]" if is_climbing else "")
    text_surface = font.render(label, True, (220, 220, 220))
    screen.blit(text_surface, (10, 10))

    # --- Show a warning if the (stale) path now crosses drifted ash ---
    if path_crosses_ash_now():
        warning_surface = font.render("!! PLANNED PATH NOW CROSSES ASH (stale plan) !!", True, WARNING_COLOR)
        screen.blit(warning_surface, (10, 40))

    # --- Drawing: side panel ---
    draw_altitude_panel()
    draw_altitude_marker(current_segment, progress, current_z)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()