import pygame
import heapq
import time
import math

pygame.init()

# --- Window layout ---
GRID_WIDTH, GRID_HEIGHT = 800, 600
PANEL_WIDTH = 260
DASHBOARD_HEIGHT = 90
WIDTH = GRID_WIDTH + PANEL_WIDTH
HEIGHT = GRID_HEIGHT + DASHBOARD_HEIGHT

CELL_SIZE = 40
GRID_COLS = GRID_WIDTH // CELL_SIZE
GRID_ROWS = GRID_HEIGHT // CELL_SIZE
GRID_LAYERS = 3

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Adaptive Drone Flight-Path Replanner")
clock = pygame.time.Clock()
font_title = pygame.font.SysFont("Segoe UI", 22, bold=True)
font = pygame.font.SysFont("Segoe UI", 20)
font_small = pygame.font.SysFont("Segoe UI", 16)
font_tiny = pygame.font.SysFont("Segoe UI", 13)

# --- Unified color palette ---
BG_COLOR = (24, 25, 33)
GRID_COLOR = (42, 43, 56)
START_COLOR = (86, 211, 145)
GOAL_COLOR = (232, 93, 93)
DRONE_COLOR = (94, 170, 240)
DRONE_CLIMB_COLOR = (247, 191, 85)
PATH_COLOR = (110, 112, 145)
HAZARD_PATH_COLOR = (232, 93, 93)
OBSTACLE_FULL_COLOR = (95, 55, 55)
OBSTACLE_PARTIAL_COLOR = (150, 105, 55)
ASH_COLOR = (214, 168, 60, 115)
TURBULENCE_COLOR = (150, 90, 210, 115)
PANEL_BG_COLOR = (18, 19, 26)
PANEL_LINE_COLOR = (94, 170, 240)
PANEL_AXIS_COLOR = (70, 72, 92)
PANEL_MARKER_COLOR = (247, 191, 85)
PANEL_ASH_BAND_COLOR = (214, 168, 60, 65)
PANEL_TURB_BAND_COLOR = (150, 90, 210, 65)
WIND_ARROW_COLOR = (170, 210, 250)
WARNING_COLOR = (240, 110, 110)
SAFE_COLOR = (120, 220, 150)
REPLAN_FLASH_COLOR = (247, 191, 85)
DASHBOARD_BG_COLOR = (15, 16, 22)
DASHBOARD_LABEL_COLOR = (140, 142, 160)
DASHBOARD_VALUE_COLOR = (232, 232, 240)
TEXT_MAIN = (225, 226, 235)
LEGEND_BG_COLOR = (18, 19, 26)

VERTICAL_COST = 2
ENERGY_PER_COST_UNIT = 0.5

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

# --- Ash ---
ASH_WIDTH = 4
ASH_ROWS = range(9, 12)
ASH_Z_MIN, ASH_Z_MAX = 0, 0
ASH_COST_PER_STEP = 12

ash_start_col = 6
ash_drift_direction = 1
ASH_DRIFT_MIN_COL = 2
ASH_DRIFT_MAX_COL = GRID_COLS - ASH_WIDTH - 1

ASH_TICK_INTERVAL = 45
tick_counter = 0

def get_ash_cols():
    return range(ash_start_col, ash_start_col + ASH_WIDTH)

def in_ash(cell):
    col, row, z = cell
    return col in get_ash_cols() and row in ASH_ROWS and ASH_Z_MIN <= z <= ASH_Z_MAX

def update_ash_drift():
    global ash_start_col, ash_drift_direction
    ash_start_col += ash_drift_direction
    if ash_start_col >= ASH_DRIFT_MAX_COL:
        ash_start_col = ASH_DRIFT_MAX_COL
        ash_drift_direction = -1
    elif ash_start_col <= ASH_DRIFT_MIN_COL:
        ash_start_col = ASH_DRIFT_MIN_COL
        ash_drift_direction = 1

# --- Turbulence ---
TURBULENCE_COLS = range(13, 19)
TURBULENCE_ROWS = range(2, 6)
TURB_Z_MIN, TURB_Z_MAX = 2, 2
TURBULENCE_COST_PER_STEP = 10

def in_turbulence(cell):
    col, row, z = cell
    return col in TURBULENCE_COLS and row in TURBULENCE_ROWS and TURB_Z_MIN <= z <= TURB_Z_MAX

def in_any_hazard(cell):
    return in_ash(cell) or in_turbulence(cell)

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
goal_cell = (18, 3, 0)

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
    result = [current]
    while current in came_from:
        current = came_from[current]
        result.append(current)
    result.reverse()
    return result

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

# ============================================================
#         PATH STATE + DERIVED-DATA RECOMPUTATION
# ============================================================

path = a_star(start_cell, goal_cell)
if path is None:
    print("WARNING: No initial path found between start and goal!")
    path = [start_cell]
else:
    print(f"Initial path found with {len(path)} waypoints.")

def grid_to_pixel(cell):
    col, row, z = cell
    x = col * CELL_SIZE + CELL_SIZE // 2
    y = row * CELL_SIZE + CELL_SIZE // 2
    return (x, y)

def compute_cumulative_distances(p):
    distances = [0]
    for i in range(1, len(p)):
        distances.append(distances[-1] + move_cost(p[i - 1], p[i]))
    return distances

pixel_path = []
cumulative_distances = []
total_distance = 1
max_altitude = max(GRID_LAYERS - 1, 1)
profile_points = []

def recompute_derived_data():
    global pixel_path, cumulative_distances, total_distance, profile_points
    pixel_path = [grid_to_pixel(cell) for cell in path]
    cumulative_distances = compute_cumulative_distances(path)
    total_distance = cumulative_distances[-1] if cumulative_distances else 1
    profile_points = [
        profile_to_pixel(cumulative_distances[i], path[i][2])
        for i in range(len(path))
    ]

replan_count = 0
last_replan_ms = 0.0

def trigger_replan(current_segment):
    global path, replan_count, last_replan_ms

    if current_segment + 1 >= len(path):
        return

    replan_from = path[current_segment + 1]

    start_time = time.perf_counter()
    new_remaining = a_star(replan_from, goal_cell)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if new_remaining is None:
        print("[REPLAN FAILED] No alternate route found -- keeping existing plan.")
        return

    path = path[:current_segment + 1] + new_remaining
    replan_count += 1
    last_replan_ms = elapsed_ms
    recompute_derived_data()
    print(f"[REPLAN #{replan_count}] New route has {len(path)} waypoints. Computed in {elapsed_ms:.3f} ms.")

# ============================================================
#              HAZARD-AHEAD DETECTION
# ============================================================

def get_remaining_path(full_path, current_segment):
    return full_path[current_segment:]

def detect_hazard_ahead(full_path, current_segment):
    remaining = get_remaining_path(full_path, current_segment)
    hazard_cells = [cell for cell in remaining if in_any_hazard(cell)]
    return (len(hazard_cells) > 0), hazard_cells

hazard_was_detected = False
replan_flash_timer = 0

# ============================================================
#              MISSION-WIDE METRICS
# ============================================================

mission_start_ticks = pygame.time.get_ticks()
hazard_cells_ever_crossed = 0
last_counted_segment = -1

def mission_elapsed_seconds():
    return (pygame.time.get_ticks() - mission_start_ticks) / 1000.0

def current_path_total_cost():
    return cumulative_distances[-1] if cumulative_distances else 0

def current_path_altitude_changes():
    changes = 0
    for i in range(1, len(path)):
        if path[i][2] != path[i - 1][2]:
            changes += 1
    return changes

def estimated_energy(total_cost):
    return total_cost * ENERGY_PER_COST_UNIT

# ============================================================
#              ALTITUDE PROFILE DATA (side panel)
# ============================================================

PANEL_MARGIN = 20
panel_rect = pygame.Rect(
    GRID_WIDTH + PANEL_MARGIN,
    70,
    PANEL_WIDTH - PANEL_MARGIN * 2,
    GRID_HEIGHT - 150,
)

def profile_to_pixel(distance, altitude):
    tx = 0 if total_distance == 0 else distance / total_distance
    ty = altitude / max_altitude
    x = panel_rect.left + tx * panel_rect.width
    y = panel_rect.bottom - ty * panel_rect.height
    return (x, y)

recompute_derived_data()

def draw_hazard_band(z_min, z_max, color, label):
    _, y_top = profile_to_pixel(0, z_max)
    _, y_bottom = profile_to_pixel(0, z_min)
    band_height = max(y_bottom - y_top, 4)
    band_surface = pygame.Surface((panel_rect.width, int(band_height)), pygame.SRCALPHA)
    band_surface.fill(color)
    screen.blit(band_surface, (panel_rect.left, int(y_top)))
    label_surface = font_tiny.render(label, True, (235, 235, 240))
    screen.blit(label_surface, (panel_rect.left + 4, int(y_top) + 2))

def draw_altitude_panel():
    pygame.draw.rect(screen, PANEL_BG_COLOR, (GRID_WIDTH, 0, PANEL_WIDTH, GRID_HEIGHT))
    pygame.draw.line(screen, PANEL_AXIS_COLOR, (GRID_WIDTH, 0), (GRID_WIDTH, GRID_HEIGHT), 2)

    title = font_title.render("Altitude Profile", True, TEXT_MAIN)
    screen.blit(title, (GRID_WIDTH + PANEL_MARGIN, 22))

    draw_hazard_band(ASH_Z_MIN, ASH_Z_MAX, PANEL_ASH_BAND_COLOR, "ash band")
    draw_hazard_band(TURB_Z_MIN, TURB_Z_MAX, PANEL_TURB_BAND_COLOR, "turbulence band")

    pygame.draw.line(screen, PANEL_AXIS_COLOR, panel_rect.bottomleft, panel_rect.topleft, 2)
    pygame.draw.line(screen, PANEL_AXIS_COLOR, panel_rect.bottomleft, panel_rect.bottomright, 2)

    for z in range(GRID_LAYERS):
        _, y = profile_to_pixel(0, z)
        label = font_tiny.render(f"z={z}", True, (150, 150, 165))
        screen.blit(label, (panel_rect.left - 30, y - 7))
        pygame.draw.line(screen, (38, 39, 50), (panel_rect.left, y), (panel_rect.right, y), 1)

    x_label = font_tiny.render("distance traveled  ->", True, (150, 150, 165))
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
    pygame.draw.circle(screen, PANEL_MARKER_COLOR, (int(x), int(y)), 7)
    pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 7, 1)

# ============================================================
#              DASHBOARD (bottom strip)
# ============================================================

dashboard_rect = pygame.Rect(0, GRID_HEIGHT, WIDTH, DASHBOARD_HEIGHT)

def draw_dashboard(hazard_ahead):
    pygame.draw.rect(screen, DASHBOARD_BG_COLOR, dashboard_rect)
    pygame.draw.line(screen, PANEL_AXIS_COLOR, (0, GRID_HEIGHT), (WIDTH, GRID_HEIGHT), 2)

    total_cost = current_path_total_cost()
    alt_changes = current_path_altitude_changes()
    energy = estimated_energy(total_cost)
    elapsed = mission_elapsed_seconds()

    col1 = [
        ("MISSION TIME", f"{elapsed:.1f} s"),
        ("CURRENT PATH COST", f"{total_cost}"),
        ("ALTITUDE CHANGES", f"{alt_changes}"),
    ]
    col2 = [
        ("HAZARD CELLS CROSSED", f"{hazard_cells_ever_crossed}"),
        ("REPLANS TRIGGERED", f"{replan_count}"),
        ("LAST REPLAN TIME", f"{last_replan_ms:.3f} ms"),
    ]
    col3 = [
        ("EST. ENERGY (illustrative)", f"{energy:.1f} units"),
        ("ROUTE STATUS", "HAZARD AHEAD" if hazard_ahead else "CLEAR"),
    ]

    def draw_column(items, x):
        y = GRID_HEIGHT + 10
        for label, value in items:
            label_surf = font_tiny.render(label, True, DASHBOARD_LABEL_COLOR)
            value_color = WARNING_COLOR if (label == "ROUTE STATUS" and hazard_ahead) else DASHBOARD_VALUE_COLOR
            value_surf = font.render(value, True, value_color)
            screen.blit(label_surf, (x, y))
            screen.blit(value_surf, (x, y + 15))
            y += 33

    draw_column(col1, 16)
    draw_column(col2, 290)
    draw_column(col3, 570)

    caption = font_tiny.render(
        "Energy is illustrative: proportional to path cost, not a calibrated power model.",
        True, (100, 100, 112)
    )
    screen.blit(caption, (16, HEIGHT - 18))

# ============================================================
#              LEGEND
# ============================================================

def draw_legend():
    """Small always-visible legend in the top-right corner of the grid area."""
    items = [
        (START_COLOR, "Start"),
        (GOAL_COLOR, "Goal"),
        (DRONE_COLOR, "Drone (level flight)"),
        (DRONE_CLIMB_COLOR, "Drone (climbing/descending)"),
        (OBSTACLE_FULL_COLOR, "Obstacle (all altitudes)"),
        (OBSTACLE_PARTIAL_COLOR, "Obstacle (some altitudes)"),
        (ASH_COLOR[:3], "Volcanic ash (moving)"),
        (TURBULENCE_COLOR[:3], "Turbulence"),
    ]

    padding = 10
    row_height = 20
    box_width = 230
    box_height = padding * 2 + row_height * len(items)
    box = pygame.Rect(GRID_WIDTH - box_width - 12, 12, box_width, box_height)

    legend_surface = pygame.Surface((box.width, box.height), pygame.SRCALPHA)
    legend_surface.fill((*LEGEND_BG_COLOR, 210))
    screen.blit(legend_surface, box.topleft)
    pygame.draw.rect(screen, PANEL_AXIS_COLOR, box, 1)

    y = box.top + padding
    for color, label in items:
        pygame.draw.rect(screen, color, (box.left + padding, y + 4, 12, 12))
        text = font_tiny.render(label, True, TEXT_MAIN)
        screen.blit(text, (box.left + padding + 20, y))
        y += row_height

# ============================================================
#              MAIN VISUALIZATION
# ============================================================

def ease_in_out(t):
    """Smooth acceleration/deceleration instead of constant linear speed."""
    return 0.5 - 0.5 * math.cos(t * math.pi)

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
    label = font_tiny.render("WIND", True, WIND_ARROW_COLOR)
    screen.blit(label, (ax - 14, ay + 12))

def draw_path(hazard_ahead):
    color = HAZARD_PATH_COLOR if hazard_ahead else PATH_COLOR
    if len(pixel_path) > 1:
        pygame.draw.lines(screen, color, False, pixel_path, 3)

def draw_hazard_status(hazard_ahead, hazard_cells):
    if hazard_ahead:
        msg = f"HAZARD AHEAD  --  {len(hazard_cells)} cell(s) on remaining route"
        surf = font.render(msg, True, WARNING_COLOR)
    else:
        msg = "Remaining route: clear"
        surf = font.render(msg, True, SAFE_COLOR)
    screen.blit(surf, (12, 44))

def draw_replan_stats(flash):
    color = REPLAN_FLASH_COLOR if flash else (190, 190, 205)
    msg = f"Replans: {replan_count}    Last replan: {last_replan_ms:.3f} ms"
    surf = font_small.render(msg, True, color)
    screen.blit(surf, (12, 74))

# --- Drone movement state ---
current_segment = 0
progress = 0.0
speed = 0.02

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    tick_counter += 1
    if tick_counter >= ASH_TICK_INTERVAL:
        tick_counter = 0
        update_ash_drift()

    if current_segment < len(path) - 1:
        progress += speed
        if progress >= 1.0:
            progress = 0.0
            if current_segment != last_counted_segment:
                if in_any_hazard(path[current_segment]):
                    hazard_cells_ever_crossed += 1
                last_counted_segment = current_segment
            current_segment += 1

    is_climbing = False
    if current_segment < len(path) - 1:
        start_point = pixel_path[current_segment]
        end_point = pixel_path[current_segment + 1]
        eased_progress = ease_in_out(progress)  # smoother motion, same underlying timing
        drone_x = lerp(start_point[0], end_point[0], eased_progress)
        drone_y = lerp(start_point[1], end_point[1], eased_progress)
        current_z = path[current_segment][2]
        is_climbing = path[current_segment][2] != path[current_segment + 1][2]
    else:
        drone_x, drone_y = pixel_path[-1]
        current_z = path[-1][2]

    hazard_ahead, hazard_cells = detect_hazard_ahead(path, current_segment)

    if hazard_ahead and not hazard_was_detected:
        print(f"[EVENT] Hazard newly detected ahead! {len(hazard_cells)} cell(s): {hazard_cells}")
        trigger_replan(current_segment)
        replan_flash_timer = 30
        hazard_ahead, hazard_cells = detect_hazard_ahead(path, current_segment)
    elif not hazard_ahead and hazard_was_detected:
        print("[EVENT] Hazard ahead has cleared.")
    hazard_was_detected = hazard_ahead

    if replan_flash_timer > 0:
        replan_flash_timer -= 1

    # --- Drawing: main grid area ---
    screen.fill(BG_COLOR)
    draw_grid()
    draw_ash()
    draw_turbulence()
    draw_obstacles()
    draw_path(hazard_ahead)
    draw_wind_indicator()

    pygame.draw.circle(screen, START_COLOR, pixel_path[0], 10)
    pygame.draw.circle(screen, GOAL_COLOR, pixel_path[-1], 10)

    drone_color = DRONE_CLIMB_COLOR if is_climbing else DRONE_COLOR
    # Subtle glow ring behind the drone for a slightly more polished look
    glow_surface = pygame.Surface((28, 28), pygame.SRCALPHA)
    pygame.draw.circle(glow_surface, (*drone_color, 60), (14, 14), 14)
    screen.blit(glow_surface, (int(drone_x) - 14, int(drone_y) - 14))
    pygame.draw.circle(screen, drone_color, (int(drone_x), int(drone_y)), 8)

    title_label = font_title.render("Adaptive Flight-Path Replanner", True, TEXT_MAIN)
    screen.blit(title_label, (12, 10))

    alt_label = font_small.render(
        f"Altitude z={current_z}" + ("  (climbing/descending)" if is_climbing else ""),
        True, (190, 190, 205)
    )
    screen.blit(alt_label, (12, GRID_HEIGHT - 26))

    draw_hazard_status(hazard_ahead, hazard_cells)
    draw_replan_stats(replan_flash_timer > 0)
    draw_legend()

    draw_altitude_panel()
    draw_altitude_marker(current_segment, progress, current_z)

    draw_dashboard(hazard_ahead)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()