import pygame
import heapq
import time
import math

pygame.init()

# --- Window layout ---
GRID_WIDTH, GRID_HEIGHT = 800, 600
PANEL_WIDTH = 320
DASHBOARD_HEIGHT = 90
WIDTH = GRID_WIDTH + PANEL_WIDTH
HEIGHT = GRID_HEIGHT + DASHBOARD_HEIGHT

CELL_SIZE = 40
GRID_COLS = GRID_WIDTH // CELL_SIZE
GRID_ROWS = GRID_HEIGHT // CELL_SIZE

# --- Altitude system: 0 to 40,000 ft in 5,000 ft steps (9 levels) ---
ALTITUDE_STEP_FT = 5000
GRID_LAYERS = 9
MAX_ALTITUDE_FT = (GRID_LAYERS - 1) * ALTITUDE_STEP_FT

def alt_ft(z_index):
    return z_index * ALTITUDE_STEP_FT

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Adaptive Drone Flight-Path Replanner")
clock = pygame.time.Clock()
font_title = pygame.font.SysFont("Segoe UI", 22, bold=True)
font = pygame.font.SysFont("Segoe UI", 20)
font_small = pygame.font.SysFont("Segoe UI", 16)
font_tiny = pygame.font.SysFont("Segoe UI", 13)

# --- Colors ---
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
BUTTON_COLOR = (55, 57, 74)
BUTTON_ACTIVE_COLOR = (80, 140, 210)
BUTTON_START_COLOR = (70, 160, 110)
BUTTON_OBSTACLE_COLOR = (140, 90, 60)

VERTICAL_COST = 2
ENERGY_PER_COST_UNIT = 0.5
WALL_OPEN_MIN_Z = 4  # kept as a reference constant for the default wall obstacle below

# ============================================================
#              GENERALIZED OBSTACLES (now editable)
# ============================================================
# Each obstacle is a rectangular col/row footprint plus an altitude
# band (z_min..z_max). Full-height = spans every altitude level.
# Partial = blocks only some altitudes (e.g. the "wall" below).

def make_obstacle(obstacle_id, col_start, width, row_start, height, z_min, z_max):
    return {
        "id": obstacle_id,
        "col_start": col_start, "width": width,
        "row_start": row_start, "height": height,
        "z_min": z_min, "z_max": z_max,
    }

def obstacle_cols(o):
    return range(o["col_start"], o["col_start"] + o["width"])

def obstacle_rows(o):
    return range(o["row_start"], o["row_start"] + o["height"])

def in_obstacle(cell, o):
    col, row, z = cell
    return col in obstacle_cols(o) and row in obstacle_rows(o) and o["z_min"] <= z <= o["z_max"]

def in_any_obstacle(cell):
    return any(in_obstacle(cell, o) for o in obstacles)

def get_selected_obstacle():
    return next((o for o in obstacles if o["id"] == selected_obstacle_id), None)

obstacles = [
    make_obstacle(1, 5, 1, 2, 4, 0, GRID_LAYERS - 1),                    # full-height block
    make_obstacle(2, 10, 1, 6, 5, 0, GRID_LAYERS - 1),                   # full-height block
    make_obstacle(3, 14, 3, 2, 1, 0, GRID_LAYERS - 1),                   # full-height block
    make_obstacle(4, 12, 1, 0, GRID_ROWS, 0, WALL_OPEN_MIN_Z - 1),       # the "wall" -- partial altitude only
]
next_obstacle_id = 5

# --- Wind (fixed, not configurable this round) ---
WIND_DIRECTION = (1, 0)
WIND_PENALTY = 3

def wind_cost(a, b):
    if a[2] != b[2]:
        return 0
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dot = dx * WIND_DIRECTION[0] + dy * WIND_DIRECTION[1]
    return WIND_PENALTY if dot < 0 else 0

# ============================================================
#                    GENERALIZED HAZARDS
# ============================================================

SPEED_PRESETS = [15, 30, 45, 60, 90]

def make_hazard(hazard_id, htype, col_start, width, row_start, height):
    if htype == "ash":
        cost, color, band_color, label = 12, ASH_COLOR, PANEL_ASH_BAND_COLOR, "Ash"
    else:
        cost, color, band_color, label = 10, TURBULENCE_COLOR, PANEL_TURB_BAND_COLOR, "Turbulence"
    return {
        "id": hazard_id, "type": htype, "label": label,
        "cost": cost, "color": color, "band_color": band_color,
        "col_start": col_start, "width": width,
        "row_start": row_start, "height": height,
        "z_min": 0, "z_max": 0,
        "drift_enabled": False, "drift_dir": 1,
        "drift_interval": 45, "tick_counter": 0,
    }

# Default demo hazards -- see altitude reasoning notes from the previous fix:
#   Wall open:    20,000 - 40,000 ft (z = 4..8)
#   Ash:          ground level, 0 ft
#   Turbulence:   35,000 ft only -- leaves 20k/25k/30k/40k ft as safe crossing options
hazards = [
    make_hazard(1, "ash", col_start=6, width=4, row_start=9, height=3),
    make_hazard(2, "turbulence", col_start=13, width=6, row_start=2, height=4),
]
hazards[0]["z_min"] = hazards[0]["z_max"] = 0
hazards[0]["drift_enabled"] = True
hazards[1]["z_min"] = hazards[1]["z_max"] = 7
next_hazard_id = 3

def hazard_cols(h):
    return range(h["col_start"], h["col_start"] + h["width"])

def hazard_rows(h):
    return range(h["row_start"], h["row_start"] + h["height"])

def in_hazard(cell, h):
    col, row, z = cell
    return col in hazard_cols(h) and row in hazard_rows(h) and h["z_min"] <= z <= h["z_max"]

def in_any_hazard(cell):
    return any(in_hazard(cell, h) for h in hazards)

def update_hazard_drift(h):
    if not h["drift_enabled"]:
        return
    h["tick_counter"] += 1
    if h["tick_counter"] >= h["drift_interval"]:
        h["tick_counter"] = 0
        col_min = 2
        col_max = max(GRID_COLS - h["width"] - 1, col_min)
        h["col_start"] += h["drift_dir"]
        if h["col_start"] >= col_max:
            h["col_start"] = col_max
            h["drift_dir"] = -1
        elif h["col_start"] <= col_min:
            h["col_start"] = col_min
            h["drift_dir"] = 1

def get_selected_hazard():
    return next((h for h in hazards if h["id"] == selected_hazard_id), None)

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
        if 0 <= cx < GRID_COLS and 0 <= cy < GRID_ROWS and 0 <= cz < GRID_LAYERS and not in_any_obstacle(c):
            valid.append(c)
    return valid

def move_cost(a, b):
    base = VERTICAL_COST if a[2] != b[2] else 1
    hazard_total = sum(h["cost"] for h in hazards if in_hazard(b, h))
    wind = wind_cost(a, b)
    return base + hazard_total + wind

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

path = []
pixel_path = []
cumulative_distances = []
total_distance = 1
profile_points = []

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

def recompute_derived_data():
    global pixel_path, cumulative_distances, total_distance, profile_points
    pixel_path = [grid_to_pixel(cell) for cell in path]
    cumulative_distances = compute_cumulative_distances(path)
    total_distance = cumulative_distances[-1] if cumulative_distances else 1
    profile_points = [
        profile_to_pixel(cumulative_distances[i], alt_ft(path[i][2]))
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
#              MISSION-WIDE METRICS + COMPLETION STATE
# ============================================================

mission_start_ticks = pygame.time.get_ticks()
mission_end_ticks = mission_start_ticks
mission_complete = False
hazard_cells_ever_crossed = 0
last_counted_segment = -1

def mission_elapsed_seconds():
    end = mission_end_ticks if mission_complete else pygame.time.get_ticks()
    return (end - mission_start_ticks) / 1000.0

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
#         ALTITUDE PROFILE DATA (side panel, RUNNING only)
# ============================================================

PANEL_MARGIN = 20
TICK_LABEL_GUTTER = 84

panel_rect = pygame.Rect(
    GRID_WIDTH + TICK_LABEL_GUTTER,
    70,
    PANEL_WIDTH - TICK_LABEL_GUTTER - PANEL_MARGIN,
    GRID_HEIGHT - 150,
)

def profile_to_pixel(distance, altitude_ft):
    tx = 0 if total_distance == 0 else distance / total_distance
    ty = 0 if MAX_ALTITUDE_FT == 0 else altitude_ft / MAX_ALTITUDE_FT
    x = panel_rect.left + tx * panel_rect.width
    y = panel_rect.bottom - ty * panel_rect.height
    return (x, y)

def draw_hazard_band(alt_min_ft, alt_max_ft, color, label):
    _, y_top = profile_to_pixel(0, alt_max_ft)
    _, y_bottom = profile_to_pixel(0, alt_min_ft)
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
    screen.blit(title, (GRID_WIDTH + 16, 22))

    for h in hazards:
        draw_hazard_band(alt_ft(h["z_min"]), alt_ft(h["z_max"]), h["band_color"], f'{h["label"]} #{h["id"]}')

    pygame.draw.line(screen, PANEL_AXIS_COLOR, panel_rect.bottomleft, panel_rect.topleft, 2)
    pygame.draw.line(screen, PANEL_AXIS_COLOR, panel_rect.bottomleft, panel_rect.bottomright, 2)

    for z in range(GRID_LAYERS):
        ft = alt_ft(z)
        _, y = profile_to_pixel(0, ft)
        label = font_tiny.render(f"{ft:,} ft", True, (150, 150, 165))
        screen.blit(label, (panel_rect.left - label.get_width() - 6, y - 7))
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

    x, y = profile_to_pixel(dist_now, alt_ft(current_z))
    pygame.draw.circle(screen, PANEL_MARKER_COLOR, (int(x), int(y)), 7)
    pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 7, 1)

# ============================================================
#              DASHBOARD (bottom strip, RUNNING only)
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
        ("ROUTE STATUS", "MISSION COMPLETE" if mission_complete else ("HAZARD AHEAD" if hazard_ahead else "CLEAR")),
    ]

    def draw_column(items, x):
        y = GRID_HEIGHT + 10
        for label, value in items:
            label_surf = font_tiny.render(label, True, DASHBOARD_LABEL_COLOR)
            value_color = DASHBOARD_VALUE_COLOR
            if label == "ROUTE STATUS":
                value_color = SAFE_COLOR if mission_complete else (WARNING_COLOR if hazard_ahead else DASHBOARD_VALUE_COLOR)
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

    press_r = font_tiny.render("Press R to return to setup", True, (120, 120, 135))
    screen.blit(press_r, (WIDTH - 190, HEIGHT - 18))

# ============================================================
#              LEGEND (RUNNING only)
# ============================================================

def draw_legend():
    items = [
        (START_COLOR, "Start"),
        (GOAL_COLOR, "Goal"),
        (DRONE_COLOR, "Drone (level flight)"),
        (DRONE_CLIMB_COLOR, "Drone (climbing/descending)"),
        (OBSTACLE_FULL_COLOR, "Obstacle (all altitudes)"),
        (OBSTACLE_PARTIAL_COLOR, "Obstacle (some altitudes)"),
    ]
    for h in hazards:
        items.append((h["color"][:3], f'{h["label"]} #{h["id"]}'))

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

def draw_mission_complete_banner():
    line1 = "MISSION COMPLETE  --  Goal Reached"
    line2 = "Press R to return to setup and run again"
    surf1 = font.render(line1, True, (255, 255, 255))
    surf2 = font_tiny.render(line2, True, (235, 235, 235))
    box_w = max(surf1.get_width(), surf2.get_width()) + 40
    box_h = surf1.get_height() + surf2.get_height() + 26
    box = pygame.Rect((GRID_WIDTH - box_w) // 2, 16, box_w, box_h)

    banner_surface = pygame.Surface((box.width, box.height), pygame.SRCALPHA)
    banner_surface.fill((*BUTTON_START_COLOR, 235))
    screen.blit(banner_surface, box.topleft)
    pygame.draw.rect(screen, (255, 255, 255), box, 2, border_radius=6)
    screen.blit(surf1, (box.x + (box.width - surf1.get_width()) // 2, box.y + 10))
    screen.blit(surf2, (box.x + (box.width - surf2.get_width()) // 2, box.y + 10 + surf1.get_height() + 6))

# ============================================================
#              SHARED DRAWING (both states)
# ============================================================

def ease_in_out(t):
    return 0.5 - 0.5 * math.cos(t * math.pi)

def lerp(a, b, t):
    return a + (b - a) * t

def draw_grid():
    for x in range(0, GRID_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, GRID_HEIGHT))
    for y in range(0, GRID_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (GRID_WIDTH, y))

def draw_obstacles():
    for o in obstacles:
        is_full = (o["z_min"] == 0 and o["z_max"] == GRID_LAYERS - 1)
        color = OBSTACLE_FULL_COLOR if is_full else OBSTACLE_PARTIAL_COLOR
        for col in obstacle_cols(o):
            for row in obstacle_rows(o):
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)

def draw_hazards():
    for h in hazards:
        surf = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        surf.fill(h["color"])
        for col in hazard_cols(h):
            for row in hazard_rows(h):
                screen.blit(surf, (col * CELL_SIZE, row * CELL_SIZE))

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
    if mission_complete:
        msg = "Mission complete -- simulation paused"
        surf = font.render(msg, True, SAFE_COLOR)
    elif hazard_ahead:
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

# ============================================================
#              SETUP-SCREEN STATE + LOGIC
# ============================================================

sim_state = "SETUP"
setup_mode = "idle"          # 'idle','placing_start','placing_goal','corner1','corner2'
pending_placement = None     # 'ash','turbulence','obstacle_full','obstacle_partial'
pending_corner1 = None
selected_hazard_id = hazards[0]["id"]
selected_obstacle_id = obstacles[0]["id"]
setup_message = ""
active_buttons = []

def handle_grid_click(pos):
    global setup_mode, pending_corner1, pending_placement
    global start_cell, goal_cell, hazards, next_hazard_id, selected_hazard_id
    global obstacles, next_obstacle_id, selected_obstacle_id

    col = pos[0] // CELL_SIZE
    row = pos[1] // CELL_SIZE
    if not (0 <= col < GRID_COLS and 0 <= row < GRID_ROWS):
        return

    if setup_mode == "placing_start":
        start_cell = (col, row, start_cell[2])
        setup_mode = "idle"
    elif setup_mode == "placing_goal":
        goal_cell = (col, row, goal_cell[2])
        setup_mode = "idle"
    elif setup_mode == "corner1":
        pending_corner1 = (col, row)
        setup_mode = "corner2"
    elif setup_mode == "corner2":
        c1, c2 = pending_corner1, (col, row)
        col_start, col_end = min(c1[0], c2[0]), max(c1[0], c2[0])
        row_start, row_end = min(c1[1], c2[1]), max(c1[1], c2[1])
        width, height = col_end - col_start + 1, row_end - row_start + 1

        if pending_placement in ("ash", "turbulence"):
            new_h = make_hazard(next_hazard_id, pending_placement, col_start, width, row_start, height)
            hazards.append(new_h)
            selected_hazard_id = new_h["id"]
            next_hazard_id += 1
        elif pending_placement in ("obstacle_full", "obstacle_partial"):
            if pending_placement == "obstacle_full":
                z_min, z_max = 0, GRID_LAYERS - 1
            else:
                z_min, z_max = 0, 0  # partial defaults to ground level; edit Min/Max Alt after placing
            new_o = make_obstacle(next_obstacle_id, col_start, width, row_start, height, z_min, z_max)
            obstacles.append(new_o)
            selected_obstacle_id = new_o["id"]
            next_obstacle_id += 1

        setup_mode = "idle"
        pending_placement = None
        pending_corner1 = None

def handle_action(action, target=None):
    global setup_mode, pending_placement, selected_hazard_id, hazards
    global selected_obstacle_id, obstacles
    global start_cell, goal_cell, sim_state, path, setup_message
    global replan_count, last_replan_ms, hazard_cells_ever_crossed, last_counted_segment
    global current_segment, progress, hazard_was_detected, replan_flash_timer
    global mission_start_ticks, mission_end_ticks, mission_complete

    if action == "place_start":
        setup_mode = "placing_start"
    elif action == "place_goal":
        setup_mode = "placing_goal"
    elif action == "cycle_start_alt":
        start_cell = (start_cell[0], start_cell[1], (start_cell[2] + 1) % GRID_LAYERS)
    elif action == "cycle_goal_alt":
        goal_cell = (goal_cell[0], goal_cell[1], (goal_cell[2] + 1) % GRID_LAYERS)
    elif action == "add_ash":
        pending_placement = "ash"
        setup_mode = "corner1"
    elif action == "add_turbulence":
        pending_placement = "turbulence"
        setup_mode = "corner1"
    elif action == "add_obstacle_full":
        pending_placement = "obstacle_full"
        setup_mode = "corner1"
    elif action == "add_obstacle_partial":
        pending_placement = "obstacle_partial"
        setup_mode = "corner1"
    elif action == "select_hazard":
        selected_hazard_id = target
    elif action == "remove_hazard":
        hazards = [h for h in hazards if h["id"] != target]
        if selected_hazard_id == target:
            selected_hazard_id = hazards[0]["id"] if hazards else None
    elif action == "select_obstacle":
        selected_obstacle_id = target
    elif action == "remove_obstacle":
        obstacles = [o for o in obstacles if o["id"] != target]
        if selected_obstacle_id == target:
            selected_obstacle_id = obstacles[0]["id"] if obstacles else None
    elif action == "cycle_minz":
        h = get_selected_hazard()
        if h:
            h["z_min"] = (h["z_min"] + 1) % GRID_LAYERS
            if h["z_min"] > h["z_max"]:
                h["z_max"] = h["z_min"]
    elif action == "cycle_maxz":
        h = get_selected_hazard()
        if h:
            h["z_max"] = (h["z_max"] + 1) % GRID_LAYERS
            if h["z_max"] < h["z_min"]:
                h["z_min"] = h["z_max"]
    elif action == "toggle_drift":
        h = get_selected_hazard()
        if h:
            h["drift_enabled"] = not h["drift_enabled"]
    elif action == "toggle_dir":
        h = get_selected_hazard()
        if h:
            h["drift_dir"] *= -1
    elif action == "cycle_speed":
        h = get_selected_hazard()
        if h:
            idx = SPEED_PRESETS.index(h["drift_interval"]) if h["drift_interval"] in SPEED_PRESETS else 1
            h["drift_interval"] = SPEED_PRESETS[(idx + 1) % len(SPEED_PRESETS)]
    elif action == "cycle_obstacle_minz":
        o = get_selected_obstacle()
        if o:
            o["z_min"] = (o["z_min"] + 1) % GRID_LAYERS
            if o["z_min"] > o["z_max"]:
                o["z_max"] = o["z_min"]
    elif action == "cycle_obstacle_maxz":
        o = get_selected_obstacle()
        if o:
            o["z_max"] = (o["z_max"] + 1) % GRID_LAYERS
            if o["z_max"] < o["z_min"]:
                o["z_min"] = o["z_max"]
    elif action == "start_simulation":
        if start_cell[:2] == goal_cell[:2] and start_cell[2] == goal_cell[2]:
            setup_message = "Start and goal cannot be the same cell."
            return
        if in_any_obstacle(start_cell) or in_any_obstacle(goal_cell):
            setup_message = "Start or goal sits inside a solid obstacle -- move it."
            return
        result = a_star(start_cell, goal_cell)
        if result is None:
            setup_message = "No valid path found with current hazards/obstacles -- adjust and try again."
            return
        path = result
        recompute_derived_data()
        for h in hazards:
            h["tick_counter"] = 0
        replan_count = 0
        last_replan_ms = 0.0
        hazard_cells_ever_crossed = 0
        last_counted_segment = -1
        current_segment = 0
        progress = 0.0
        hazard_was_detected = False
        replan_flash_timer = 0
        mission_start_ticks = pygame.time.get_ticks()
        mission_end_ticks = mission_start_ticks
        mission_complete = False
        setup_message = ""
        sim_state = "RUNNING"

def layout_setup_ui():
    buttons = []
    x = GRID_WIDTH + 16
    y = 14

    screen.blit(font_title.render("SETUP", True, TEXT_MAIN), (x, y))
    y += 30
    for line in ["Place Start/Goal, add zones,", "then Start Simulation.",
                 f"Altitude: 0 - {MAX_ALTITUDE_FT:,} ft ({GRID_LAYERS} levels)"]:
        screen.blit(font_tiny.render(line, True, (150, 150, 165)), (x, y))
        y += 14
    y += 10

    def add_button(label, action, w=280, h=28, target=None, highlight=False, color_override=None):
        nonlocal y
        rect = pygame.Rect(x, y, w, h)
        color = color_override if color_override else (BUTTON_ACTIVE_COLOR if highlight else BUTTON_COLOR)
        pygame.draw.rect(screen, color, rect, border_radius=4)
        pygame.draw.rect(screen, PANEL_AXIS_COLOR, rect, 1, border_radius=4)
        text = font_small.render(label, True, TEXT_MAIN)
        screen.blit(text, (rect.x + 8, rect.y + (rect.height - text.get_height()) // 2))
        buttons.append({"rect": rect, "action": action, "target": target})
        return rect

    add_button("Place Start", "place_start", highlight=(setup_mode == "placing_start"))
    y += 33
    coord = font_tiny.render(f"  ({start_cell[0]}, {start_cell[1]})", True, (170, 170, 185))
    screen.blit(coord, (x, y - 31))

    add_button("Place Goal", "place_goal", highlight=(setup_mode == "placing_goal"))
    y += 33
    coord = font_tiny.render(f"  ({goal_cell[0]}, {goal_cell[1]})", True, (170, 170, 185))
    screen.blit(coord, (x, y - 31))

    add_button(f"Start Alt: {alt_ft(start_cell[2]):,} ft", "cycle_start_alt")
    y += 31
    add_button(f"Goal Alt: {alt_ft(goal_cell[2]):,} ft", "cycle_goal_alt")
    y += 36

    add_button("+ Add Ash Zone", "add_ash", highlight=(pending_placement == "ash"))
    y += 33
    add_button("+ Add Turbulence Zone", "add_turbulence", highlight=(pending_placement == "turbulence"))
    y += 33
    add_button("+ Add Obstacle (Full Height)", "add_obstacle_full",
               highlight=(pending_placement == "obstacle_full"), color_override=BUTTON_OBSTACLE_COLOR if pending_placement != "obstacle_full" else None)
    y += 33
    add_button("+ Add Obstacle (Partial Alt.)", "add_obstacle_partial",
               highlight=(pending_placement == "obstacle_partial"), color_override=BUTTON_OBSTACLE_COLOR if pending_placement != "obstacle_partial" else None)
    y += 32

    mode_text = {
        "placing_start": "Click a grid cell to set START.",
        "placing_goal": "Click a grid cell to set GOAL.",
        "corner1": "Click the FIRST corner of the zone.",
        "corner2": "Click the SECOND corner of the zone.",
    }.get(setup_mode, "")
    if mode_text:
        msg = font_tiny.render(mode_text, True, PANEL_MARKER_COLOR)
        screen.blit(msg, (x, y))
    y += 20

    screen.blit(font_small.render("Hazards:", True, TEXT_MAIN), (x, y))
    y += 22
    for h in hazards:
        row_rect = pygame.Rect(x, y, 246, 23)
        color = BUTTON_ACTIVE_COLOR if h["id"] == selected_hazard_id else BUTTON_COLOR
        pygame.draw.rect(screen, color, row_rect, border_radius=4)
        text = font_tiny.render(f'#{h["id"]} {h["label"]}', True, TEXT_MAIN)
        screen.blit(text, (row_rect.x + 6, row_rect.y + 4))
        buttons.append({"rect": row_rect, "action": "select_hazard", "target": h["id"]})

        remove_rect = pygame.Rect(x + 250, y, 28, 23)
        pygame.draw.rect(screen, (100, 55, 55), remove_rect, border_radius=4)
        screen.blit(font_tiny.render("X", True, TEXT_MAIN), (remove_rect.x + 9, remove_rect.y + 4))
        buttons.append({"rect": remove_rect, "action": "remove_hazard", "target": h["id"]})
        y += 26

    y += 6
    selected_h = get_selected_hazard()
    if selected_h:
        screen.blit(font_tiny.render(f'Editing hazard #{selected_h["id"]} ({selected_h["label"]}):', True, (170, 170, 185)), (x, y))
        y += 18
        add_button(f'Min Alt: {alt_ft(selected_h["z_min"]):,} ft', "cycle_minz")
        y += 30
        add_button(f'Max Alt: {alt_ft(selected_h["z_max"]):,} ft', "cycle_maxz")
        y += 30
        drift_label = "Drift: ON" if selected_h["drift_enabled"] else "Drift: OFF"
        add_button(drift_label, "toggle_drift")
        y += 30
        dir_label = "Direction: ->" if selected_h["drift_dir"] == 1 else "Direction: <-"
        add_button(dir_label, "toggle_dir")
        y += 30
        add_button(f'Speed: {selected_h["drift_interval"]}', "cycle_speed")
        y += 34

    screen.blit(font_small.render("Obstacles:", True, TEXT_MAIN), (x, y))
    y += 22
    for o in obstacles:
        is_full = (o["z_min"] == 0 and o["z_max"] == GRID_LAYERS - 1)
        row_rect = pygame.Rect(x, y, 246, 23)
        color = BUTTON_ACTIVE_COLOR if o["id"] == selected_obstacle_id else BUTTON_COLOR
        pygame.draw.rect(screen, color, row_rect, border_radius=4)
        kind = "Full" if is_full else f'{alt_ft(o["z_min"]):,}-{alt_ft(o["z_max"]):,}ft'
        text = font_tiny.render(f'#{o["id"]} Obstacle ({kind})', True, TEXT_MAIN)
        screen.blit(text, (row_rect.x + 6, row_rect.y + 4))
        buttons.append({"rect": row_rect, "action": "select_obstacle", "target": o["id"]})

        remove_rect = pygame.Rect(x + 250, y, 28, 23)
        pygame.draw.rect(screen, (100, 55, 55), remove_rect, border_radius=4)
        screen.blit(font_tiny.render("X", True, TEXT_MAIN), (remove_rect.x + 9, remove_rect.y + 4))
        buttons.append({"rect": remove_rect, "action": "remove_obstacle", "target": o["id"]})
        y += 26

    y += 6
    selected_o = get_selected_obstacle()
    if selected_o:
        screen.blit(font_tiny.render(f'Editing obstacle #{selected_o["id"]}:', True, (170, 170, 185)), (x, y))
        y += 18
        add_button(f'Min Alt: {alt_ft(selected_o["z_min"]):,} ft', "cycle_obstacle_minz")
        y += 30
        add_button(f'Max Alt: {alt_ft(selected_o["z_max"]):,} ft', "cycle_obstacle_maxz")
        y += 34

    return buttons

def draw_setup_bottom_bar():
    pygame.draw.rect(screen, DASHBOARD_BG_COLOR, dashboard_rect)
    pygame.draw.line(screen, PANEL_AXIS_COLOR, (0, GRID_HEIGHT), (WIDTH, GRID_HEIGHT), 2)

    rect = pygame.Rect(20, GRID_HEIGHT + 20, 240, 44)
    pygame.draw.rect(screen, BUTTON_START_COLOR, rect, border_radius=6)
    text = font.render("Start Simulation", True, (255, 255, 255))
    screen.blit(text, (rect.x + 20, rect.y + 11))

    if setup_message:
        msg = font_small.render(setup_message, True, WARNING_COLOR)
        screen.blit(msg, (280, GRID_HEIGHT + 33))

    return {"rect": rect, "action": "start_simulation", "target": None}

def draw_corner_preview():
    if setup_mode == "corner2" and pending_corner1 is not None:
        mx, my = pygame.mouse.get_pos()
        if mx < GRID_WIDTH and my < GRID_HEIGHT:
            col2, row2 = mx // CELL_SIZE, my // CELL_SIZE
            c1 = pending_corner1
            x1 = min(c1[0], col2) * CELL_SIZE
            y1 = min(c1[1], row2) * CELL_SIZE
            x2 = (max(c1[0], col2) + 1) * CELL_SIZE
            y2 = (max(c1[1], row2) + 1) * CELL_SIZE
            pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(x1, y1, x2 - x1, y2 - y1), 2)

# --- Drone movement state ---
current_segment = 0
progress = 0.0
speed = 0.02
hazard_ahead, hazard_cells = False, []

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and sim_state == "SETUP":
            pos = event.pos
            clicked = next((b for b in active_buttons if b["rect"].collidepoint(pos)), None)
            if clicked:
                handle_action(clicked["action"], clicked.get("target"))
            elif pos[0] < GRID_WIDTH and pos[1] < GRID_HEIGHT:
                handle_grid_click(pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and sim_state == "RUNNING":
                sim_state = "SETUP"

    if sim_state == "RUNNING":
        if not mission_complete:
            for h in hazards:
                update_hazard_drift(h)

            if current_segment < len(path) - 1:
                progress += speed
                if progress >= 1.0:
                    progress = 0.0
                    current_segment += 1
                    if current_segment != last_counted_segment:
                        if in_any_hazard(path[current_segment]):
                            hazard_cells_ever_crossed += 1
                        last_counted_segment = current_segment

        is_climbing = False
        if current_segment < len(path) - 1:
            start_point = pixel_path[current_segment]
            end_point = pixel_path[current_segment + 1]
            eased_progress = ease_in_out(progress)
            drone_x = lerp(start_point[0], end_point[0], eased_progress)
            drone_y = lerp(start_point[1], end_point[1], eased_progress)
            current_z = path[current_segment][2]
            is_climbing = path[current_segment][2] != path[current_segment + 1][2]
        else:
            drone_x, drone_y = pixel_path[-1]
            current_z = path[-1][2]
            if not mission_complete:
                mission_complete = True
                mission_end_ticks = pygame.time.get_ticks()
                print("[EVENT] Mission complete -- drone reached the goal. Simulation paused.")

        if not mission_complete:
            hazard_ahead, hazard_cells = detect_hazard_ahead(path, current_segment)
            if hazard_ahead and not hazard_was_detected:
                print(f"[EVENT] Hazard newly detected ahead! {len(hazard_cells)} cell(s): {hazard_cells}")
                trigger_replan(current_segment)
                replan_flash_timer = 30
                hazard_ahead, hazard_cells = detect_hazard_ahead(path, current_segment)
                if not hazard_ahead:
                    print("[EVENT] Hazard resolved immediately by replan.")
            elif not hazard_ahead and hazard_was_detected:
                print("[EVENT] Hazard ahead has cleared.")
            hazard_was_detected = hazard_ahead
            if replan_flash_timer > 0:
                replan_flash_timer -= 1
        else:
            hazard_ahead, hazard_cells = False, []

    # --- Drawing ---
    screen.fill(BG_COLOR)
    draw_grid()
    draw_hazards()
    draw_obstacles()
    if sim_state == "RUNNING":
        draw_path(hazard_ahead)
    draw_wind_indicator()

    pygame.draw.circle(screen, START_COLOR, grid_to_pixel(start_cell), 10)
    pygame.draw.circle(screen, GOAL_COLOR, grid_to_pixel(goal_cell), 10)

    if sim_state == "RUNNING":
        drone_color = DRONE_CLIMB_COLOR if is_climbing else DRONE_COLOR
        glow_surface = pygame.Surface((28, 28), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*drone_color, 60), (14, 14), 14)
        screen.blit(glow_surface, (int(drone_x) - 14, int(drone_y) - 14))
        pygame.draw.circle(screen, drone_color, (int(drone_x), int(drone_y)), 8)

        title_label = font_title.render("Adaptive Flight-Path Replanner", True, TEXT_MAIN)
        screen.blit(title_label, (12, 10))

        alt_label = font_small.render(
            f"Altitude: {alt_ft(current_z):,} ft" + ("  (climbing/descending)" if is_climbing else ""),
            True, (190, 190, 205)
        )
        screen.blit(alt_label, (12, GRID_HEIGHT - 26))

        draw_hazard_status(hazard_ahead, hazard_cells)
        draw_replan_stats(replan_flash_timer > 0)
        draw_legend()
        if mission_complete:
            draw_mission_complete_banner()

        draw_altitude_panel()
        draw_altitude_marker(current_segment, progress, current_z)

        draw_dashboard(hazard_ahead)
    else:
        pygame.draw.rect(screen, PANEL_BG_COLOR, (GRID_WIDTH, 0, PANEL_WIDTH, GRID_HEIGHT))
        pygame.draw.line(screen, PANEL_AXIS_COLOR, (GRID_WIDTH, 0), (GRID_WIDTH, GRID_HEIGHT), 2)
        draw_corner_preview()
        buttons = layout_setup_ui()
        bottom_button = draw_setup_bottom_bar()
        buttons.append(bottom_button)
        active_buttons = buttons

    pygame.display.flip()
    clock.tick(60)

pygame.quit()