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

# ============================================================
#      DISPLAY / CANVAS SYSTEM (supports fullscreen + resize)
# ============================================================
# Everything in this program draws onto a fixed-size logical
# "canvas" surface (WIDTH x HEIGHT). Each frame that canvas is
# scaled (preserving aspect ratio, letterboxed) onto the actual
# OS window, which can be resized or made fullscreen freely.
# `screen` is kept as the name every draw call already uses, so
# none of the drawing code below needs to change -- it always
# draws onto the canvas.

CANVAS_WIDTH, CANVAS_HEIGHT = WIDTH, HEIGHT
canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
screen = canvas

display_surface = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Adaptive Drone Flight-Path Replanner")

is_fullscreen = False
_present_scale = 1.0
_present_offset = (0, 0)

def set_display_mode(fullscreen):
    global display_surface, is_fullscreen
    is_fullscreen = fullscreen
    if fullscreen:
        display_surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        display_surface = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

def present():
    """Scale the logical canvas onto the real window and flip."""
    global _present_scale, _present_offset
    dw, dh = display_surface.get_size()
    scale = max(0.01, min(dw / CANVAS_WIDTH, dh / CANVAS_HEIGHT))
    sw, sh = max(1, int(CANVAS_WIDTH * scale)), max(1, int(CANVAS_HEIGHT * scale))
    ox, oy = (dw - sw) // 2, (dh - sh) // 2
    _present_scale, _present_offset = scale, (ox, oy)
    display_surface.fill((0, 0, 0))
    if (sw, sh) == (CANVAS_WIDTH, CANVAS_HEIGHT):
        display_surface.blit(canvas, (ox, oy))
    else:
        display_surface.blit(pygame.transform.smoothscale(canvas, (sw, sh)), (ox, oy))
    pygame.display.flip()

def to_canvas_pos(pos):
    """Convert a real-window mouse position into logical canvas coordinates."""
    x, y = pos
    scale = _present_scale if _present_scale else 1.0
    ox, oy = _present_offset
    return ((x - ox) / scale, (y - oy) / scale)

def mouse_canvas_pos():
    return to_canvas_pos(pygame.mouse.get_pos())

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

def profile_to_pixel(distance, altitude_ft, total_dist=None):
    if total_dist is None:
        total_dist = total_distance
    tx = 0 if total_dist == 0 else distance / total_dist
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

    hint = "Press R: setup   F11: fullscreen"
    press_r = font_tiny.render(hint, True, (120, 120, 135))
    screen.blit(press_r, (WIDTH - 210, HEIGHT - 18))

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

    col = int(pos[0] // CELL_SIZE)
    row = int(pos[1] // CELL_SIZE)
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
        mx, my = mouse_canvas_pos()
        if mx < GRID_WIDTH and my < GRID_HEIGHT:
            col2, row2 = int(mx // CELL_SIZE), int(my // CELL_SIZE)
            c1 = pending_corner1
            x1 = min(c1[0], col2) * CELL_SIZE
            y1 = min(c1[1], row2) * CELL_SIZE
            x2 = (max(c1[0], col2) + 1) * CELL_SIZE
            y2 = (max(c1[1], row2) + 1) * CELL_SIZE
            pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(x1, y1, x2 - x1, y2 - y1), 2)


# ============================================================
#                  REAL-WORLD MAP MODE
# ============================================================
# Map mode uses OpenStreetMap raster tiles as the geographic background.
# The planner itself uses a local latitude/longitude planning grid over
# the visible map. Obstacles and hazards placed here are stored as
# geographic (lat/lon) boxes, so they stay correctly positioned as the
# map is panned and zoomed.

try:
    import requests
    from PIL import Image
except ImportError:
    requests = None
    Image = None

MAP_WIDTH, MAP_HEIGHT = GRID_WIDTH, GRID_HEIGHT
MAP_ZOOM_MIN, MAP_ZOOM_MAX = 3, 18
map_zoom_level = 13          # requested zoom -- changed by scroll wheel / +- keys / buttons
map_zoom_actual = map_zoom_level   # zoom level of the currently loaded tiles
MAP_CENTER_LAT = 12.9716       # Bengaluru default; change these two values for another area
MAP_CENTER_LON = 77.5946
MAP_TILE_SIZE = 256
MAP_COLS = 80
MAP_ROWS = 60
MAP_CACHE_DIR = "osm_tile_cache"
MAP_USER_AGENT = "AdaptiveDroneFlightPathReplanner/3.0 (educational project)"

map_mode = False
map_setup_mode = "idle"       # idle / placing_start / placing_goal / corner1 / corner2
map_pending_placement = None  # 'ash','turbulence','obstacle_full','obstacle_partial'
map_pending_corner1_latlon = None
map_start_latlon = (MAP_CENTER_LAT, MAP_CENTER_LON)
map_goal_latlon = (12.9352, 77.6245)
map_start_z = 0
map_goal_z = 0
map_path = []
map_pixel_path = []
map_current_segment = 0
map_progress = 0.0
map_mission_complete = False
map_origin_world = (0, 0)
map_surface = None
map_status = "Map mode: load the map, then place START and GOAL."
map_replan_count = 0
map_last_replan_ms = 0.0
map_hazard_ahead = False
map_hazard_cells = []
map_hazard_was_detected = False
map_replan_flash_timer = 0
map_hazard_cells_ever_crossed = 0
map_last_counted_segment = -1
map_mission_start_ticks = 0
map_mission_end_ticks = 0

# Altitude-profile data (mirrors the grid mode profile, driven by map_path)
map_cumulative_distances = []
map_total_distance = 1
map_profile_points = []

# Pan (click-drag) state -- live-previewed, tiles refetched on release.
map_is_panning = False
map_pan_last_pos = None
map_pan_live_offset = (0, 0)
MAP_PAN_CLICK_THRESHOLD = 4

# Geo-referenced obstacles/hazards for map mode
map_obstacles = []
next_map_obstacle_id = 1
map_hazards = []
next_map_hazard_id = 1
selected_map_hazard_id = None
selected_map_obstacle_id = None


def latlon_to_world(lat, lon, zoom):
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n * MAP_TILE_SIZE
    lat_rad = math.radians(max(-85.05112878, min(85.05112878, lat)))
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * MAP_TILE_SIZE
    return x, y


def world_to_latlon(x, y, zoom):
    n = 2 ** zoom
    lon = x / (n * MAP_TILE_SIZE) * 360.0 - 180.0
    merc = math.pi * (1.0 - 2.0 * y / (n * MAP_TILE_SIZE))
    lat = math.degrees(math.atan(math.sinh(merc)))
    return lat, lon


def tile_xy_for_world(x, y):
    return int(x // MAP_TILE_SIZE), int(y // MAP_TILE_SIZE)


def download_osm_tile(tx, ty, zoom):
    if requests is None or Image is None:
        raise RuntimeError("Map mode requires the 'requests' and 'Pillow' packages.")

    import os
    os.makedirs(MAP_CACHE_DIR, exist_ok=True)
    path = os.path.join(MAP_CACHE_DIR, f"{zoom}_{tx}_{ty}.png")
    if os.path.exists(path):
        return path

    max_tile = 2 ** zoom
    if not (0 <= tx < max_tile and 0 <= ty < max_tile):
        return None

    url = f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png"
    response = requests.get(url, headers={"User-Agent": MAP_USER_AGENT}, timeout=10)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)
    return path


def load_real_map():
    """(Re)download tiles for the current MAP_CENTER_LAT/LON at map_zoom_level."""
    global map_surface, map_origin_world, map_zoom_actual, map_status

    if requests is None or Image is None:
        map_status = "Install: pip install requests pillow"
        return False

    try:
        zoom = map_zoom_level
        cx, cy = latlon_to_world(MAP_CENTER_LAT, MAP_CENTER_LON, zoom)
        left_world = cx - MAP_WIDTH / 2
        top_world = cy - MAP_HEIGHT / 2
        tx0, ty0 = tile_xy_for_world(left_world, top_world)
        tx1, ty1 = tile_xy_for_world(left_world + MAP_WIDTH, top_world + MAP_HEIGHT)

        mosaic_w = (tx1 - tx0 + 1) * MAP_TILE_SIZE
        mosaic_h = (ty1 - ty0 + 1) * MAP_TILE_SIZE
        mosaic = Image.new("RGB", (mosaic_w, mosaic_h))

        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile_path = download_osm_tile(tx, ty, zoom)
                if tile_path:
                    tile = Image.open(tile_path).convert("RGB")
                    mosaic.paste(tile, ((tx - tx0) * MAP_TILE_SIZE,
                                        (ty - ty0) * MAP_TILE_SIZE))

        crop_x = int(left_world - tx0 * MAP_TILE_SIZE)
        crop_y = int(top_world - ty0 * MAP_TILE_SIZE)
        cropped = mosaic.crop((crop_x, crop_y,
                               crop_x + MAP_WIDTH, crop_y + MAP_HEIGHT))
        map_surface = pygame.image.fromstring(cropped.tobytes(), cropped.size, "RGB")
        map_origin_world = (left_world, top_world)
        map_zoom_actual = zoom
        map_status = f"Map loaded (zoom {zoom}). Select START and GOAL."
        return True
    except Exception as exc:
        map_status = f"Map load failed: {str(exc)[:70]}"
        return False


def screen_to_map_latlon(pos):
    if map_surface is None:
        return None
    ox, oy = map_pan_live_offset if map_is_panning else (0, 0)
    wx = map_origin_world[0] + (pos[0] - ox)
    wy = map_origin_world[1] + (pos[1] - oy)
    return world_to_latlon(wx, wy, map_zoom_actual)


def map_latlon_to_screen(lat, lon):
    wx, wy = latlon_to_world(lat, lon, map_zoom_actual)
    ox, oy = map_pan_live_offset if map_is_panning else (0, 0)
    return (int(wx - map_origin_world[0] + ox), int(wy - map_origin_world[1] + oy))


def map_coord_from_latlon(latlon, z):
    lat, lon = latlon
    x0, y0 = map_origin_world
    wx, wy = latlon_to_world(lat, lon, map_zoom_actual)
    px = wx - x0
    py = wy - y0
    gx = max(0, min(MAP_COLS - 1, int(px / MAP_WIDTH * MAP_COLS)))
    gy = max(0, min(MAP_ROWS - 1, int(py / MAP_HEIGHT * MAP_ROWS)))
    return gx, gy, z


def map_cell_to_latlon(cell):
    gx, gy, z = cell
    px = (gx + 0.5) / MAP_COLS * MAP_WIDTH
    py = (gy + 0.5) / MAP_ROWS * MAP_HEIGHT
    return screen_to_map_latlon((px, py))


# --- Geo-referenced obstacles & hazards ------------------------------------

def make_map_obstacle(obstacle_id, lat1, lon1, lat2, lon2, z_min, z_max):
    return {
        "id": obstacle_id,
        "lat1": lat1, "lon1": lon1, "lat2": lat2, "lon2": lon2,
        "z_min": z_min, "z_max": z_max,
    }

def make_map_hazard(hazard_id, htype, lat1, lon1, lat2, lon2):
    if htype == "ash":
        cost, color, band_color, label = 12, ASH_COLOR, PANEL_ASH_BAND_COLOR, "Ash"
    else:
        cost, color, band_color, label = 10, TURBULENCE_COLOR, PANEL_TURB_BAND_COLOR, "Turbulence"
    return {
        "id": hazard_id, "type": htype, "label": label, "cost": cost,
        "color": color, "band_color": band_color,
        "lat1": lat1, "lon1": lon1, "lat2": lat2, "lon2": lon2,
        "z_min": 0, "z_max": 0,
        "drift_enabled": False, "drift_dir": 1,
        "drift_interval": 45, "tick_counter": 0,
    }

def map_zone_bounds(z):
    lat_min, lat_max = sorted((z["lat1"], z["lat2"]))
    lon_min, lon_max = sorted((z["lon1"], z["lon2"]))
    return lat_min, lat_max, lon_min, lon_max

def map_zone_grid_bounds(z):
    lat_min, lat_max, lon_min, lon_max = map_zone_bounds(z)
    gx1, gy1, _ = map_coord_from_latlon((lat_min, lon_min), 0)
    gx2, gy2, _ = map_coord_from_latlon((lat_max, lon_max), 0)
    col_min, col_max = sorted((gx1, gx2))
    row_min, row_max = sorted((gy1, gy2))
    return col_min, col_max, row_min, row_max

def in_map_obstacle(cell, o):
    col, row, z = cell
    col_min, col_max, row_min, row_max = map_zone_grid_bounds(o)
    return col_min <= col <= col_max and row_min <= row <= row_max and o["z_min"] <= z <= o["z_max"]

def in_any_map_obstacle(cell):
    return any(in_map_obstacle(cell, o) for o in map_obstacles)

def in_map_hazard(cell, h):
    col, row, z = cell
    col_min, col_max, row_min, row_max = map_zone_grid_bounds(h)
    return col_min <= col <= col_max and row_min <= row <= row_max and h["z_min"] <= z <= h["z_max"]

def in_any_map_hazard(cell):
    return any(in_map_hazard(cell, h) for h in map_hazards)

def get_selected_map_hazard():
    return next((h for h in map_hazards if h["id"] == selected_map_hazard_id), None)

def get_selected_map_obstacle():
    return next((o for o in map_obstacles if o["id"] == selected_map_obstacle_id), None)

def update_map_hazard_drift(h):
    if not h["drift_enabled"] or map_surface is None:
        return
    h["tick_counter"] += 1
    if h["tick_counter"] < h["drift_interval"]:
        return
    h["tick_counter"] = 0

    cell_world_span = MAP_WIDTH / MAP_COLS
    wx, wy = latlon_to_world(h["lat1"], h["lon1"], map_zoom_actual)
    new_wx = wx + h["drift_dir"] * cell_world_span
    _, new_lon = world_to_latlon(new_wx, wy, map_zoom_actual)
    delta_lon = new_lon - h["lon1"]
    h["lon1"] += delta_lon
    h["lon2"] += delta_lon

    lon_min, lon_max = sorted((h["lon1"], h["lon2"]))
    view_lat_tl, view_lon_tl = world_to_latlon(map_origin_world[0], map_origin_world[1], map_zoom_actual)
    view_lat_br, view_lon_br = world_to_latlon(map_origin_world[0] + MAP_WIDTH, map_origin_world[1] + MAP_HEIGHT, map_zoom_actual)
    min_bound, max_bound = sorted((view_lon_tl, view_lon_br))
    if lon_max >= max_bound:
        shift = max_bound - lon_max
        h["lon1"] += shift
        h["lon2"] += shift
        h["drift_dir"] = -1
    elif lon_min <= min_bound:
        shift = min_bound - lon_min
        h["lon1"] += shift
        h["lon2"] += shift
        h["drift_dir"] = 1


def map_neighbors(cell):
    x, y, z = cell
    candidates = [(x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)]
    return [c for c in candidates
            if 0 <= c[0] < MAP_COLS and 0 <= c[1] < MAP_ROWS and 0 <= c[2] < GRID_LAYERS
            and not in_any_map_obstacle(c)]


def map_move_cost(a, b):
    base = VERTICAL_COST if a[2] != b[2] else 1
    hazard_total = sum(h["cost"] for h in map_hazards if in_map_hazard(b, h))
    dx, dy = b[0] - a[0], b[1] - a[1]
    wind = WIND_PENALTY if (a[2] == b[2] and dx * WIND_DIRECTION[0] + dy * WIND_DIRECTION[1] < 0) else 0
    return base + hazard_total + wind


def map_astar(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in map_neighbors(current):
            tentative = g_score[current] + map_move_cost(current, neighbor)
            if tentative < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative
                f = tentative + abs(neighbor[0]-goal[0]) + abs(neighbor[1]-goal[1]) + VERTICAL_COST * abs(neighbor[2]-goal[2])
                heapq.heappush(open_set, (f, neighbor))
                came_from[neighbor] = current
    return None


def map_compute_cumulative_distances(p):
    distances = [0]
    for i in range(1, len(p)):
        distances.append(distances[-1] + map_move_cost(p[i - 1], p[i]))
    return distances


def map_recompute_profile():
    global map_cumulative_distances, map_total_distance, map_profile_points
    map_cumulative_distances = map_compute_cumulative_distances(map_path)
    map_total_distance = map_cumulative_distances[-1] if map_cumulative_distances else 1
    map_profile_points = [
        profile_to_pixel(map_cumulative_distances[i], alt_ft(map_path[i][2]), map_total_distance)
        for i in range(len(map_path))
    ]


def build_map_route():
    global map_path, map_pixel_path, map_current_segment, map_progress, map_mission_complete
    global map_replan_count, map_last_replan_ms, map_hazard_cells_ever_crossed, map_last_counted_segment
    global map_hazard_was_detected, map_replan_flash_timer, map_mission_start_ticks, map_mission_end_ticks
    start = map_coord_from_latlon(map_start_latlon, map_start_z)
    goal = map_coord_from_latlon(map_goal_latlon, map_goal_z)
    if in_any_map_obstacle(start) or in_any_map_obstacle(goal):
        return False, 0.0, "Start or goal sits inside a solid obstacle -- move it."
    start_time = time.perf_counter()
    result = map_astar(start, goal)
    elapsed = (time.perf_counter() - start_time) * 1000
    if result is None:
        return False, elapsed, "No valid geographic route found -- adjust and try again."
    map_path = result
    map_pixel_path = [map_latlon_to_screen(*map_cell_to_latlon(c)) for c in map_path]
    map_recompute_profile()
    map_current_segment = 0
    map_progress = 0.0
    map_mission_complete = False
    map_replan_count = 0
    map_last_replan_ms = elapsed
    map_hazard_cells_ever_crossed = 0
    map_last_counted_segment = -1
    map_hazard_was_detected = False
    map_replan_flash_timer = 0
    for h in map_hazards:
        h["tick_counter"] = 0
    map_mission_start_ticks = pygame.time.get_ticks()
    map_mission_end_ticks = map_mission_start_ticks
    return True, elapsed, "Route calculated."


def map_detect_hazard_ahead(current_segment):
    remaining = map_path[current_segment:]
    cells = [c for c in remaining if in_any_map_hazard(c)]
    return (len(cells) > 0), cells


def map_trigger_replan(current_segment):
    global map_path, map_pixel_path, map_replan_count, map_last_replan_ms, map_status
    if current_segment + 1 >= len(map_path):
        return
    replan_from = map_path[current_segment + 1]
    goal = map_coord_from_latlon(map_goal_latlon, map_goal_z)
    start_t = time.perf_counter()
    new_remaining = map_astar(replan_from, goal)
    elapsed = (time.perf_counter() - start_t) * 1000
    if new_remaining is None:
        map_status = "Replan failed -- keeping existing map route."
        return
    map_path = map_path[:current_segment + 1] + new_remaining
    map_pixel_path = [map_latlon_to_screen(*map_cell_to_latlon(c)) for c in map_path]
    map_recompute_profile()
    map_replan_count += 1
    map_last_replan_ms = elapsed


def map_current_path_total_cost():
    total = 0
    for i in range(1, len(map_path)):
        total += map_move_cost(map_path[i-1], map_path[i])
    return total


def map_current_path_altitude_changes():
    changes = 0
    for i in range(1, len(map_path)):
        if map_path[i][2] != map_path[i-1][2]:
            changes += 1
    return changes


def map_mission_elapsed_seconds():
    end = map_mission_end_ticks if map_mission_complete else pygame.time.get_ticks()
    return (end - map_mission_start_ticks) / 1000.0


def draw_real_map():
    if map_surface is not None:
        screen.fill((20, 24, 30), pygame.Rect(0, 0, MAP_WIDTH, MAP_HEIGHT))
        ox, oy = map_pan_live_offset if map_is_panning else (0, 0)
        screen.blit(map_surface, (ox, oy))
    else:
        screen.fill((35, 45, 55), pygame.Rect(0, 0, MAP_WIDTH, MAP_HEIGHT))
        msg = font.render("Real-world map unavailable", True, TEXT_MAIN)
        screen.blit(msg, (20, 20))

    for h in map_hazards:
        p1 = map_latlon_to_screen(h["lat1"], h["lon1"])
        p2 = map_latlon_to_screen(h["lat2"], h["lon2"])
        x1, x2 = sorted((p1[0], p2[0]))
        y1, y2 = sorted((p1[1], p2[1]))
        rect = pygame.Rect(x1, y1, max(x2 - x1, 2), max(y2 - y1, 2))
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        surf.fill(h["color"])
        screen.blit(surf, rect.topleft)

    for o in map_obstacles:
        is_full = (o["z_min"] == 0 and o["z_max"] == GRID_LAYERS - 1)
        color = OBSTACLE_FULL_COLOR if is_full else OBSTACLE_PARTIAL_COLOR
        p1 = map_latlon_to_screen(o["lat1"], o["lon1"])
        p2 = map_latlon_to_screen(o["lat2"], o["lon2"])
        x1, x2 = sorted((p1[0], p2[0]))
        y1, y2 = sorted((p1[1], p2[1]))
        rect = pygame.Rect(x1, y1, max(x2 - x1, 2), max(y2 - y1, 2))
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        surf.fill((*color, 190))
        screen.blit(surf, rect.topleft)
        pygame.draw.rect(screen, (255, 255, 255), rect, 1)

    if len(map_pixel_path) > 1:
        color = HAZARD_PATH_COLOR if map_hazard_ahead else (50, 105, 235)
        pygame.draw.lines(screen, color, False, map_pixel_path, 4)

    start_screen = map_latlon_to_screen(*map_start_latlon)
    goal_screen = map_latlon_to_screen(*map_goal_latlon)
    pygame.draw.circle(screen, START_COLOR, start_screen, 10)
    pygame.draw.circle(screen, (255, 255, 255), start_screen, 11, 1)
    pygame.draw.circle(screen, GOAL_COLOR, goal_screen, 10)
    pygame.draw.circle(screen, (255, 255, 255), goal_screen, 11, 1)

    if map_mode and map_mission_complete and map_pixel_path:
        pygame.draw.circle(screen, DRONE_COLOR, map_pixel_path[-1], 8)
    elif map_mode and sim_state == "RUNNING" and map_pixel_path and map_current_segment < len(map_pixel_path) - 1:
        a = map_pixel_path[map_current_segment]
        b = map_pixel_path[map_current_segment + 1]
        t = ease_in_out(map_progress)
        dx = a[0] + (b[0] - a[0]) * t
        dy = a[1] + (b[1] - a[1]) * t
        drone_color = DRONE_CLIMB_COLOR if map_path[map_current_segment][2] != map_path[map_current_segment+1][2] else DRONE_COLOR
        pygame.draw.circle(screen, drone_color, (int(dx), int(dy)), 8)

    title = font_title.render("Real-World Map Mode", True, (20, 25, 35))
    screen.blit(title, (12, 10))
    attribution = font_tiny.render("(c) OpenStreetMap contributors", True, (20, 25, 35))
    screen.blit(attribution, (12, MAP_HEIGHT - 22))


def draw_map_corner_preview():
    if map_setup_mode == "corner2" and map_pending_corner1_latlon is not None:
        mx, my = mouse_canvas_pos()
        if mx < MAP_WIDTH and my < MAP_HEIGHT:
            ll = screen_to_map_latlon((mx, my))
            if ll:
                p1 = map_latlon_to_screen(*map_pending_corner1_latlon)
                p2 = map_latlon_to_screen(*ll)
                x1, x2 = sorted((p1[0], p2[0]))
                y1, y2 = sorted((p1[1], p2[1]))
                pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(x1, y1, max(x2-x1,2), max(y2-y1,2)), 2)


def draw_map_hazard_band(alt_min_ft, alt_max_ft, color, label):
    _, y_top = profile_to_pixel(0, alt_max_ft, map_total_distance)
    _, y_bottom = profile_to_pixel(0, alt_min_ft, map_total_distance)
    band_height = max(y_bottom - y_top, 4)
    band_surface = pygame.Surface((panel_rect.width, int(band_height)), pygame.SRCALPHA)
    band_surface.fill(color)
    screen.blit(band_surface, (panel_rect.left, int(y_top)))
    label_surface = font_tiny.render(label, True, (235, 235, 240))
    screen.blit(label_surface, (panel_rect.left + 4, int(y_top) + 2))


def draw_map_altitude_panel():
    pygame.draw.rect(screen, PANEL_BG_COLOR, (GRID_WIDTH, 0, PANEL_WIDTH, GRID_HEIGHT))
    pygame.draw.line(screen, PANEL_AXIS_COLOR, (GRID_WIDTH, 0), (GRID_WIDTH, GRID_HEIGHT), 2)

    title = font_title.render("Altitude Profile", True, TEXT_MAIN)
    screen.blit(title, (GRID_WIDTH + 16, 22))

    for h in map_hazards:
        draw_map_hazard_band(alt_ft(h["z_min"]), alt_ft(h["z_max"]), h["band_color"], f'{h["label"]} #{h["id"]}')

    pygame.draw.line(screen, PANEL_AXIS_COLOR, panel_rect.bottomleft, panel_rect.topleft, 2)
    pygame.draw.line(screen, PANEL_AXIS_COLOR, panel_rect.bottomleft, panel_rect.bottomright, 2)

    for z in range(GRID_LAYERS):
        ft = alt_ft(z)
        _, y = profile_to_pixel(0, ft, map_total_distance)
        label = font_tiny.render(f"{ft:,} ft", True, (150, 150, 165))
        screen.blit(label, (panel_rect.left - label.get_width() - 6, y - 7))
        pygame.draw.line(screen, (38, 39, 50), (panel_rect.left, y), (panel_rect.right, y), 1)

    x_label = font_tiny.render("distance traveled  ->", True, (150, 150, 165))
    screen.blit(x_label, (panel_rect.left, panel_rect.bottom + 8))

    if len(map_profile_points) > 1:
        pygame.draw.lines(screen, PANEL_LINE_COLOR, False, map_profile_points, 3)
    for point in map_profile_points:
        pygame.draw.circle(screen, PANEL_LINE_COLOR, (int(point[0]), int(point[1])), 3)


def draw_map_altitude_marker(segment_index, progress, current_z):
    if not map_cumulative_distances:
        return
    if segment_index >= len(map_cumulative_distances) - 1:
        dist_now = map_cumulative_distances[-1]
    else:
        d_start = map_cumulative_distances[segment_index]
        d_end = map_cumulative_distances[segment_index + 1]
        dist_now = d_start + (d_end - d_start) * progress

    x, y = profile_to_pixel(dist_now, alt_ft(current_z), map_total_distance)
    pygame.draw.circle(screen, PANEL_MARKER_COLOR, (int(x), int(y)), 7)
    pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 7, 1)


def draw_map_legend():
    items = [
        (START_COLOR, "Start"),
        (GOAL_COLOR, "Goal"),
        (DRONE_COLOR, "Drone (level flight)"),
        (DRONE_CLIMB_COLOR, "Drone (climbing/descending)"),
        (OBSTACLE_FULL_COLOR, "Obstacle (all altitudes)"),
        (OBSTACLE_PARTIAL_COLOR, "Obstacle (some altitudes)"),
    ]
    for h in map_hazards:
        items.append((h["color"][:3], f'{h["label"]} #{h["id"]}'))

    padding = 10
    row_height = 20
    box_width = 230
    box_height = padding * 2 + row_height * len(items)
    box = pygame.Rect(MAP_WIDTH - box_width - 12, 12, box_width, box_height)
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


def draw_map_mission_complete_banner():
    line1 = "MISSION COMPLETE  --  Goal Reached"
    line2 = "Press R to return to setup and run again"
    surf1 = font.render(line1, True, (255, 255, 255))
    surf2 = font_tiny.render(line2, True, (235, 235, 235))
    box_w = max(surf1.get_width(), surf2.get_width()) + 40
    box_h = surf1.get_height() + surf2.get_height() + 26
    box = pygame.Rect((MAP_WIDTH - box_w) // 2, 16, box_w, box_h)
    banner_surface = pygame.Surface((box.width, box.height), pygame.SRCALPHA)
    banner_surface.fill((*BUTTON_START_COLOR, 235))
    screen.blit(banner_surface, box.topleft)
    pygame.draw.rect(screen, (255, 255, 255), box, 2, border_radius=6)
    screen.blit(surf1, (box.x + (box.width - surf1.get_width()) // 2, box.y + 10))
    screen.blit(surf2, (box.x + (box.width - surf2.get_width()) // 2, box.y + 10 + surf1.get_height() + 6))


def draw_map_dashboard():
    pygame.draw.rect(screen, DASHBOARD_BG_COLOR, dashboard_rect)
    pygame.draw.line(screen, PANEL_AXIS_COLOR, (0, GRID_HEIGHT), (WIDTH, GRID_HEIGHT), 2)

    total_cost = map_current_path_total_cost()
    alt_changes = map_current_path_altitude_changes()
    energy = estimated_energy(total_cost)
    elapsed = map_mission_elapsed_seconds()

    col1 = [
        ("MISSION TIME", f"{elapsed:.1f} s"),
        ("CURRENT PATH COST", f"{total_cost}"),
        ("ALTITUDE CHANGES", f"{alt_changes}"),
    ]
    col2 = [
        ("HAZARD CELLS CROSSED", f"{map_hazard_cells_ever_crossed}"),
        ("REPLANS TRIGGERED", f"{map_replan_count}"),
        ("LAST REPLAN TIME", f"{map_last_replan_ms:.3f} ms"),
    ]
    col3 = [
        ("EST. ENERGY (illustrative)", f"{energy:.1f} units"),
        ("ROUTE STATUS", "MISSION COMPLETE" if map_mission_complete else ("HAZARD AHEAD" if map_hazard_ahead else "CLEAR")),
    ]

    def draw_column(items, x):
        y = GRID_HEIGHT + 10
        for label, value in items:
            label_surf = font_tiny.render(label, True, DASHBOARD_LABEL_COLOR)
            value_color = DASHBOARD_VALUE_COLOR
            if label == "ROUTE STATUS":
                value_color = SAFE_COLOR if map_mission_complete else (WARNING_COLOR if map_hazard_ahead else DASHBOARD_VALUE_COLOR)
            value_surf = font.render(value, True, value_color)
            screen.blit(label_surf, (x, y))
            screen.blit(value_surf, (x, y + 15))
            y += 33

    draw_column(col1, 16)
    draw_column(col2, 290)
    draw_column(col3, 570)
    press_r = font_tiny.render("Press R: setup   F11: fullscreen", True, (120, 120, 135))
    screen.blit(press_r, (WIDTH - 210, HEIGHT - 18))


MAP_PANEL_CONTENT_HEIGHT = 1800   # generous virtual height; actual content is measured and clamped each frame
map_panel_scroll = 0
map_panel_content_height = GRID_HEIGHT

def draw_map_setup_panel():
    """Renders the map-mode setup panel onto an off-screen surface (so it can be
    taller than the visible area) and blits only the scrolled-into-view slice.
    Sections are separated by divider lines + accent-colored headers so the
    panel reads as distinct groups instead of one long stack of controls."""
    global map_panel_content_height, map_panel_scroll

    panel = pygame.Surface((PANEL_WIDTH, MAP_PANEL_CONTENT_HEIGHT))
    panel.fill(PANEL_BG_COLOR)
    x = 14
    y = 14
    local_buttons = []   # (local_rect, action, target)

    panel.blit(font_title.render("REAL-WORLD MAP", True, TEXT_MAIN), (x, y)); y += 28

    def section(title):
        nonlocal y
        y += 6
        pygame.draw.line(panel, PANEL_AXIS_COLOR, (x - 2, y), (PANEL_WIDTH - 12, y), 1)
        y += 10
        panel.blit(font_small.render(title, True, PANEL_MARKER_COLOR), (x, y))
        y += 24

    def button(label, action, w=286, h=27, active=False, target=None, color_override=None):
        nonlocal y
        r = pygame.Rect(x, y, w, h)
        color = color_override if color_override else (BUTTON_ACTIVE_COLOR if active else BUTTON_COLOR)
        pygame.draw.rect(panel, color, r, border_radius=4)
        pygame.draw.rect(panel, PANEL_AXIS_COLOR, r, 1, border_radius=4)
        panel.blit(font_small.render(label, True, TEXT_MAIN), (r.x + 8, r.y + (r.height - font_small.get_height()) // 2))
        local_buttons.append((r, action, target))
        y += h + 7
        return r

    def button_pair(label1, action1, label2, action2, h=27, active1=False, active2=False, color1=None, color2=None):
        nonlocal y
        w = (286 - 10) // 2
        r1 = pygame.Rect(x, y, w, h)
        r2 = pygame.Rect(x + w + 10, y, w, h)
        for r, label, action, active, color_override in ((r1, label1, action1, active1, color1), (r2, label2, action2, active2, color2)):
            color = color_override if color_override else (BUTTON_ACTIVE_COLOR if active else BUTTON_COLOR)
            pygame.draw.rect(panel, color, r, border_radius=4)
            pygame.draw.rect(panel, PANEL_AXIS_COLOR, r, 1, border_radius=4)
            panel.blit(font_small.render(label, True, TEXT_MAIN), (r.x + 6, r.y + (r.height - font_small.get_height()) // 2))
            local_buttons.append((r, action, None))
        y += h + 7
        return r1, r2

    def caption(text, color=(150, 150, 165)):
        nonlocal y
        panel.blit(font_tiny.render(text, True, color), (x, y))
        y += 17

    # ---------------- MAP VIEW ----------------
    section("MAP VIEW")
    button_pair("Zoom -", "zoom_out", "Zoom +", "zoom_in", h=25)
    caption(f"Zoom level {map_zoom_level}  --  drag map to pan, scroll to zoom")

    # ---------------- START / GOAL ----------------
    section("START / GOAL")
    button_pair("Set START", "map_place_start", "Set GOAL", "map_place_goal", h=25,
                active1=(map_setup_mode == "placing_start"), active2=(map_setup_mode == "placing_goal"))
    button_pair(f"Start: {alt_ft(map_start_z):,} ft", "cycle_map_start_alt",
                f"Goal: {alt_ft(map_goal_z):,} ft", "cycle_map_goal_alt", h=25)
    caption(f"START  {map_start_latlon[0]:.5f}, {map_start_latlon[1]:.5f}")
    caption(f"GOAL   {map_goal_latlon[0]:.5f}, {map_goal_latlon[1]:.5f}")

    # ---------------- HAZARDS & OBSTACLES ----------------
    section("ADD ZONES")
    button_pair("+ Ash Zone", "add_map_ash", "+ Turbulence", "add_map_turbulence", h=25,
                active1=(map_pending_placement == "ash"), active2=(map_pending_placement == "turbulence"))
    button_pair("+ Obstacle (Full)", "add_map_obstacle_full", "+ Obstacle (Partial)", "add_map_obstacle_partial", h=25,
                active1=(map_pending_placement == "obstacle_full"), active2=(map_pending_placement == "obstacle_partial"),
                color1=BUTTON_OBSTACLE_COLOR if map_pending_placement != "obstacle_full" else None,
                color2=BUTTON_OBSTACLE_COLOR if map_pending_placement != "obstacle_partial" else None)

    mode_text = {
        "placing_start": "Click the map to set START.",
        "placing_goal": "Click the map to set GOAL.",
        "corner1": "Click the FIRST corner of the zone.",
        "corner2": "Click the SECOND corner of the zone.",
    }.get(map_setup_mode, "")
    if mode_text:
        caption(mode_text, PANEL_MARKER_COLOR)

    y += 2
    panel.blit(font_small.render(f"Hazards ({len(map_hazards)})", True, TEXT_MAIN), (x, y)); y += 22
    for h in map_hazards:
        row_rect = pygame.Rect(x, y, 246, 22)
        color = BUTTON_ACTIVE_COLOR if h["id"] == selected_map_hazard_id else BUTTON_COLOR
        pygame.draw.rect(panel, color, row_rect, border_radius=4)
        panel.blit(font_tiny.render(f'#{h["id"]} {h["label"]}', True, TEXT_MAIN), (row_rect.x + 6, row_rect.y + 4))
        local_buttons.append((row_rect, "select_map_hazard", h["id"]))
        rm_rect = pygame.Rect(x + 250, y, 28, 22)
        pygame.draw.rect(panel, (100, 55, 55), rm_rect, border_radius=4)
        panel.blit(font_tiny.render("X", True, TEXT_MAIN), (rm_rect.x + 9, rm_rect.y + 4))
        local_buttons.append((rm_rect, "remove_map_hazard", h["id"]))
        y += 26
    if not map_hazards:
        caption("(none placed yet)")

    selected_h = get_selected_map_hazard()
    if selected_h:
        y += 4
        panel.blit(font_tiny.render(f'Editing hazard #{selected_h["id"]} ({selected_h["label"]})', True, (185, 185, 200)), (x, y)); y += 18
        button_pair(f'Min: {alt_ft(selected_h["z_min"]):,} ft', "cycle_map_minz",
                    f'Max: {alt_ft(selected_h["z_max"]):,} ft', "cycle_map_maxz", h=24)
        button_pair("Drift: ON" if selected_h["drift_enabled"] else "Drift: OFF", "toggle_map_drift",
                    "Dir: ->" if selected_h["drift_dir"] == 1 else "Dir: <-", "toggle_map_dir", h=24)
        button(f'Speed: {selected_h["drift_interval"]}', "cycle_map_speed", h=24)

    y += 6
    panel.blit(font_small.render(f"Obstacles ({len(map_obstacles)})", True, TEXT_MAIN), (x, y)); y += 22
    for o in map_obstacles:
        is_full = (o["z_min"] == 0 and o["z_max"] == GRID_LAYERS - 1)
        row_rect = pygame.Rect(x, y, 246, 22)
        color = BUTTON_ACTIVE_COLOR if o["id"] == selected_map_obstacle_id else BUTTON_COLOR
        pygame.draw.rect(panel, color, row_rect, border_radius=4)
        kind = "Full" if is_full else f'{alt_ft(o["z_min"]):,}-{alt_ft(o["z_max"]):,}ft'
        panel.blit(font_tiny.render(f'#{o["id"]} Obstacle ({kind})', True, TEXT_MAIN), (row_rect.x + 6, row_rect.y + 4))
        local_buttons.append((row_rect, "select_map_obstacle", o["id"]))
        rm_rect = pygame.Rect(x + 250, y, 28, 22)
        pygame.draw.rect(panel, (100, 55, 55), rm_rect, border_radius=4)
        panel.blit(font_tiny.render("X", True, TEXT_MAIN), (rm_rect.x + 9, rm_rect.y + 4))
        local_buttons.append((rm_rect, "remove_map_obstacle", o["id"]))
        y += 26
    if not map_obstacles:
        caption("(none placed yet)")

    selected_o = get_selected_map_obstacle()
    if selected_o:
        y += 4
        panel.blit(font_tiny.render(f'Editing obstacle #{selected_o["id"]}', True, (185, 185, 200)), (x, y)); y += 18
        button_pair(f'Min: {alt_ft(selected_o["z_min"]):,} ft', "cycle_map_obstacle_minz",
                    f'Max: {alt_ft(selected_o["z_max"]):,} ft', "cycle_map_obstacle_maxz", h=24)

    # ---------------- SIMULATION ----------------
    section("SIMULATION")
    button("Start Map Simulation", "map_start_sim", h=32, color_override=BUTTON_START_COLOR)
    button("Back to Grid Mode", "switch_grid")
    caption(map_status[:52], WARNING_COLOR if "fail" in map_status.lower() else (170, 170, 185))

    # --- measure, clamp scroll, blit the visible slice, draw a scrollbar ---
    map_panel_content_height = y + 16
    max_scroll = max(0, map_panel_content_height - GRID_HEIGHT)
    map_panel_scroll = max(0, min(map_panel_scroll, max_scroll))
    scroll = map_panel_scroll

    pygame.draw.rect(screen, PANEL_BG_COLOR, (GRID_WIDTH, 0, PANEL_WIDTH, GRID_HEIGHT))
    screen.blit(panel, (GRID_WIDTH, 0), area=pygame.Rect(0, scroll, PANEL_WIDTH, GRID_HEIGHT))
    pygame.draw.line(screen, PANEL_AXIS_COLOR, (GRID_WIDTH, 0), (GRID_WIDTH, GRID_HEIGHT), 2)

    if max_scroll > 0:
        track = pygame.Rect(GRID_WIDTH + PANEL_WIDTH - 7, 4, 4, GRID_HEIGHT - 8)
        pygame.draw.rect(screen, (45, 46, 60), track, border_radius=2)
        thumb_h = max(30, int(track.height * GRID_HEIGHT / map_panel_content_height))
        thumb_y = track.y + int((track.height - thumb_h) * (scroll / max_scroll))
        pygame.draw.rect(screen, BUTTON_ACTIVE_COLOR, (track.x, thumb_y, track.width, thumb_h), border_radius=2)

    for r, action, target in local_buttons:
        abs_rect = pygame.Rect(GRID_WIDTH + r.x, r.y - scroll, r.w, r.h)
        active_buttons.append({"rect": abs_rect, "action": action, "target": target})


def handle_map_click(pos):
    global map_start_latlon, map_goal_latlon, map_setup_mode, map_status
    global map_pending_corner1_latlon, map_pending_placement
    global map_hazards, next_map_hazard_id, selected_map_hazard_id
    global map_obstacles, next_map_obstacle_id, selected_map_obstacle_id

    ll = screen_to_map_latlon(pos)
    if ll is None:
        return

    if map_setup_mode == "placing_start":
        map_start_latlon = ll
        map_setup_mode = "idle"
        map_status = "START updated."
    elif map_setup_mode == "placing_goal":
        map_goal_latlon = ll
        map_setup_mode = "idle"
        map_status = "GOAL updated."
    elif map_setup_mode == "corner1":
        map_pending_corner1_latlon = ll
        map_setup_mode = "corner2"
        map_status = "Click the second corner of the zone."
    elif map_setup_mode == "corner2":
        lat1, lon1 = map_pending_corner1_latlon
        lat2, lon2 = ll
        if map_pending_placement in ("ash", "turbulence"):
            h = make_map_hazard(next_map_hazard_id, map_pending_placement, lat1, lon1, lat2, lon2)
            map_hazards.append(h)
            selected_map_hazard_id = h["id"]
            next_map_hazard_id += 1
        elif map_pending_placement in ("obstacle_full", "obstacle_partial"):
            z_min, z_max = (0, GRID_LAYERS - 1) if map_pending_placement == "obstacle_full" else (0, 0)
            o = make_map_obstacle(next_map_obstacle_id, lat1, lon1, lat2, lon2, z_min, z_max)
            map_obstacles.append(o)
            selected_map_obstacle_id = o["id"]
            next_map_obstacle_id += 1
        map_setup_mode = "idle"
        map_pending_placement = None
        map_pending_corner1_latlon = None
        map_status = "Zone added."


def handle_map_action(action, target=None):
    global map_setup_mode, map_mode, sim_state, map_status, map_replan_count, map_last_replan_ms
    global map_pending_placement, map_start_z, map_goal_z
    global selected_map_hazard_id, map_hazards, selected_map_obstacle_id, map_obstacles
    global map_zoom_level, map_panel_scroll

    if action == "map_place_start":
        map_setup_mode = "placing_start"
    elif action == "map_place_goal":
        map_setup_mode = "placing_goal"
    elif action == "cycle_map_start_alt":
        map_start_z = (map_start_z + 1) % GRID_LAYERS
    elif action == "cycle_map_goal_alt":
        map_goal_z = (map_goal_z + 1) % GRID_LAYERS
    elif action == "add_map_ash":
        map_pending_placement = "ash"; map_setup_mode = "corner1"
    elif action == "add_map_turbulence":
        map_pending_placement = "turbulence"; map_setup_mode = "corner1"
    elif action == "add_map_obstacle_full":
        map_pending_placement = "obstacle_full"; map_setup_mode = "corner1"
    elif action == "add_map_obstacle_partial":
        map_pending_placement = "obstacle_partial"; map_setup_mode = "corner1"
    elif action == "select_map_hazard":
        selected_map_hazard_id = target
    elif action == "remove_map_hazard":
        map_hazards = [h for h in map_hazards if h["id"] != target]
        if selected_map_hazard_id == target:
            selected_map_hazard_id = map_hazards[0]["id"] if map_hazards else None
    elif action == "select_map_obstacle":
        selected_map_obstacle_id = target
    elif action == "remove_map_obstacle":
        map_obstacles = [o for o in map_obstacles if o["id"] != target]
        if selected_map_obstacle_id == target:
            selected_map_obstacle_id = map_obstacles[0]["id"] if map_obstacles else None
    elif action == "cycle_map_minz":
        h = get_selected_map_hazard()
        if h:
            h["z_min"] = (h["z_min"] + 1) % GRID_LAYERS
            if h["z_min"] > h["z_max"]:
                h["z_max"] = h["z_min"]
    elif action == "cycle_map_maxz":
        h = get_selected_map_hazard()
        if h:
            h["z_max"] = (h["z_max"] + 1) % GRID_LAYERS
            if h["z_max"] < h["z_min"]:
                h["z_min"] = h["z_max"]
    elif action == "toggle_map_drift":
        h = get_selected_map_hazard()
        if h:
            h["drift_enabled"] = not h["drift_enabled"]
    elif action == "toggle_map_dir":
        h = get_selected_map_hazard()
        if h:
            h["drift_dir"] *= -1
    elif action == "cycle_map_speed":
        h = get_selected_map_hazard()
        if h:
            idx = SPEED_PRESETS.index(h["drift_interval"]) if h["drift_interval"] in SPEED_PRESETS else 1
            h["drift_interval"] = SPEED_PRESETS[(idx + 1) % len(SPEED_PRESETS)]
    elif action == "cycle_map_obstacle_minz":
        o = get_selected_map_obstacle()
        if o:
            o["z_min"] = (o["z_min"] + 1) % GRID_LAYERS
            if o["z_min"] > o["z_max"]:
                o["z_max"] = o["z_min"]
    elif action == "cycle_map_obstacle_maxz":
        o = get_selected_map_obstacle()
        if o:
            o["z_max"] = (o["z_max"] + 1) % GRID_LAYERS
            if o["z_max"] < o["z_min"]:
                o["z_min"] = o["z_max"]
    elif action == "zoom_in":
        map_zoom_level = min(MAP_ZOOM_MAX, map_zoom_level + 1)
        load_real_map()
    elif action == "zoom_out":
        map_zoom_level = max(MAP_ZOOM_MIN, map_zoom_level - 1)
        load_real_map()
    elif action == "switch_grid":
        map_mode = False
        sim_state = "SETUP"
        map_setup_mode = "idle"
        map_panel_scroll = 0
        map_status = "Switched to Grid Mode."
    elif action == "map_start_sim":
        ok, elapsed, msg = build_map_route()
        map_status = msg if not ok else f"Route calculated in {elapsed:.3f} ms."
        if ok:
            sim_state = "RUNNING"


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

        elif event.type == pygame.VIDEORESIZE and not is_fullscreen:
            display_surface = pygame.display.set_mode(event.size, pygame.RESIZABLE)

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                set_display_mode(not is_fullscreen)
            elif event.key == pygame.K_m and sim_state == "SETUP":
                map_mode = not map_mode
                map_setup_mode = "idle"
                map_panel_scroll = 0
                if map_mode:
                    load_real_map()
                else:
                    setup_message = "Switched to Grid Mode."
            elif event.key == pygame.K_r and sim_state == "RUNNING":
                sim_state = "SETUP"
                if map_mode:
                    map_status = "Map simulation stopped."
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS) and map_mode and sim_state == "SETUP":
                map_zoom_level = min(MAP_ZOOM_MAX, map_zoom_level + 1)
                load_real_map()
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS) and map_mode and sim_state == "SETUP":
                map_zoom_level = max(MAP_ZOOM_MIN, map_zoom_level - 1)
                load_real_map()
            elif (event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN)
                  and map_mode and sim_state == "SETUP" and map_setup_mode == "idle" and map_pending_placement is None):
                step = 80
                dx = {pygame.K_LEFT: -step, pygame.K_RIGHT: step}.get(event.key, 0)
                dy = {pygame.K_UP: -step, pygame.K_DOWN: step}.get(event.key, 0)
                cx, cy = latlon_to_world(MAP_CENTER_LAT, MAP_CENTER_LON, map_zoom_level)
                MAP_CENTER_LAT, MAP_CENTER_LON = world_to_latlon(cx + dx, cy + dy, map_zoom_level)
                load_real_map()

        elif event.type == pygame.MOUSEWHEEL and map_mode and sim_state == "SETUP" and not map_is_panning:
            mx, my = mouse_canvas_pos()
            if mx < MAP_WIDTH and my < MAP_HEIGHT and event.y != 0:
                anchor = screen_to_map_latlon((mx, my))
                new_zoom = max(MAP_ZOOM_MIN, min(MAP_ZOOM_MAX, map_zoom_level + (1 if event.y > 0 else -1)))
                if new_zoom != map_zoom_level and anchor is not None:
                    map_zoom_level = new_zoom
                    MAP_CENTER_LAT, MAP_CENTER_LON = anchor
                    load_real_map()
            elif mx >= GRID_WIDTH and my < GRID_HEIGHT:
                map_panel_scroll -= event.y * 42

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and sim_state == "SETUP":
            pos = to_canvas_pos(event.pos)
            clicked = next((b for b in active_buttons if b["rect"].collidepoint(pos)), None)
            if clicked:
                if map_mode:
                    handle_map_action(clicked["action"], clicked.get("target"))
                else:
                    handle_action(clicked["action"], clicked.get("target"))
            elif map_mode and pos[0] < GRID_WIDTH and pos[1] < GRID_HEIGHT:
                if map_setup_mode in ("placing_start", "placing_goal", "corner1", "corner2"):
                    handle_map_click(pos)
                else:
                    map_is_panning = True
                    map_pan_last_pos = pos
                    map_pan_live_offset = (0, 0)
            elif (not map_mode) and pos[0] < GRID_WIDTH and pos[1] < GRID_HEIGHT:
                handle_grid_click(pos)

        elif event.type == pygame.MOUSEMOTION and map_is_panning:
            pos = to_canvas_pos(event.pos)
            dx = pos[0] - map_pan_last_pos[0]
            dy = pos[1] - map_pan_last_pos[1]
            map_pan_live_offset = (map_pan_live_offset[0] + dx, map_pan_live_offset[1] + dy)
            map_pan_last_pos = pos

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and map_is_panning:
            map_is_panning = False
            off_x, off_y = map_pan_live_offset
            if abs(off_x) > MAP_PAN_CLICK_THRESHOLD or abs(off_y) > MAP_PAN_CLICK_THRESHOLD:
                cx_world = map_origin_world[0] + MAP_WIDTH / 2 - off_x
                cy_world = map_origin_world[1] + MAP_HEIGHT / 2 - off_y
                MAP_CENTER_LAT, MAP_CENTER_LON = world_to_latlon(cx_world, cy_world, map_zoom_actual)
                load_real_map()
            map_pan_live_offset = (0, 0)

    if map_mode:
        if sim_state == "SETUP":
            active_buttons = []
            screen.fill(BG_COLOR)
            draw_real_map()
            draw_map_corner_preview()
            draw_map_setup_panel()
            hint = font_tiny.render("Press M: grid mode   F11: fullscreen   Drag map to pan, scroll to zoom", True, (145, 145, 160))
            screen.blit(hint, (12, GRID_HEIGHT - 22))
            present()
            clock.tick(30)
            continue

        # ---- Map-mode RUNNING: movement, hazard detection, replanning ----
        if not map_mission_complete:
            for h in map_hazards:
                update_map_hazard_drift(h)
            if map_current_segment < len(map_pixel_path) - 1:
                map_progress += 0.02
                if map_progress >= 1.0:
                    map_progress = 0.0
                    map_current_segment += 1
                    if map_current_segment != map_last_counted_segment:
                        if in_any_map_hazard(map_path[map_current_segment]):
                            map_hazard_cells_ever_crossed += 1
                        map_last_counted_segment = map_current_segment
            else:
                map_mission_complete = True
                map_mission_end_ticks = pygame.time.get_ticks()
                map_status = "Map mission complete. Press R to return to setup."

            if not map_mission_complete:
                map_hazard_ahead, map_hazard_cells = map_detect_hazard_ahead(map_current_segment)
                if map_hazard_ahead and not map_hazard_was_detected:
                    map_trigger_replan(map_current_segment)
                    map_replan_flash_timer = 30
                    map_hazard_ahead, map_hazard_cells = map_detect_hazard_ahead(map_current_segment)
                map_hazard_was_detected = map_hazard_ahead
                if map_replan_flash_timer > 0:
                    map_replan_flash_timer -= 1
        else:
            map_hazard_ahead, map_hazard_cells = False, []

        if map_path:
            if map_current_segment < len(map_path) - 1:
                current_map_z = map_path[map_current_segment][2]
            else:
                current_map_z = map_path[-1][2]
        else:
            current_map_z = 0

        active_buttons = []
        screen.fill(BG_COLOR)
        draw_real_map()
        draw_map_altitude_panel()
        draw_map_altitude_marker(map_current_segment, map_progress, current_map_z)
        draw_map_legend()
        if map_mission_complete:
            draw_map_mission_complete_banner()
        draw_map_dashboard()
        screen.blit(font_tiny.render("Press R to return to setup", True, (120, 120, 135)), (WIDTH - 190, HEIGHT - 18))
        present()
        clock.tick(60)
        continue

    # ------------------------- GRID MODE -------------------------
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
        active_buttons = layout_setup_ui()
        bottom_button = draw_setup_bottom_bar()
        active_buttons.append(bottom_button)
        # Map mode can also be entered from the keyboard with M.
        map_hint = font_tiny.render("Press M for Real-World Map Mode   F11 for fullscreen", True, (145, 145, 160))
        screen.blit(map_hint, (GRID_WIDTH + 16, GRID_HEIGHT - 22))

    present()
    clock.tick(60)

pygame.quit()