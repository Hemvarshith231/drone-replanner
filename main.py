import pygame

pygame.init()

# --- Window and grid settings ---
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 40

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Replanner - Stage 2")
clock = pygame.time.Clock()

# --- Colors ---
BG_COLOR = (30, 30, 40)
GRID_COLOR = (60, 60, 75)
START_COLOR = (80, 200, 120)
GOAL_COLOR = (220, 90, 90)
DRONE_COLOR = (90, 160, 230)
PATH_COLOR = (120, 120, 160)

# --- Hardcoded path (list of grid cells, in order) ---
# Later, A* will generate this list automatically. For now, we fake it by hand.
path = [
    (1, 1),
    (4, 1),
    (4, 5),
    (8, 5),
    (8, 9),
    (13, 9),
    (13, 3),
    (18, 3),
    (18, 13),
]

def grid_to_pixel(cell):
    col, row = cell
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

def draw_path():
    if len(pixel_path) > 1:
        pygame.draw.lines(screen, PATH_COLOR, False, pixel_path, 3)

# --- Drone movement state ---
current_segment = 0   # which pair of waypoints we're currently moving between
progress = 0.0         # 0.0 to 1.0 across the CURRENT segment only
speed = 0.02           # progress gained per frame

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Only move if we haven't reached the final waypoint yet
    if current_segment < len(pixel_path) - 1:
        progress += speed
        if progress >= 1.0:
            progress = 0.0
            current_segment += 1  # move on to the next segment

    # Figure out drone's pixel position based on current segment + progress
    if current_segment < len(pixel_path) - 1:
        start_point = pixel_path[current_segment]
        end_point = pixel_path[current_segment + 1]
        drone_x = lerp(start_point[0], end_point[0], progress)
        drone_y = lerp(start_point[1], end_point[1], progress)
    else:
        # Reached the end of the path — stay at the final point
        drone_x, drone_y = pixel_path[-1]

    # --- Drawing ---
    screen.fill(BG_COLOR)
    draw_grid()
    draw_path()

    pygame.draw.circle(screen, START_COLOR, pixel_path[0], 10)
    pygame.draw.circle(screen, GOAL_COLOR, pixel_path[-1], 10)
    pygame.draw.circle(screen, DRONE_COLOR, (int(drone_x), int(drone_y)), 8)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()