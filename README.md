A Pygame-based educational simulation that demonstrates live hazard detection and on-the-fly trajectory replanning for a drone. The aircraft begins on a simple nominal route. A forward-looking sensor continuously scans ahead; when volcanic ash, moderate/high turbulence, or solid obstacles enter the sensor footprint, the system invokes A* and locks a new safe trajectory.

Key Capabilities

Two environments
Grid Mode – Discrete 3-D grid (columns × rows × altitude layers) for clear visualisation of the algorithm.
Real-World Map Mode – OpenStreetMap tiles (Bengaluru area by default) with geographic start/goal/waypoints and synthetic hazard zones.

Live sensing & adaptive replanning
Sensor range: 6 cells.
Nominal path is hazard-blind; A* is called only after a threat is detected.
Avoidance rules:
Volcanic ash and moderate/high turbulence are hard exclusion volumes.
Low turbulence is a soft (cost-only) pass-through region.
Full-height and partial-altitude obstacles are solid barriers.


3-D flight model
9 altitude layers (0 – 40 000 ft in 5 000 ft steps).
Vertical moves carry higher cost; wind penalty is applied when flying against the prevailing wind.

Mission planning tools
Place / edit Start, Goal and intermediate waypoints (each with selectable altitude).
Draw rectangular ash, turbulence and obstacle zones by two-corner selection.
Random scenario generator that guarantees at least one route-intersecting high-severity threat for reliable demos.
Scrollable side panel for all controls.

Telemetry & visualisation
Real-time altitude-vs-distance profile with threat bands localised to the actual route intercept.
Dashboard: mission time, path cost, altitude changes, hazard cells crossed, replan count & latency, estimated energy, route status, speed.
Mission Performance Report overlay.
HUD-style event banners for detection and course-correction events.



Requirements
BashPython 3.8+
pygame
requests          # Map Mode only
Pillow            # Map Mode only
Install with:
Bashpip install pygame requests Pillow
Internet access is required the first time Map Mode downloads OSM tiles (they are cached locally under osm_tile_cache/).

Running the Application
Bashpython adaptive_drone_replanner.py
(Use the actual filename of the script you saved.)
The application opens on a short boot screen, then presents two large buttons:

GRID MODE – PRESS G
REAL-WORLD MAP MODE – PRESS M

You can also press G or M on the home screen.

Quick-Start Workflow
Grid Mode

Use the right-hand panel to:
Place Start / Goal (or cycle their altitudes).
Add waypoints at a chosen altitude.
Draw ash / turbulence / obstacle zones (click “Add …”, then two corners on the grid).
Optionally press RANDOMIZE SCENARIO.

Click LAUNCH LIVE-REPLAN MISSION.
Watch the drone fly the nominal path. When a threat enters the sensor circle the system announces detection, pauses briefly, then locks a corrected trajectory (old path shown dashed).
Press P for the performance report, R to reset the flight while keeping the scenario, or use the panel buttons.

Map Mode

Click REFRESH MAP TILES (or let it load automatically).
Place Start / Goal / waypoints by clicking the map.
Draw geographic hazard/obstacle rectangles the same way.
Use mouse-wheel over the map to zoom, drag to pan, wheel over the panel to scroll.
Launch the mission. Behaviour is identical to Grid Mode; the altitude profile and dashboard remain fully functional.


Keyboard Shortcuts

KeyActionG / MSwitch to Grid / Map mode (home or setup)RReset current flight (scenario retained)PToggle Mission Performance Report (while running)F11Toggle fullscreen+ / -Zoom map (Map Mode, setup)EscCancel current placement actionMouse wheelPanel scroll (right side) or map zoom (over map)

Important Notes for Judges & Users

Path costs, energy figures and wind penalties are illustrative demonstrator values, not flight-control limits.
The nominal route deliberately ignores hazards so that the live sensor + replan sequence is visible every time.
Ash plumes that reach the top altitude layer are treated as full vertical exclusion volumes (the planner cannot “climb over” them).
Low turbulence is intentionally pass-through; only moderate/high turbulence and ash force a replan.
Map tiles are cached; subsequent runs work offline once the desired zoom levels have been downloaded.