# id_controller — safely contained identification takes

Drives predefined feed-forward excitation maneuvers and records one bag per
take. A deliberately slow, capped pose-feedback correction keeps the car near
the intended path without hiding the faster identification signal. The map
safety layer reduces speed near walls and stops on stale or jumping localization.

## Quick start

Three terminals on the car (`RACECAR_VERSION` must be a `NUCx` — `config/SIM`
has no cartographer `.lua` files).

The bundled steering takes are bounded to ±0.34 rad for NUC2. Before using a
different car, compare that with its `actuation_manager.steering_min/max` and
revalidate the maneuver footprints.

```bash
# 1. once per room: build a map and press 'y' in this terminal to save it
ros2 launch id_controller sysid_bringup_launch.xml mapping:=True map_name:=room5x5

# 2. for every take afterwards: localization against that saved map
ros2 launch id_controller sysid_bringup_launch.xml mapping:=False map_name:=room5x5

# 3. run the takes, holding joystick button 5 throughout each one
ros2 run id_controller run_take M1_circle_0.34_L
ros2 run id_controller run_take room                           # all 19 box takes
ros2 run id_controller run_take corridor --repeat 4            # the 5 straight takes
ros2 run id_controller run_take --list                         # the manifest
```

`run_take` does the whole sequence per take: preflight, start `ros2 bag record`,
count down, play, stop, write a metadata sidecar. One take = one bag, never
concatenated. Multi-take runs pause for you to reposition the stopped car before
each new path. `--no-pause` suppresses this only when you intentionally want it.
When `--out` is omitted, bags go to `sysid_bags/` at the `race_stack` repository
root; the directory is created on the first recording and is gitignored. Docker
Compose sets `RACE_STACK_ROOT` to its bind-mounted source path, so the same
default writes into the host-visible project directory from inside the container.

## Check localization in RViz

Enable the focused RViz view while mapping or localizing:

```bash
ros2 launch id_controller sysid_bringup_launch.xml \
  mapping:=False map_name:=room5x5 rviz:=True
```

Its fixed frame is `map`; it overlays `/scan` on Cartographer's `/map`, overlays
the editable `/sysid/safety_map`, draws `/car_state/odom` as an arrow, shows the
blue `/sysid/reference_path`, marks `base_link`, and exposes the TF tree.
Before recording, drive slowly and check that the laser points remain on the
walls, the arrow moves and turns in the correct direction, returning to the
same physical spot returns to the same map pose, and no discontinuous jumps
occur. A plausible moving arrow alone is not enough: a scan that swims across
walls indicates bad localization or a bad sensor transform.

RViz needs a display. Over SSH, use X forwarding or run RViz on a ROS-connected
workstation with:

```bash
rviz2 -d "$(ros2 pkg prefix --share id_controller)/rviz/sysid_localization.rviz"
```

## Hold button 5

Button 5 is the autonomous deadman. The actuation manager only accepts
`/drive` while it is held. Button 4 is the human deadman and only authorizes
`/teleop`. The player also aborts if button 5 is released or button 4 is
pressed, so a human command cannot be mixed into a take. Hold button 5 for
preflight; after publishing the reference path, the player deliberately asks
you to release and press it again to arm the take.

## Entry points

| Command | What it does |
|---|---|
| `run_take` | preflight + record + play + metadata; the one you want |
| `play_take` | just plays a take to `/drive`; use when you run the bag yourself |
| `preflight` | just the command-path checks |
| `takes` | manifest and CSV export, no ROS needed |

```bash
ros2 run id_controller play_take --ros-args -p take:=M1_circle_0.20_L
ros2 run id_controller play_take --ros-args -p csv:=/path/to/take.csv
ros2 run id_controller preflight
python3 -m id_controller.takes --csv out/
```

Useful `run_take` flags: `--repeat N`, `--rate HZ`, `--no-bag`, `--dry-run`
(runs the timing loop and publishes nothing), `--no-deadman` (bench only),
`--skip-preflight`. `--open-loop` disables containment and is only for a
deliberately controlled comparison in a sufficiently large clear area.

## What preflight checks, and why

Each of these has silently ruined a recording before — the bag looks fine and is
unidentifiable.

- **`drive_exclusive`** — exactly one publisher on `/drive`. Containment lives
  inside `play_take`; any other controller would interleave commands with it.
- **`teleop_silent`** — unexpected human commands are reported, although button
  5 prevents them from being selected.
- **`deadman_held`** — button 5 held for the whole preflight window.
- **`applied_single_publisher`** — exactly one actuation manager publishes the
  SI-unit command that actually proceeds to conversion.
- **`single_publisher/commands/motor/speed`** and `.../servo/position` — expect
  exactly one (`ackermann_to_vesc`). See the note below.
- **`alive/...`** — the five live data/command topics are producing messages.

## Command path and recorded input

```text
/drive or /teleop
  -> actuation_manager
  -> /ackermann_cmd_applied       (m/s and radians)
  -> ackermann_to_vesc
  -> /commands/motor/speed        (ERPM)
     /commands/servo/position     (normalized servo position)
  -> vesc_driver
```

The player records `/sysid/nominal_cmd`, adds its slow steering correction in
`/sysid/containment_correction`, and publishes their bounded sum on `/drive`.
The manager applies the configured speed/steering bounds, acceleration and
steering-rate limits, deadman selection, and stale-command stop. Therefore the
requested steps on `/drive` may become ramps or clips. This is intentional and
observable: fit the vehicle model using `/ackermann_cmd_applied`, not `/drive`.
The metadata sidecar records those limits and the VESC conversion constants.

## Containment behavior

At the start of a take, the current map pose anchors its kinematic reference
path. The run is refused before motion if any reference point has less than
0.42 m occupancy-map clearance. Reposition or rotate the stationary car until
the blue path fits in RViz, then start the take again. For an accepted path the
player publishes it first and requires button 5 to be released and pressed
again, so you have time to inspect it while the command is still disarmed.

During the take, a Stanley-like steering correction is low-pass filtered,
rate-limited, and capped at 0.10 rad. Fast chirps and doublets are deliberately
excluded from the reference geometry, so the feedback does not chase them.
Closed circular references are matched by position rather than elapsed-time
phase because their angular progress is one of the unknown vehicle responses.
The M2 skidpad records some radial departure from its nominal circle, then ends
successfully at the 0.75 m cross-track limit: pushing wide is the intended
saturation signal. Heading loss and the live map-clearance checks remain aborts.
The predictor begins reducing requested speed below 0.75 m clearance. It stops
at 0.32 m, approximately the footprint radius, and aborts if the measured car
centre crosses that hard limit. Releasing button 5 is still the primary stop.

These are conservative initial values, not a guarantee against collision:
Cartographer delay, map error, tire slip, and braking distance still matter.
Start with `M1_circle_0.34_L` at 0.6 m/s while watching RViz. Only increase the
maneuver severity after the path, pose, correction, and stopping behavior look
sound. Use `/ackermann_cmd_applied` for fitting; the two `/sysid/...` command
topics let analysis measure or reject intervals with excessive intervention.

## Editable keep-out map

Containment deliberately does not use Cartographer's live `/map`. Localization
continues to scan-match against the saved `.pbstream`, while a dedicated map
server loads `safety_map_yaml` onto `/sysid/safety_map`. By default this is the
saved `<map_name>.yaml` and PNG.

To add virtual keep-out areas, preferably copy the PNG and YAML to names such as
`room5x5_safety.png` and `room5x5_safety.yaml`, then change the YAML's `image`
entry to the copied PNG. Paint forbidden space black; leave usable space white.
Unknown/gray space and pixels above the occupied threshold are also treated as
blocked. Preserve the image dimensions, orientation, YAML resolution, and YAML
origin so the mask stays registered with Cartographer.

Launch localization with the edited safety YAML:

```bash
ros2 launch id_controller sysid_bringup_launch.xml \
  mapping:=False map_name:=room5x5 rviz:=True \
  safety_map_yaml:=/home/race_crew/ws/src/race_stack/stack_master/maps/room5x5/room5x5_safety.yaml
```

Restart the launch after every edit. If you edit the default map inside the
source tree rather than passing an absolute safety YAML, rebuild `stack_master`
and source the workspace so its installed share contains the change. Do not
edit, resize, or regenerate the `.pbstream` for keep-out zones.

The old mux and throttle interpolator are not in the active path. The former
hardcoded steering factor is already rolled into
`steering_angle_to_servo_gain` in each car's `vesc.yaml`.

## Safety

`play_take` publishes several zero commands on every normal exit. In addition,
the actuation manager stops on a stale input/deadman and the VESC driver has a
final command watchdog. Releasing button 5 remains the operator's immediate
stop action.

M2 is the one take that deliberately drives to the limit and will push wide at
the end by design. Run it with the most clearance you have.
