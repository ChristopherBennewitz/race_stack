# id_controller — open-loop identification takes

Drives predefined feed-forward excitation maneuvers and records one bag per
take. Nothing in this package reacts to the car except the abort path, which
can stop it but never steer it.

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
ros2 run id_controller run_take M2_skidpad_L --out ~/sysid_bags
ros2 run id_controller run_take room --out ~/sysid_bags        # all 19 box takes
ros2 run id_controller run_take corridor --repeat 4            # the 5 straight takes
ros2 run id_controller run_take --list                         # the manifest
```

`run_take` does the whole sequence per take: preflight, start `ros2 bag record`,
count down, play, stop, write a metadata sidecar. One take = one bag, never
concatenated.

## Check localization in RViz

Enable the focused RViz view while mapping or localizing:

```bash
ros2 launch id_controller sysid_bringup_launch.xml \
  mapping:=False map_name:=room5x5 rviz:=True
```

Its fixed frame is `map`; it overlays `/scan` on `/map`, draws the
`/car_state/odom` pose as an arrow, marks `base_link`, and exposes the TF tree.
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
pressed, so a human command cannot be mixed into a take.

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
`--skip-preflight`.

## What preflight checks, and why

Each of these has silently ruined a recording before — the bag looks fine and is
unidentifiable.

- **`drive_exclusive`** — exactly one publisher on `/drive`. If the controller or
  state machine is up it publishes too, the streams interleave, and the command
  ends up correlated with the state it was reacting to.
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

The manager applies the configured speed/steering bounds, acceleration and
steering-rate limits, deadman selection, and stale-command stop. Therefore the
requested steps on `/drive` may become ramps or clips. This is intentional and
observable: fit the vehicle model using `/ackermann_cmd_applied`, not `/drive`.
The metadata sidecar records those limits and the VESC conversion constants.

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
