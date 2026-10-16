# id_controller — joystick-driven identification takes

Records one bag per manually driven take. Nothing is scripted: you drive with the
joystick and the recorder captures the command path (`/teleop`, `/joy`,
`/ackermann_cmd_applied`) together with the state and sensor topics needed to fit
a vehicle model.

## Quick start

Two terminals on the car (`RACECAR_VERSION` must be a `NUCx` — `config/SIM`
has no cartographer `.lua` files).

```bash
# 1. once per room: build a map and press 'y' in this terminal to save it
ros2 launch id_controller sysid_bringup_launch.xml mapping:=True map_name:=room5x5

# 2. for every take afterwards: localization against that saved map
ros2 launch id_controller sysid_bringup_launch.xml mapping:=False map_name:=room5x5

# 3. record a take (hold button 4 to drive, release it to stop)
ros2 run id_controller record_teleop slaloms
```

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

Its fixed frame is `map`; it overlays `/scan` on Cartographer's `/map`, draws `/car_state/odom` as an arrow, shows marks `base_link`, and exposes the TF tree.
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

## Deadman buttons

Button 4 is the human deadman and authorizes `/teleop`. Button 5 is the
autonomous deadman for `/drive`; nothing should publish `/drive` during a take.

## Entry points

| Command | What it does |
|---|---|
| `record_teleop` | preflight + record one manually driven take while button 4 is held |
| `preflight` | just the command-path checks |

`record_teleop` first requires button 4 to be released, starts and settles the
bag recorder, then asks you to press and hold button 4. Drive the maneuver and
release button 4 to command the car to stop; recording continues for a one-second
tail before the bag is closed. Finally, answer the keep/discard prompt. There is
no path containment, so the driver remains responsible for staying inside the
available space. Flags: `--out`, `--tail`, `--preflight-window`, `--skip-preflight`.

## What preflight checks, and why

Each of these has silently ruined a recording before — the bag looks fine and is
unidentifiable.

- **`drive_exclusive`** — nothing publishes `/drive`.
- **`teleop_silent`** — unexpected `/teleop` traffic before the take is reported.
- **`applied_single_publisher`** — exactly one actuation manager publishes the
  SI-unit command that actually proceeds to conversion.
- **`single_publisher/commands/motor/speed`** and `.../servo/position` — expect
  exactly one (`ackermann_to_vesc`).
- **`alive/...`** — the live data/command topics (including the raw VESC `/odom` and the
  estimator's odometry) are producing messages.

## Command path and recorded input

```text
/teleop
  -> actuation_manager
  -> /ackermann_cmd_applied       (m/s and radians)
  -> ackermann_to_vesc
  -> /commands/motor/speed        (ERPM)
     /commands/servo/position     (normalized servo position)
  -> vesc_driver
```

The manager applies the configured speed/steering bounds, acceleration and
steering-rate limits, deadman selection, and stale-command stop, so requested
steps on `/teleop` may become ramps or clips. Fit the vehicle model using
`/ackermann_cmd_applied`, not `/teleop`. The metadata sidecar records those
limits and the VESC conversion constants.

## Safety

The actuation manager stops on a stale input or released deadman and the VESC
driver has a final command watchdog. Releasing button 4 is the operator's
immediate stop action.
