import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'misc'))
import offline_slam_poses as osp  # noqa: E402
from ekf_replay import load_pose_file  # noqa: E402


@pytest.fixture
def config_root(tmp_path):
    root = tmp_path / 'config'
    (root / 'slam').mkdir(parents=True)
    (root / 'slam' / 'base.lua').write_text('options = {use_odometry = true, published_frame = "odom"}\nreturn options\n')
    (root / 'slam' / 'loc.lua').write_text('-- localization\n')
    (root / 'NUC9' / 'slam').mkdir(parents=True)
    (root / 'NUC9' / 'slam' / 'f110_2d.lua').write_text('include "../../slam/base.lua"\nreturn options\n')
    (root / 'NUC9' / 'slam' / 'f110_2d_loc.lua').write_text('include "../../slam/base.lua"\ninclude "../../slam/loc.lua"\nreturn options\n')
    return root


def includes(directory, name):
    return re.findall(r'include "([^"]+)"', (Path(directory) / name).read_text())


def resolves(directory, name):
    """Every include of ``name``, and of what it includes, exists relative to ``directory``."""
    for inc in includes(directory, name):
        target = Path(directory) / inc
        if not target.exists() or not resolves(target.parent, target.name):
            return False
    return True


def test_derived_config_keeps_the_relative_includes_working_and_drops_the_odometry(tmp_path, config_root):
    directory, basename = osp.derive_config(config_root, 'NUC9', tmp_path / 'work')
    text = (directory / basename).read_text()
    assert includes(directory, basename) == ['f110_2d.lua']
    assert resolves(directory, basename)
    assert 'options.use_odometry = false' in text
    assert 'options.provide_odom_frame = false' in text
    assert 'return options' in text


def test_localization_mode_starts_from_the_localization_config(tmp_path, config_root):
    directory, basename = osp.derive_config(config_root, 'NUC9', tmp_path / 'work', localization=True)
    assert includes(directory, basename) == ['f110_2d_loc.lua']
    assert resolves(directory, basename)


def test_a_missing_car_configuration_is_reported(tmp_path, config_root):
    with pytest.raises(FileNotFoundError):
        osp.derive_config(config_root, 'NUC1', tmp_path / 'work')


def test_commands_replay_only_what_cartographer_needs():
    cmds = osp.commands('bag', '/cfg', 'offline_2d.lua', 'out.csv', pbstream='map.pbstream', rate=0.5)
    player = cmds['player']
    assert player[:4] == ['ros2', 'bag', 'play', 'bag'] and '--clock' in player
    assert player[player.index('--topics') + 1:] == ['/scan', '/tf_static']  # not /tf, /odom, /imu: the live estimator's
    assert player[player.index('--rate') + 1] == '0.5'
    cartographer = cmds['cartographer']
    assert cartographer[cartographer.index('-load_state_filename') + 1] == 'map.pbstream'
    assert 'use_sim_time:=true' in cartographer
    assert '-load_state_filename' not in osp.commands('bag', '/cfg', 'x.lua', 'o.csv')['cartographer']
    assert cmds['recorder'][-2:] == ['--record', 'out.csv']


def test_the_run_gets_a_private_loopback_dds_domain():
    env = osp.environment(77, 'rmw_fastrtps_cpp')
    assert env['ROS_DOMAIN_ID'] == '77' and env['ROS_LOCALHOST_ONLY'] == '1'
    assert env['RMW_IMPLEMENTATION'] == 'rmw_fastrtps_cpp'


def test_pose_rows_use_the_header_stamp_and_the_yaw_of_the_quaternion():
    yaw = 0.7
    msg = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=500_000_000)),
        pose=SimpleNamespace(position=SimpleNamespace(x=1.5, y=-2.0),
                             orientation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))))
    assert osp.pose_row(msg) == pytest.approx((12.5, 1.5, -2.0, yaw))


def test_recorded_poses_load_into_the_replay_and_are_summarized(tmp_path):
    path = tmp_path / 'take.csv'
    rows = [(10.0 + 0.1 * k, 0.5 * k, 0.0, 0.0) for k in range(11)]
    path.write_text('\n'.join(osp.format_row(r) for r in rows) + '\n')
    n, duration, length = osp.summarize_csv(path)
    assert (n, duration, length) == pytest.approx((11, 1.0, 5.0))
    loaded = load_pose_file(path, latency=0.02)
    np.testing.assert_allclose(loaded[:, 1], [r[0] for r in rows])  # the stamp
    np.testing.assert_allclose(loaded[:, 0], loaded[:, 1] + 0.02)   # arrival
    np.testing.assert_allclose(loaded[:, 2], [r[1] for r in rows])


def test_a_pose_file_with_a_single_row_is_not_a_trajectory(tmp_path):
    path = tmp_path / 'short.csv'
    path.write_text('1,0,0,0\n')
    assert osp.summarize_csv(path) is None
