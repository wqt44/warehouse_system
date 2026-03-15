"""
warehouse_env 模块测试：通过 create_env 构造环境，测试 reset、step、充电站分配、移动规则等
"""
import pytest

from utils.env_utils import create_env


def _minimal_config():
    """最小可运行配置：小网格、少量机器人、明确充电站"""
    return {
        "warehouse": {
            "width": 8,
            "height": 8,
            "shelf_regions": [[2, 2, 4, 4]],
            "workstation_positions": [[6, 6]],
            "obstacle_positions": [[1, 1]],
            "charging_stations": [[0, 0], [7, 0]],
            "task_spawn_rate": 0.0,
            "max_tasks": 2,
        },
        "robot": {
            "num_robots": 2,
            "max_battery": 100.0,
            "battery_consumption_rate": 0.1,
            "charging_rate": 2.0,
        },
        "max_steps": 100,
    }


class TestCreateEnvAndReset:
    """环境创建与 reset"""

    def test_create_env_returns_warehouse_env(self):
        env = create_env(_minimal_config())
        assert env is not None
        assert hasattr(env, "reset") and hasattr(env, "step")
        assert env.width == 8 and env.height == 8
        assert len(env.robots) == 2

    def test_reset_returns_observations_and_infos(self):
        env = create_env(_minimal_config())
        observations, infos = env.reset(seed=42)
        assert isinstance(observations, dict)
        assert isinstance(infos, dict)
        for rid in range(len(env.robots)):
            assert rid in observations
            assert rid in infos

    def test_reset_with_seed_reproducible_robot_positions(self):
        env = create_env(_minimal_config())
        env.reset(seed=123)
        pos1 = [env.robots[i].position for i in range(len(env.robots))]
        env.reset(seed=123)
        pos2 = [env.robots[i].position for i in range(len(env.robots))]
        assert pos1 == pos2


class TestChargingStationAssignment:
    """充电站分配：robot_id % n"""

    def test_get_charging_station_for_robot(self):
        env = create_env(_minimal_config())
        env.reset(seed=0)
        # 充电站 [(0,0), (7,0)]，robot_id % 2
        s0 = env.get_charging_station_for_robot(0)
        s1 = env.get_charging_station_for_robot(1)
        assert s0 == (0, 0)
        assert s1 == (7, 0)

    def test_get_charging_station_for_robot_mod_wrap(self):
        env = create_env(_minimal_config())
        env.reset(seed=0)
        s2 = env.get_charging_station_for_robot(2)
        assert s2 == (0, 0)


class TestStepAndMovement:
    """step 与移动规则"""

    def test_step_returns_five_tuple(self):
        env = create_env(_minimal_config())
        env.reset(seed=42)
        actions = {0: 4, 1: 4}
        obs, rewards, terminated, truncated, infos = env.step(actions)
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminated, dict)
        assert isinstance(truncated, dict)
        assert isinstance(infos, dict)
        for r in env.robots:
            assert r.robot_id in rewards and r.robot_id in terminated and r.robot_id in truncated

    def test_wait_action_keeps_positions(self):
        env = create_env(_minimal_config())
        env.reset(seed=42)
        positions_before = [env.robots[i].position for i in range(len(env.robots))]
        actions = {i: 4 for i in range(len(env.robots))}
        env.step(actions)
        positions_after = [env.robots[i].position for i in range(len(env.robots))]
        assert positions_before == positions_after

    def test_move_into_obstacle_stays_put(self):
        config = _minimal_config()
        config["warehouse"]["obstacle_positions"] = [[1, 0]]
        config["warehouse"]["charging_stations"] = [[0, 0]]
        env = create_env(config)
        env.reset(seed=42)
        # 假设机器人 0 在 (0,0)，向右 (1,0) 是障碍则不应移动
        for r in env.robots:
            if r.position == (0, 0):
                actions = {r.robot_id: 1}
                env.step(actions)
                assert env.robots[r.robot_id].position == (0, 0)
                break
