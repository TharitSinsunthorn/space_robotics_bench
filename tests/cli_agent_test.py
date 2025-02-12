import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable

import pytest

from srb.utils import logging
from srb.utils.cache import read_env_list_cache
from srb.utils.isaacsim import get_isaacsim_python
from srb.utils.subprocess import terminate_process

HEADLESS: bool = True
TEST_VISUAL_ENVS: bool = True

MAX_NUM_ENVS: int = 256
MAX_DURATION_RAND: float = 20.0
MAX_DURATION_TRAIN: float = 60.0


def _list_envs() -> Iterable[str]:
    if envs := read_env_list_cache():
        if not TEST_VISUAL_ENVS:
            envs = filter(lambda env: not _is_visual_env(env), envs)
        envs = sorted(envs)
    else:
        test_filepath = Path(__file__)
        logging.warning(
            f"Skipping {test_filepath.parent.name}/{test_filepath.name} because no environments were found in the cache (repeat the test)"
        )
        envs = []

    return envs


def _is_visual_env(env: str) -> bool:
    return env.endswith("_visual")


@pytest.mark.parametrize(
    "env,num_envs",
    (
        (env, num_envs)
        for env in _list_envs()
        for num_envs in (
            (1, math.floor(math.sqrt(MAX_NUM_ENVS)), MAX_NUM_ENVS)
            if MAX_NUM_ENVS > 1
            else (1,)
        )
        if (not _is_visual_env(env) or (num_envs < MAX_NUM_ENVS and MAX_NUM_ENVS > 1))
    ),
)
def test_cli_agent_rand(env: str, num_envs: int):
    cmd = (
        get_isaacsim_python(),
        "-m",
        "srb",
        "agent",
        "rand",
        "--headless" if HEADLESS else "--hide_ui",
        f"env.scene.num_envs={num_envs}",
        "--env",
        env,
    )

    environ = os.environ.copy()
    environ["SF_BAKER"] = "0"

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            env=environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )

        start = time.time()
        while time.time() - start < MAX_DURATION_RAND:
            if process.poll() is not None:
                logging.critical(f"[{env}] Failed command: {' '.join(cmd)}")
                stdout, stderr = process.communicate()
                pytest.fail(
                    f'Process failed for env "{env}"\n[env={env}] STDOUT:\n{stdout}\n[env={env}] STDERR:\n{stderr}'
                )
            time.sleep(0.01 * MAX_DURATION_RAND)
    except Exception as e:
        logging.critical(f"[{env}] Failed command: {' '.join(cmd)}")
        pytest.fail(
            f'Failed to start process for env "{env}"\n[env={env}] Exception: {e}'
        )
    finally:
        terminate_process("srb", process)


@pytest.mark.parametrize(
    "env,num_envs,algo",
    (
        (env, num_envs, algo)
        for env in _list_envs()
        for num_envs in (
            (1, math.floor(MAX_NUM_ENVS ** (1.0 / 3.0))) if MAX_NUM_ENVS > 1 else (1,)
        )
        for algo in ("dreamer",)
        if (not _is_visual_env(env) or (num_envs < MAX_NUM_ENVS and MAX_NUM_ENVS > 1))
    ),
)
def test_cli_agent_train(env: str, num_envs: int, algo: str):
    with tempfile.TemporaryDirectory(prefix="srb_") as tmpdir:
        match algo:
            case _:
                agent_params = ()

        cmd = (
            get_isaacsim_python(),
            "-m",
            "srb",
            "agent",
            "train",
            "--headless" if HEADLESS else "--hide_ui",
            f"env.scene.num_envs={num_envs}",
            "--env",
            env,
            "--algo",
            algo,
            "--logs",
            tmpdir,
            *agent_params,
        )

        environ = os.environ.copy()
        environ["SF_BAKER"] = "0"

        process = None
        try:
            process = subprocess.Popen(
                cmd,
                env=environ,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid,
            )

            start = time.time()
            while time.time() - start < MAX_DURATION_TRAIN:
                if process.poll() is not None:
                    logging.critical(f"[{env}] Failed command: {' '.join(cmd)}")
                    stdout, stderr = process.communicate()
                    pytest.fail(
                        f'Process failed for env "{env}"\n[env={env}] STDOUT:\n{stdout}\n[env={env}] STDERR:\n{stderr}'
                    )
                time.sleep(0.01 * MAX_DURATION_TRAIN)
        except Exception as e:
            logging.critical(f"[{env}] Failed command: {' '.join(cmd)}")
            pytest.fail(
                f'Failed to start process for env "{env}"\n[env={env}] Exception: {e}'
            )
        finally:
            terminate_process("srb", process)
