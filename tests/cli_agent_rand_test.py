import os
import subprocess
import time
from pathlib import Path

import pytest

from srb.utils import logging
from srb.utils.cache import read_env_list_cache
from srb.utils.isaacsim import get_isaacsim_python
from srb.utils.subprocess import terminate_process

TEST_DURATION: float = 20.0


if envs := read_env_list_cache():

    @pytest.mark.parametrize("env", envs)
    def test_cli_agent_rand(env):
        cmd = (
            get_isaacsim_python(),
            "-m",
            "srb",
            "agent",
            "rand",
            "--headless",
            "--env",
            env,
        )

        environ = os.environ.copy()
        environ["SF_BAKER"] = "false"

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
                text=True,
                env=environ,
            )

            start = time.time()
            while time.time() - start < TEST_DURATION:
                if process.poll():
                    stdout, stderr = process.communicate()
                    pytest.fail(
                        f'Process failed for env "{env}"\n[env={env}] STDOUT:\n{stdout}\n[env={env}] STDERR:\n{stderr}'
                    )
                time.sleep(0.1)
        except Exception as e:
            pytest.fail(
                f'Failed to start process for env "{env}"\n[env={env}] Exception: {e}'
            )
        finally:
            terminate_process(cmd, process)

else:
    test_filepath = Path(__file__)
    logging.warning(
        f"Skipping {test_filepath.parent.name}/{test_filepath.name} because no environments were found in the cache"
    )
