from pyutils.pylogger import Logger
import subprocess


from shutil import which
import subprocess
from typing import Optional

from pyutils.pylogger import Logger


class Display:
    """Run EventDisplay helper commands.

    Parameters
    - verbosity: logger verbosity (int)
    - config: path to an EventDisplay FCL config used by `mu2e` when
      launching the display.

      Note:
      For this to work:
      * mu2einit
      * muse setup
      * assumes local copy of EventDisplay via clone or musing
    """

    def __init__(self, verbosity: int = 1, config: str = "/exp/mu2e/app/users/sophie/newOffline/EventDisplay/examples/nominal_MDC2025.fcl") -> None:
        self.logger = Logger(print_prefix="[pydisplay]", verbosity=verbosity)
        self.config = config

    def _check_cmd(self, cmd: str) -> bool:
        """Return True if ``cmd`` is available on PATH, log otherwise."""
        if which(cmd) is None:
            self.logger.log(f"Required command '{cmd}' not found on PATH", "error")
            return False
        return True

    def pick_event(self, dataset: str, run: int, subrun: int, event: int) -> Optional[str]:
        """Invoke ``pickEvent`` to extract a single event.

        Returns stdout on success, otherwise returns ``None``.
        """
        if not self._check_cmd("pickEvent"):
            return None

        target = f"{run}/{subrun}/{event}"
        cmd = ["pickEvent", "-e", "-v", str(dataset), target]
        self.logger.log(f"Running: {' '.join(cmd)}", "info")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            self.logger.log(f"pickEvent failed: {proc.stderr}", "error")
            return None

        self.logger.log("pickEvent completed", "success")
        return proc.stdout

    def launch_display(self, dataset: str, run: int, subrun: int, event: int, artifact_dir: str = ".") -> Optional[str]:
        """Launch the EventDisplay pointing at an artifact file.

        The method builds the artifact filename from the identifiers passed
        and invokes the ``mu2e`` command with the configured FCL file.
        Returns stdout on success, otherwise ``None``.
        """
        if not self._check_cmd("mu2e"):
            return None
        print(f"Launching display for {dataset} run {run} subrun {subrun} event {event} with config {self.config}")
        artifact = f"{dataset}_{run}_{subrun}_{event}.art"
        cmd = ["mu2e", "-c", self.config, artifact]
        self.logger.log(f"Launching display: {' '.join(cmd)}", "info")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            self.logger.log(f"mu2e failed: {proc.stderr}", "error")
            return None

        self.logger.log("EventDisplay launched", "success")
        return proc.stdout


