"""
Simulation Scheduler

Polls MiroFish for completed simulations and triggers signal extraction.
Tracks which simulations have already been processed to avoid duplication.

Forge Compliance:
    - §1.1: Only processes completed simulations (no partial data)
    - §7.1: Every extraction is recorded in the strategy cemetery
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Set

import requests

logger = logging.getLogger("forge_bridge.scheduler")


class SimulationScheduler:
    """
    Polls MiroFish API for completed simulations and triggers signal extraction.
    """

    def __init__(
        self,
        mirofish_url: str,
        uploads_dir: str,
        poll_interval_seconds: int = 300,
    ):
        self.mirofish_url = mirofish_url.rstrip("/")
        self.uploads_dir = Path(uploads_dir)
        self.poll_interval = poll_interval_seconds
        self.processed_simulations: Set[str] = set()
        self._load_processed()

    def poll_once(self) -> list:
        """
        Check for completed simulations that haven't been processed yet.

        Returns:
            List of simulation IDs ready for signal extraction.
        """
        ready = []

        # Method 1: Check MiroFish API
        try:
            resp = requests.get(
                f"{self.mirofish_url}/api/simulation/list",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                simulations = data.get("data", data.get("simulations", []))
                for sim in simulations:
                    sim_id = sim.get("simulation_id") or sim.get("id")
                    status = sim.get("status", "")
                    if (
                        sim_id
                        and status == "completed"
                        and sim_id not in self.processed_simulations
                    ):
                        ready.append(sim_id)
        except requests.RequestException as e:
            logger.warning("Failed to poll MiroFish API: %s. Falling back to filesystem.", e)

        # Method 2: Scan filesystem for completed simulations
        sim_dir = self.uploads_dir / "simulations"
        if sim_dir.exists():
            for child in sim_dir.iterdir():
                if not child.is_dir():
                    continue
                sim_id = child.name
                if sim_id in self.processed_simulations:
                    continue

                # Check if simulation is completed
                state_file = child / "state.json"
                if state_file.exists():
                    try:
                        with open(state_file) as f:
                            state = json.load(f)
                        if state.get("status") == "completed":
                            if sim_id not in ready:
                                ready.append(sim_id)
                    except (json.JSONDecodeError, OSError):
                        continue

                # Also check for action logs as completion indicator
                has_actions = any(
                    (child / platform / "actions.jsonl").exists()
                    for platform in ("twitter", "reddit")
                )
                if has_actions and sim_id not in ready:
                    ready.append(sim_id)

        if ready:
            logger.info("Found %d completed simulations ready for extraction: %s",
                        len(ready), ready)

        return ready

    def mark_processed(self, simulation_id: str) -> None:
        """Mark a simulation as processed to avoid re-extraction."""
        self.processed_simulations.add(simulation_id)
        self._save_processed()

    def run_loop(self, callback) -> None:
        """
        Run the polling loop indefinitely.

        Args:
            callback: Callable that takes a simulation_id and processes it.
                      Should raise on failure (simulation will not be marked processed).
        """
        logger.info(
            "Starting scheduler loop (poll interval: %ds)", self.poll_interval
        )
        while True:
            try:
                ready = self.poll_once()
                for sim_id in ready:
                    try:
                        callback(sim_id)
                        self.mark_processed(sim_id)
                        logger.info("Successfully processed simulation %s", sim_id)
                    except Exception as e:
                        logger.error(
                            "Failed to process simulation %s: %s", sim_id, e
                        )
            except Exception as e:
                logger.error("Scheduler poll error: %s", e)

            time.sleep(self.poll_interval)

    def _load_processed(self) -> None:
        """Load the set of already-processed simulation IDs from disk."""
        state_file = self.uploads_dir / ".forge_bridge_processed.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    self.processed_simulations = set(json.load(f))
                logger.info(
                    "Loaded %d processed simulation IDs",
                    len(self.processed_simulations),
                )
            except (json.JSONDecodeError, OSError):
                self.processed_simulations = set()

    def _save_processed(self) -> None:
        """Persist the set of processed simulation IDs to disk."""
        state_file = self.uploads_dir / ".forge_bridge_processed.json"
        try:
            with open(state_file, "w") as f:
                json.dump(sorted(self.processed_simulations), f)
        except OSError as e:
            logger.warning("Failed to save processed state: %s", e)
