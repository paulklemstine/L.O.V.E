# Skyvern MyRobotLab Service
#
# This service manages the Skyvern web automation tool, which runs in Docker.
# It provides a simple interface for the L.O.V.E. agent to run web automation tasks.

from org.myrobotlab.framework import Service
import subprocess
import os
import requests
import json

class SkyvernService(Service):
    def __init__(self, name):
        super().__init__(name)
        self.info("SkyvernService instance created: %s", name)
        self.is_docker_running = False
        self.skyvern_running = False
        self.skyvern_api_url = "http://localhost:8000/api/v1/tasks"

    def _check_docker(self):
        """Checks if the Docker daemon is running."""
        self.info("Checking for Docker daemon...")
        try:
            subprocess.check_output(["docker", "info"], stderr=subprocess.STDOUT)
            self.info("Docker daemon is running.")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.error("Docker daemon is not running or Docker is not installed. Skyvern service will be disabled.")
            return False

    def startService(self):
        """Starts the Skyvern Docker containers if Docker is available."""
        super().startService()
        self.is_docker_running = self._check_docker()
        if not self.is_docker_running:
            return

        if not os.path.exists(".env"):
            self.error("A .env file with LLM provider keys is required to run Skyvern. Please create one. Skyvern service will be disabled.")
            return

        self.info("Starting Skyvern services via docker-compose...")
        try:
            subprocess.check_call(["docker", "compose", "pull"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            subprocess.check_call(["docker", "compose", "up", "-d"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.skyvern_running = True
            self.info("Skyvern services started successfully.")
        except subprocess.CalledProcessError as e:
            self.error(f"Failed to start Skyvern services: {e.stderr.decode()}")
            self.skyvern_running = False

    def stopService(self):
        """Stops the Skyvern Docker containers."""
        super().stopService()
        if not self.skyvern_running:
            return

        self.info("Stopping Skyvern services via docker-compose...")
        try:
            subprocess.check_call(["docker", "compose", "down"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.info("Skyvern services stopped successfully.")
        except subprocess.CalledProcessError as e:
            self.error(f"Failed to stop Skyvern services: {e.stderr.decode()}")
        finally:
            self.skyvern_running = False

    def run_task(self, prompt: str):
        """
        Runs a web automation task by making a direct HTTP request to the Skyvern API.
        """
        if not self.skyvern_running:
            return "Skyvern service is not running. Cannot execute task."

        self.info(f"Received task from L.O.V.E.: '{prompt}'")

        # The first part of the prompt is the url, the rest is the task
        parts = prompt.split(" ", 1)
        if len(parts) < 2:
            return "ERROR: The prompt must contain a URL and a task, separated by a space."

        url = parts[0]
        task_prompt = parts[1]

        headers = {
            "Content-Type": "application/json",
        }

        # The data payload for the Skyvern API
        data = {
            "url": url,
            "prompt": task_prompt,
        }

        try:
            response = requests.post(self.skyvern_api_url, headers=headers, data=json.dumps(data), timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.error(f"An error occurred while running the Skyvern task: {e}")
            return f"ERROR: An error occurred in Skyvern: {e}"

# To use this service within MyRobotLab's Jython environment:
#
# skyvern = Runtime.createAndStart("skyvern", "SkyvernService")
# result = skyvern.run_task("https://www.google.com search for 'lorem ipsum'")
# print(result)