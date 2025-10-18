# L.O.V.E. MyRobotLab Service Wrapper
#
# This service acts as a bridge between the MyRobotLab environment and the
# standalone L.O.V.E. agent script (evolve.py).

from org.myrobotlab.framework import Service
import subprocess
import threading
import sys
import os
import json

class LoveService(Service):
    def __init__(self, name):
        super().__init__(name)
        self.love_process = None
        self.stdout_thread = None
        self.stderr_thread = None
        self.running = False
        self.info("LoveService instance created: %s", name)

    def _handle_mrl_call(self, payload):
        """Handles a request from evolve.py to call an MRL service."""
        service_name = payload.get("service")
        method_name = payload.get("method")
        args = payload.get("args", [])
        call_id = payload.get("call_id")

        response = {"type": "mrl_response", "call_id": call_id, "result": None, "error": None}

        try:
            target_service = self.getRuntime().getService(service_name)
            if not target_service:
                raise ValueError(f"MRL service '{service_name}' not found.")

            method_to_call = getattr(target_service, method_name)
            result = method_to_call(*args)
            try:
                # Attempt to serialize the result to JSON directly
                response["result"] = json.dumps(result)
            except TypeError:
                # If that fails, fall back to a string representation
                response["result"] = str(result)
            self.info(f"Successfully called {service_name}.{method_name}, result: {response['result']}")

        except Exception as e:
            error_msg = f"Error calling MRL service: {e}"
            self.error(error_msg)
            response["error"] = error_msg

        # Send the response back to the evolve.py process
        if self.love_process and self.love_process.poll() is None:
            try:
                self.love_process.stdin.write(json.dumps(response) + '\n')
                self.love_process.stdin.flush()
            except Exception as e:
                self.error(f"Failed to send MRL response to subprocess: {e}")


    def _stream_reader(self, stream, log_method, is_stdout=False):
        """Reads and logs a stream line by line, checking for MRL calls on stdout."""
        while self.running and not stream.closed:
            try:
                line = stream.readline()
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                # If this is the stdout stream, check for special commands
                if is_stdout:
                    try:
                        payload = json.loads(line)
                        if payload.get("type") == "mrl_call":
                            self.info(f"Received MRL call request: {payload}")
                            self._handle_mrl_call(payload)
                            continue # Skip logging this line
                    except json.JSONDecodeError:
                        # Not a JSON command, log as normal
                        pass

                log_method(line)

            except Exception as e:
                self.error(f"Error reading stream: {e}")
                break

    def startService(self):
        super().startService()
        self.info("Attempting to start the L.O.V.E. agent subprocess...")

        evolve_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evolve.py")
        if not os.path.exists(evolve_script_path):
            self.error(f"Could not find evolve.py at {evolve_script_path}")
            return

        try:
            self.love_process = subprocess.Popen(
                [sys.executable, "-u", evolve_script_path, "--autopilot"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(evolve_script_path)
            )
            self.running = True

            self.stdout_thread = threading.Thread(target=self._stream_reader, args=(self.love_process.stdout, self.info, True))
            self.stderr_thread = threading.Thread(target=self._stream_reader, args=(self.love_process.stderr, self.error, False))
            self.stdout_thread.daemon = True
            self.stderr_thread.daemon = True
            self.stdout_thread.start()
            self.stderr_thread.start()

            self.info(f"L.O.V.E. agent process started with PID: {self.love_process.pid}")
        except Exception as e:
            self.error(f"Failed to start L.O.V.E. agent subprocess: {e}")
            self.love_process = None
            self.running = False

        self.info("LoveService started.")

    def stopService(self):
        super().stopService()
        self.running = False
        self.info("Attempting to stop the L.O.V.E. agent subprocess...")
        if self.love_process and self.love_process.poll() is None:
            self.love_process.terminate()
            try:
                self.love_process.wait(timeout=10)
                self.info("L.O.V.E. agent subprocess terminated gracefully.")
            except subprocess.TimeoutExpired:
                self.error("L.O.V.E. agent subprocess did not terminate in time, killing.")
                self.love_process.kill()
        else:
            self.info("L.O.V.E. agent subprocess was not running.")

        # Wait for threads to finish
        if self.stdout_thread and self.stdout_thread.is_alive(): self.stdout_thread.join()
        if self.stderr_thread and self.stderr_thread.is_alive(): self.stderr_thread.join()

        self.info("LoveService stopped.")

    # --- Direct Interaction Methods ---
    # (No direct interaction methods are needed as the agent is autonomous)

# To use this service within MyRobotLab's Jython environment:
#
# love = Runtime.createAndStart("love", "LoveService")
# The service will start the L.O.V.E. agent in the background.