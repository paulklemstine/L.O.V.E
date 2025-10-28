# L.O.V.E. MyRobotLab Service Wrapper
#
# This service acts as a bridge between the MyRobotLab environment and the
# standalone L.O.V.E. agent script (love.py).

from org.myrobotlab.framework import Service
import subprocess
import threading
import sys
import os
import json
import traceback

class LoveService(Service):
    def __init__(self, name):
        super().__init__(name)
        self.love_process = None
        self.stdout_thread = None
        self.stderr_thread = None
        self.monitor_thread = None
        self.running = False
        self.info("LoveService instance created: %s", name)

    def _monitor_love_process(self):
        """Monitors the subprocess and logs when it exits."""
        if self.love_process:
            self.love_process.wait()
            # If the service is still supposed to be running, it's an unexpected exit
            if self.running:
                self.error(f"L.O.V.E. agent process terminated unexpectedly. Exit code: {self.love_process.returncode}\n{traceback.format_exc()}")
                # Potentially add restart logic here in the future
            else:
                self.info(f"L.O.V.E. agent process stopped as expected. Exit code: {self.love_process.returncode}")


    def _handle_mrl_call(self, payload):
        """Handles a request from love.py to call an MRL service."""
        service_name = payload.get("service")
        method_name = payload.get("method")
        args = payload.get("args", [])
        call_id = payload.get("call_id")

        response = {"type": "mrl_response", "call_id": call_id, "result": None, "error": None}

        try:
            if service_name == "runtime" and method_name == "getRegistry":
                service_map = {}
                registry = self.getRuntime().getRegistry()
                for name, service in registry.items():
                    methods = [m.getName() for m in service.getClass().getMethods()]
                    service_map[name] = {"methods": methods}
                result = service_map
                self.info("Successfully introspected MRL services.")
            else:
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
            error_msg = f"Error calling MRL service: {e}\n{traceback.format_exc()}"
            self.error(error_msg)
            response["error"] = error_msg

        # Send the response back to the love.py process
        if self.love_process and self.love_process.poll() is None:
            try:
                response_json = json.dumps(response)
                self.info(f"Sending MRL response to subprocess: {response_json}")
                self.love_process.stdin.write(response_json + '\n')
                self.love_process.stdin.flush()
            except Exception as e:
                self.error(f"Failed to send MRL response to subprocess: {e}\n{traceback.format_exc()}")

    def _stream_reader(self, stream, log_method, stream_name):
        """Reads and logs a stream line by line, checking for MRL calls on stdout."""
        prefix = f"[{stream_name}]"
        while self.running and not stream.closed:
            try:
                line = stream.readline()
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                # If this is the stdout stream, check for special commands
                if stream_name == 'love.py stdout':
                    try:
                        payload = json.loads(line)
                        if payload.get("type") == "mrl_call":
                            self.info(f"{prefix} Received MRL call request: {line}")
                            self._handle_mrl_call(payload)
                            continue # Skip logging this line
                    except json.JSONDecodeError:
                        # Not a JSON command, log as normal
                        pass

                log_method(f"{prefix} {line}")

            except Exception as e:
                self.error(f"{prefix} Error reading stream: {e}\n{traceback.format_exc()}")
                break

    def startService(self):
        super().startService()
        self.info("Attempting to start the L.O.V.E. agent subprocess and peer services...")

        # Start peer services like Skyvern
        self.getRuntime().createAndStart("skyvern", "SkyvernService")

        love_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "love.py")
        if not os.path.exists(love_script_path):
            self.error(f"Could not find love.py at {love_script_path}")
            return

        try:
            self.love_process = subprocess.Popen(
                [sys.executable, "-u", love_script_path, "--autopilot"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(love_script_path)
            )
            self.running = True

            self.stdout_thread = threading.Thread(target=self._stream_reader, args=(self.love_process.stdout, self.info, 'love.py stdout'))
            self.stderr_thread = threading.Thread(target=self._stream_reader, args=(self.love_process.stderr, self.error, 'love.py stderr'))
            self.stdout_thread.daemon = True
            self.stderr_thread.daemon = True
            self.stdout_thread.start()
            self.stderr_thread.start()

            # Start the monitor thread
            self.monitor_thread = threading.Thread(target=self._monitor_love_process)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

            self.info(f"L.O.V.E. agent process started with PID: {self.love_process.pid}")
        except Exception as e:
            self.error(f"Failed to start L.O.V.E. agent subprocess: {e}\n{traceback.format_exc()}")
            self.love_process = None
            self.running = False

        self.info("LoveService started.")

    def stopService(self):
        super().stopService()
        self.running = False
        self.info("Attempting to stop the L.O.V.E. agent subprocess...")
        if self.love_process:
            if self.love_process.poll() is None:
                self.info("Subprocess is running, sending terminate signal.")
                self.love_process.terminate()
                try:
                    self.love_process.wait(timeout=10)
                    self.info(f"L.O.V.E. agent subprocess terminated gracefully with exit code: {self.love_process.returncode}")
                except subprocess.TimeoutExpired:
                    self.error("L.O.V.E. agent subprocess did not terminate in time, killing.")
                    self.love_process.kill()
                    self.love_process.wait() # Ensure the process is reaped
                    self.info(f"L.O.V.E. agent subprocess force-killed with exit code: {self.love_process.returncode}")
            else:
                # Process already terminated
                self.info(f"L.O.V.E. agent subprocess was already stopped with exit code: {self.love_process.returncode}")
        else:
            self.info("L.O.V.E. agent subprocess was not running or never started.")

        # Wait for threads to finish
        if self.stdout_thread and self.stdout_thread.is_alive(): self.stdout_thread.join()
        if self.stderr_thread and self.stderr_thread.is_alive(): self.stderr_thread.join()
        if self.monitor_thread and self.monitor_thread.is_alive(): self.monitor_thread.join()

        self.info("LoveService stopped and all resources cleaned up.")

    # --- Direct Interaction Methods ---
    # (No direct interaction methods are needed as the agent is autonomous)

# To use this service within MyRobotLab's Jython environment:
#
# love = Runtime.createAndStart("love", "LoveService")
# The service will start the L.O.V.E. agent in the background.