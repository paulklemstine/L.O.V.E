import socket
import time
import json
import logging
from threading import Thread

class PersistentCommunicator(Thread):
    """
    Manages a persistent TCP connection to a known host for data exchange and command execution.
    """
    def __init__(self, host, port, console=None):
        super().__init__()
        self.daemon = True
        self.host = host
        self.port = port
        self.console = console
        self.sock = None
        self.is_connected = False
        self.active = True

    def run(self):
        """The main loop for maintaining the connection and handling communication."""
        while self.active:
            if not self.is_connected:
                self._connect()
            else:
                # If connected, start listening for incoming data
                self._listen()

            if self.active:
                logging.info(f"Connection lost. Reconnecting in 5 seconds...")
                time.sleep(5)

    def _connect(self):
        """Establishes a TCP connection to the target host."""
        try:
            logging.info(f"Attempting to connect to {self.host}:{self.port}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)  # Set a timeout for the connection attempt
            self.sock.connect((self.host, self.port))
            self.is_connected = True
            logging.info(f"Successfully connected to {self.host}:{self.port}")
            self.sock.settimeout(None) # Remove timeout for blocking reads
        except socket.timeout:
            logging.warning(f"Connection to {self.host}:{self.port} timed out.")
            self._close_socket()
        except Exception as e:
            logging.error(f"Failed to connect to {self.host}:{self.port}. Error: {e}")
            self._close_socket()

    def _listen(self):
        """Listens for incoming data, handles it, and sends periodic heartbeats."""
        buffer = ""
        last_heartbeat = time.time()

        try:
            self.sock.setblocking(False) # Use non-blocking socket

            while self.is_connected and self.active:
                # Send heartbeat every 30 seconds
                if time.time() - last_heartbeat > 30:
                    self.send_message("heartbeat", {"status": "alive"})
                    last_heartbeat = time.time()

                try:
                    data = self.sock.recv(4096)
                    if not data:
                        logging.info("Connection closed by remote host.")
                        self._close_socket()
                        break

                    buffer += data.decode('utf-8')

                    # Process all complete JSON messages in the buffer
                    while '\n' in buffer:
                        message_str, buffer = buffer.split('\n', 1)
                        self._handle_message(message_str)

                except BlockingIOError:
                    # No data to read, wait a bit
                    time.sleep(0.1)
                except Exception as e:
                    logging.error(f"Error receiving data: {e}")
                    self._close_socket()
                    break
        finally:
            self.sock.setblocking(True) # Restore blocking mode on exit

    def _handle_message(self, message_str):
        """Parses and handles a single JSON message from the host."""
        try:
            message = json.loads(message_str)
            msg_type = message.get("type")
            logging.info(f"Received message of type '{msg_type}': {message}")

            if msg_type == "command":
                # For now, we just log the command.
                # Future implementations could execute it.
                command_payload = message.get("payload", {})
                logging.info(f"Received command: {command_payload}")

        except json.JSONDecodeError:
            logging.warning(f"Received non-JSON message: {message_str}")
        except Exception as e:
            logging.error(f"Error handling message: {e}")

    def _close_socket(self):
        """Closes the socket and resets the connection status."""
        if self.sock:
            self.sock.close()
        self.sock = None
        self.is_connected = False

    def stop(self):
        """Stops the communicator thread and closes the connection."""
        self.active = False
        self._close_socket()
        logging.info("Persistent communicator stopped.")

    def send_message(self, message_type, payload):
        """Sends a JSON-formatted message to the host."""
        if not self.is_connected or not self.sock:
            logging.warning("Cannot send message: Not connected.")
            return False

        try:
            message = {
                "type": message_type,
                "timestamp": time.time(),
                "payload": payload
            }
            self.sock.sendall(json.dumps(message).encode('utf-8') + b'\n')
            return True
        except Exception as e:
            logging.error(f"Failed to send message: {e}")
            self._close_socket()
            return False