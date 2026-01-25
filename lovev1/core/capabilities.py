import platform
import shutil
import os

class PlatformCaps:
    """A simple class to hold detected platform capabilities."""
    def __init__(self):
        self.os = platform.system()
        self.arch = platform.machine()
        self.is_termux = 'TERMUX_VERSION' in os.environ
        self.has_cuda = False
        self.has_metal = False
        self.gpu_type = "none"

    def __str__(self):
        return f"OS: {self.os}, Arch: {self.arch}, GPU: {self.gpu_type}, Termux: {self.is_termux}"

# Instantiate the capabilities class globally so it can be accessed by other functions.
CAPS = PlatformCaps()