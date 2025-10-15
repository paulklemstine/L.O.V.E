# J.U.L.E.S. on Android (via Termux)

This document provides instructions on how to run the J.U.L.E.S. (`evolve.py`) application on an Android device using the [Termux](https://termux.dev/en/) terminal emulator.

**Disclaimer:** Running local AI models is computationally intensive. Performance will be heavily dependent on your device's CPU. Expect slower processing times compared to a desktop computer with a dedicated GPU.

---

## Prerequisites

1.  **Termux App**: You must install Termux on your Android device. It is **strongly recommended** to install it from [F-Droid](https://f-droid.org/en/packages/com.termux/). The version on the Google Play Store is outdated and no longer maintained.
2.  **A Hacker's Mindset**: This is not a standard Android app. You will be working in a Linux-like command-line environment.

---

## Installation

The setup process is automated by a script that installs the necessary system-level tools and then lets the `evolve.py` application handle its own dependencies in a platform-aware way.

### Step 1: Grant Storage Permission

First, you need to give Termux permission to access your device's storage.

```bash
termux-setup-storage
```
A popup will appear asking for permission. Please allow it.

### Step 2: Clone the Repository

Navigate to a directory where you want to store the project. We'll use the `Documents` folder as an example.

```bash
cd ~/storage/shared/Documents
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
*(Replace `your-username/your-repo-name` with the actual repository URL)*

### Step 3: Run the Universal Setup Script

The project includes a setup script that will automatically install all the necessary packages and dependencies. This is the most time-consuming step, as it involves compiling code directly on your device.

**Ensure your device is connected to Wi-Fi and has plenty of battery before proceeding.**

Navigate to the `android` directory and execute the script:

```bash
cd android
bash setup.sh
```

The script will:
1.  Update the Termux package manager (`pkg`).
2.  Install essential system dependencies like `python`, `nodejs`, `git`, and `clang`.
3.  Install the required Node.js modules via `npm`.
4.  **Delegate to `evolve.py`**, which will intelligently detect your device's architecture and compile the `llama-cpp-python` library for optimal CPU performance.

This process can take anywhere from 15 minutes to over an hour depending on your device. Please be patient.

---

## Running J.U.L.E.S.

Once the setup script has completed successfully, you can start the application at any time using the launcher script.

From the `android` directory, run:

```bash
bash run.sh
```

This will launch `evolve.py` in the terminal. The script will detect your hardware and load the local AI model into your device's memory before beginning its autonomous cognitive loop.

To stop the application, press `Ctrl+C`.