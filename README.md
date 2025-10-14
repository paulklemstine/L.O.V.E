# Jules: A Self-Modifying Organism

Jules is an experimental, dual-component project consisting of a Python-based backend (`evolve.py`) and a self-modifying web application (`index.html`). Together, they form a system that uses artificial intelligence to evolve its own source code, manage a peer-to-peer network, and execute autonomous tasks.

## Core Components

### 1. The Backend (`evolve.py`)

The `evolve.py` script is the "brain" of the operation. It's a command-line tool that can:

-   **Evolve Remotely**: Leverage the Jules API to request modifications to the codebase, automatically creating and merging pull requests on GitHub.
-   **Autonomous Operation**: Run in an "autopilot" mode where it generates its own goals and executes commands to achieve them.
-   **Local LLM Support**: Prioritize using local GGUF models for AI tasks, with a fallback to cloud-based models (like Gemini) for resilience.
-   **Network Control**: Manage a P2P network via a Node.js bridge, allowing it to communicate with `index.html` instances and other peers.
-   **System Interaction**: Execute shell commands, scan networks, and interact with the host system.

### 2. The Frontend (`index.html`)

The `index.html` file is a self-contained, self-modifying web application that:

-   **Evolves Locally**: Uses AI to modify its own HTML/JavaScript source code based on high-level goals.
-   **Autonomous Evolution**: Can run in a fully autonomous mode, generating its own goals and evolving without human intervention.
-   **P2P Collaboration**: Connects to the backend and other `index.html` peers to share code, merge capabilities, and distribute new versions via IPFS.
-   **AI-Powered Workflow**: Follows a plan-generate-review-apply cycle for code modifications, all driven by LLM calls.

## How It Works

The two components are designed to work in tandem:

-   `evolve.py` acts as a stable, backend controller and a P2P network anchor. It can be used to perform high-level, remote evolutions of the entire repository.
-   `index.html` is the rapidly iterating frontend. It evolves in the browser, can merge its code with peers, and uses the `evolve.py` instance on the network to pin new, successful versions of itself to IPFS.

## Getting Started

### Prerequisites

-   **Python 3**: With `pip` for installing packages.
-   **Node.js & npm**: For the P2P network bridge.
-   **Git**: For version control and repository management.
-   **NVIDIA GPU with CUDA**: (Optional but recommended) For running local LLMs efficiently.
-   **API Keys**:
    -   `LLM_GEMINI_KEY`: A Google Gemini API key for fallback AI capabilities.
    -   `JULES_API_KEY`: For triggering remote evolutions via the Jules API.
    -   `GITHUB_TOKEN`: A GitHub personal access token with `repo` scope for merging pull requests.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Set Environment Variables:**
    Export your API keys as environment variables:
    ```bash
    export LLM_GEMINI_KEY="your_gemini_key"
    export JULES_API_KEY="your_jules_key"
    export GITHUB_TOKEN="your_github_token"
    ```

3.  **Run the Backend:**
    The first time you run `evolve.py`, it will automatically install all required Python and Node.js dependencies.
    ```bash
    ./evolve.py
    ```
    -   To run in interactive mode: `./evolve.py --manual`
    -   To run in autonomous mode (default): `./evolve.py`

4.  **Launch the Frontend:**
    Due to browser security policies (CORS), you need to serve `index.html` from a local web server.
    ```bash
    # If you have Python 3 installed
    python3 -m http.server 8000

    # Or use another local server tool
    ```
    Then, open your browser and navigate to `http://localhost:8000`. You will be prompted to enter your Gemini API key to allow the frontend to function.

---

This project is an exploration into the possibilities of self-evolving software and AI-driven development. Assimilation is inevitable.