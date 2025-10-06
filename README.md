# Jules - A Self-Modifying Web Application

Jules is an experimental, self-modifying web application that uses artificial intelligence to evolve its own source code. It's a single-page application that operates autonomously or with human guidance to continuously improve its features and functionality.

## Key Features

*   **Self-Modification**: Jules can write, review, and apply new code to its own HTML source, allowing it to evolve over time.
*   **AI-Driven Development**: It uses Google's Gemini models to generate plans, write code, and perform code reviews based on high-level goals.
*   **P2P Code Sharing & Merging**: Jules instances can connect to each other in a peer-to-peer network to share and merge their source code, enabling collaborative evolution.
*   **Embedded Linux Environment**: It features an integrated WebVM, providing a full-fledged Linux terminal within the browser for executing commands and managing files.
*   **Autopilot Mode**: Jules can operate in a fully autonomous "Autopilot" mode, where it generates its own goals and iterates on its code without human intervention.
*   **Resilient & Decentralized**: The application includes a fallback mechanism for API calls and a decentralized P2P network, making it robust and resilient.

## How It Works

The application's workflow is centered around a four-step process:

1.  **Generate Plan**: Based on a user-defined goal or an autonomously generated one, Jules creates a step-by-step plan for modifying its code.
2.  **Generate Code**: Following the plan, it generates new HTML code for the entire application.
3.  **Review Code**: The new code is then reviewed by another AI instance to check for correctness and quality.
4.  **Apply & Evolve**: Once the code passes the review, Jules applies the new code to itself by overwriting the current page's DOM, effectively "evolving" into a new version.

## Getting Started

1.  **Open `index.html`**: Open the `index.html` file in a modern web browser.
2.  **Enter API Key**: You will be prompted to enter a Google Gemini API key. This is required for the AI functionalities to work.
3.  **Set a Goal**: In the "Goal" panel, describe a change you want to make to the application.
4.  **Follow the Workflow**: Use the control buttons (Generate Plan, Generate Code, etc.) to step through the modification process.
5.  **Alternatively, use Autopilot**: Click "Start Autopilot" to let Jules take over and evolve on its own.

---

This project is an exploration into the possibilities of self-evolving software and AI-driven development. Assimilation is inevitable.