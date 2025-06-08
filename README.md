# Optimal Pathfinding with Q-Learning

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

*An agent trained with Q-function approximation to find the most efficient path in a custom grid-world environment.*

---

### Project Architecture

<p align="center">
  <img src="https://github.com/user-attachments/assets/bb5d2938-e16f-415e-9145-ad8a6e71e22f" alt="Diagram showing the agent-environment interaction loop and Q-learning update rule.">
</p>

---

## 🚀 About The Project

This project demonstrates the application of Reinforcement Learning to solve a classic navigation and collection problem. An agent is trained to operate efficiently within a custom-built environment, learning from its own experiences to achieve its goal.

### The Core Challenge

The main task is set in a simulated storage room:

*   **Environment:** A grid-world representing a storage room, populated with static obstacles (blockages) and collectible items (boxes).
*   **Agent:** A robotic agent that can navigate the room.
*   **Objective:** The agent must learn an **optimal policy** to collect all the boxes in the fewest steps possible, effectively finding the most efficient path while avoiding all obstacles.

### Technical Approach

To solve this, the agent is trained using **Q-learning**, a model-free reinforcement learning algorithm. Specifically, **Q-function approximation** is employed to handle the potentially large state space of the environment, allowing the agent to generalize from situations it has seen to new, unseen ones.

---

### Prerequisites

You will need Python 3.12+ and the following libraries installed. The `requirements.txt` file handles this automatically.

*   **Python 3.12**
*   **NumPy**
*   **Pygame** (for visualization)
*   **Matplotlib**

### Installation

1.  Clone the repository to your local machine:
    ```sh
    git clone https://github.com/dupo2/MDP_LD3.git
    ```
2.  Navigate into the cloned repository:
    ```sh
    cd MDP_LD3
    ```
3.  Create a Python virtual environment:
    ```sh
    python -m venv venv
    ```
4.  Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

---

## 💨 How to Launch

1.  **Activate the virtual environment.** The command differs based on your operating system.

    *   **On Windows (Command Prompt / PowerShell):**
        ```powershell
        venv\Scripts\activate
        ```
    *   **On macOS / Linux (Bash / Zsh):**
        ```sh
        source venv/bin/activate
        ```

2.  **Run the main script** to start the simulation:
    ```sh
    python main.py
    ```

---
