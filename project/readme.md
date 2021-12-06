## Agents

The [`Agent`](agent.py#L10) class from [`agent.py`](agent.py) contains the algorithm implementation.

## Evaluate your agents locally

To run the evaluation locally, run the following commands.

```bash
ENV_NAME="acrobot" python run.py
ENV_NAME="taxi" python run.py
ENV_NAME="kbca" python run.py
ENV_NAME="kbcb" python run.py
ENV_NAME="kbcc" python run.py
```

## Repository structure

**File/Directory** | **Description**
--- | ---
[`agent.py`](agent.py) | File for implementing the Agent class. Your code goes in this file.
[`config.py`](config.py) | File containing the configuration options for  Agent class.
[`run.py`](run.py) | File used to evaluate the agent class. Use this file to test your agents locally
[`requirements.txt`](requirements.txt) | File containing the list of python packages you want to install for the submission to run. Refer [runtime configuration](#runtime-configuration) for more information.
[`apt.txt`](apt.txt) | File containing the list of packages you want to install for submission to run. Refer [runtime configuration](#runtime-configuration) for more information.
[`gym-bellman`](gym-bellman/) | Folder containing the gym environment for the Bellman's DP problem
[`docs`](docs/) | Folder containing the descriptions for the environments in the challenge
[`aicrowd.json`](docs/) | Submission configuration
