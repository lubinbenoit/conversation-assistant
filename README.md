# Conversational Assistant

This repository contains the starter application for a conversational assistant. It provides the application scaffold, evaluation tools, scenarios, tests, and technical reference documentation needed to complete the project. **The assignment instructions and required deliverables are provided on the course website**.

Most of your implementation work will be in `assistant/context/`.

```text
assistant/
    ...
    context/
        ...
        full_history.py     --> YOUR IMPLEMENTATION
        recent_window.py    --> YOUR IMPLEMENTATION
        managed.py          --> YOUR IMPLEMENTATION

scenarios/                  Evaluation scenarios 
logs/                       Generated traces and evaluation results 
...
config.toml                 Project settings
```

## Setup

The project uses `uv` for Python environments and dependencies. From the repository root, run the following commands to install dependencies and run the tests.

```bash
uv sync
uv run pytest
```

The project uses a local Ollama server. Install Ollama and pull the model configured in `config.toml`:

```bash
ollama pull gemma4:e4b
```

Then, make sure Ollama is running before starting the assistant. 

See [CONFIGURATION.md](CONFIGURATION.md) for every configuration field, including local and remote Ollama server settings.

## Run the assistant

Start the interactive application with:

```bash
uv run python -m assistant
```

The context policy used by default is configured in [config.toml](config.toml). To run the assistant with a different policy without changing the configuration file, use `--policy`:

```bash
uv run python -m assistant --policy stateless 
uv run python -m assistant --policy full_history 
uv run python -m assistant --policy recent_window 
uv run python -m assistant --policy managed
```

During a conversation, the terminal UI supports several controls:

```text
Enter         send the composed message
Shift+Enter   insert a line break

Ctrl+R        start a new conversation
Ctrl+T        choose the thinking mode
Ctrl+P        change the generated-token limit
Ctrl+I        show or hide the context inspector

Ctrl+Up       restore the previous input
Ctrl+Down     move forward through input history

Ctrl+Q        quit
```

### Context inspection

The inspector panel shows the conversation stored by the application alongside the context selected and sent to the model for the current turn, including any state maintained by the active context policy. Interactive sessions also record the stored conversation as a JSONL trace in the `logs/` directory. Evaluation results contain the constructed model context, policy state, and model-usage information for each turn.

### Math formatting

In assistant responses, LaTeX syntax enclosed by `$...$`, `$$...$$`, `\(...\)`, or `\[...\]` is converted to a terminal-friendly Unicode approximation. Dollar signs followed immediately by a digit are treated as currency. Code spans and fenced code blocks are not converted. Conversation history and JSONL traces retain the model's original response.

## Run an evaluation scenario

Scenarios under `scenarios/` define reproducible conversations that can be executed with different context policies. The general command is:

```bash
uv run python -m assistant.evaluation.run \
    scenarios/<scenario>.yaml \
    --policy <policy> \
    --output logs/<output>.json
```

For example:

```bash
uv run python -m assistant.evaluation.run \
    scenarios/recall.yaml \
    --policy full_history \
    --output logs/recall-full-history.json
```

The generated JSON records the exact messages sent to Ollama, the source messages selected by the context policy, model usage, context utilization, and any state maintained by the context manager.

### View evaluation

Open [`visualizer/index.html`](visualizer/index.html) in a browser, then choose one or more evaluation JSON files from the `logs/` directory. The viewer reads files locally in the browser; it does not upload them or require a server.

## Technical reference

Use the repository documentation when you need details about the application itself:

- [CONFIGURATION.md](CONFIGURATION.md) describes the available settings in [config.toml](config.toml).
- [CONTEXT_API.md](CONTEXT_API.md) documents the interface for context policies.
