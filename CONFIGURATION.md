# Configuration Reference

The application reads its settings from `config.toml` in the current working directory by default. (Use `--config` if you want to specify a different path.) All documented fields are required. Unknown, missing, or invalid values are reported when the configuration is loaded, before the model client or UI is constructed.

## `[model]`

These settings configure requests to Ollama's native `/api/chat` endpoint.

### `name`

The Ollama model name.

```toml
name = "gemma4:e4b"
```

The model must already be available to the Ollama server. For a local server, download it with:

```bash
ollama pull gemma4:e4b
```

### `host`

The base URL of the Ollama server, without `/api/chat`. 

```toml
host = "http://localhost:11434"
```

Use `localhost` when Ollama and the assistant run on the same machine. To use an Ollama server on another machine, provide its reachable hostname or LAN address:

```toml
host = "http://192.168.1.75:11434"
```

A remote server must be configured to listen on a network interface, and its firewall must allow TCP port `11434`. Verify the connection from the same terminal environment in which you will run the assistant:

```bash
curl http://192.168.1.75:11434/api/version
```

### `num_ctx`

The context-window size requested from Ollama, in tokens. 

```toml
num_ctx = 4096
```

Use a positive integer supported by the selected model. Traces report prompt usage relative to this value.

### `num_predict`

The maximum number of tokens Ollama may generate for a model call.

```toml
num_predict = 500
```

During an interactive session, use `Ctrl+P` to override the limit for subsequent calls. The override is session-scoped and does not modify this file.

When Ollama stops a response because it reaches this limit, it reports `done_reason = "length"`. The UI appends an **Output truncated** warning to the response, and the context inspector shows `output limit reached`. Increase `num_predict` if the task regularly requires longer answers, while keeping enough context-window capacity for the prompt.

### `temperature`

Controls sampling variability. Use a number from `0.0` to `2.0`, inclusive.

```toml
temperature = 0.2
```

Lower values generally make responses more consistent; higher values generally make them more varied.

### `seed`

The sampling seed sent to Ollama. 

```toml
seed = 42
```

A fixed seed improves repeatability, although it does not guarantee identical results across model or runtime versions.

### `think`

Controls which Ollama calls use thinking for models that support it.

```toml
think = "off"
```

During an interactive session, use `Ctrl+T` to override thinking for subsequent calls. The override is session-scoped and does not modify this file.

Accepted values are:

- `"off"`: no calls use thinking;
- `"answers"`: only user-facing answer calls use thinking;
- `"maintenance"`: only policy-maintenance calls use thinking;
- `"all"`: both answer and maintenance calls use thinking.

Maintenance calls are policy-dependent. For example, the managed policy makes one to update its retained state, while simpler policies do not make one. Unsupported models may ignore or reject thinking.

When thinking is enabled, the model may generate additional internal reasoning before producing its final answer. **The tokens used for thinking count against the configured `num_predict` limit.** It is advised to increase the limit when thinking is enabled, so that the model has enough room to produce a complete answer after thinking.

### `timeout_seconds`

The maximum time the HTTP client waits for an Ollama request, in seconds. 

```toml
timeout_seconds = 300
```

Use a positive integer long enough for the configured model and hardware. Connection failures and timeouts are reported as an inability to reach the configured Ollama host.

## `[assistant]`

### `system_prompt`

The system instruction placed at the start of every model context. 

```toml
[assistant]
system_prompt = """You are a study assistant..."""
```

It must be a non-empty string. Changing the system prompt can materially change model behavior.

## `[context]`

### `policy`

The context policy used by the interactive assistant. 

```toml
policy = "stateless"
```

Valid values are:

- `stateless`: system prompt plus the current user message;
- `full_history`: the complete stored conversation;
- `recent_window`: the system prompt, recent completed turns, and current user message;
- `managed`: recent completed turns plus retained conversational state.

The application's `--policy` option overrides this setting for one run. The scenario runner requires `--policy` explicitly.

### `recent_turns`

The configured size of the recent-turn window available to context policies.

```toml
recent_turns = 3
```

The value is passed to each context policy as `self.recent_turns`; it does not change the model context on its own. The `recent_window` policy must use it as the maximum number of completed user/assistant turns retained before the current user message. A managed policy may also use it when constructing its context.

## `[trace]`

### `directory`

The directory used for interactive `.jsonl` conversation traces.

```toml
directory = "logs"
```

Relative paths are resolved from the process's current working directory, not from the location of the configuration file. Parent directories are created automatically.

Evaluation output paths are selected independently with the scenario runner's `--output` option.
