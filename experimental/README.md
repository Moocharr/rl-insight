# RL-Insight Monitor

RL-Insight Monitor provides a server stack for RL training metrics and traces based on Prometheus, Tempo, and Grafana.

It has two parts:

- `rl-insight server ...`: install and manage local server services.
- `rl_insight`: training-side Python APIs for metrics and traces.

## Quickstart

### 1. Install

From the repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Install server services

RL-Insight currently manages Prometheus, Tempo, and Grafana on Linux. The simplest path is RL-Insight's user-managed installer:

```bash
rl-insight server install
```

See the concise service installation guide for Linux support, CPU architecture, and version requirements: [`docs/server_installation.md`](docs/server_installation.md).

### 3. Start the server stack

Default foreground mode:

```bash
rl-insight server start
```

This mode starts local Prometheus, Tempo, and Grafana processes, keeps the CLI attached, and stops the whole stack when you press `Ctrl+C`.

Prometheus, Tempo, and Grafana data is persisted by default under `~/.rl-insight/data`. Prometheus and Tempo retain data for `30d` by default. Stopping the server does not delete this directory.

Grafana will be provisioned automatically with Prometheus and Tempo datasources plus an empty starter dashboard. The datasources follow the configured Prometheus and Tempo published ports.

Background mode:

```bash
rl-insight server start --detach
```

Foreground mode with service logs attached:

```bash
rl-insight server start --attach-logs
```

Use a custom config file:

```bash
rl-insight server start --config path/to/config.yaml
```

Stop the stack explicitly from another terminal:

```bash
rl-insight server stop
```

After startup, the CLI prints the trainer OTLP traces URL and the Prometheus,
Tempo, and Grafana access URLs.

If a service binary is missing, `server start` prints a dependency table and tells you to run:

```bash
rl-insight server install
```

### 4. Initialize the training side

```python
import os
import ray
import rl_insight as insight

os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://<server-ip>:4318/v1/traces"

ray.init(address="auto", namespace="rl-insight-monitor")
insight.init()
```

Notes:

- `ray.init(namespace="rl-insight-monitor")` is used to find the monitor hub actor.
- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` takes precedence over `insight.init(config)` -> `otel.traces_endpoint`.

### 5. Emit metrics and traces

```python
import rl_insight as insight

insight.metric_count("train_step_total", amount=1, worker="trainer_0")
insight.metric_value("reward_mean", value=1.23, worker="trainer_0")
insight.metric_distribution("step_latency_ms", value=42.5, worker="trainer_0")

with insight.trace_state("rollout", state_lane_id="trainer_0", step=10):
    run_rollout()

@insight.trace_op("update_model", stage="optimizer")
def update_model(batch):
    ...
```

## APIs

| API | Purpose |
|---|---|
| `init(config=None)` | Initialize training-side monitoring |
| `close()` | Reset monitor state in the current process |
| `metric_count()` | Report a counter |
| `metric_value()` | Report a gauge |
| `metric_distribution()` | Report a histogram |
| `trace_state()` | Report a state interval |
| `trace_op()` | Decorator for operation latency traces |

## CLI Reference

### `rl-insight server install`

| Argument | Default | Description |
|---|---:|---|
| `--install-dir` | `~/.rl-insight/services` | User-managed directory for downloaded Linux binaries |
| `--force` | `false` | Reinstall enabled services even when binaries already exist |
| `--config` | `experimental/config/services/config.yaml` | Server config file path |
| `--log-level` | `INFO` | Python log level |

### `rl-insight server start`

| Argument | Default | Description |
|---|---:|---|
| `--detach` | `false` | Start in background and return immediately |
| `--attach-logs` | `false` | Run in foreground and stream service logs |
| `--config` | `experimental/config/services/config.yaml` | Server config file path |
| `--log-level` | `INFO` | Python log level |

### `rl-insight server stop`

| Argument | Default | Description |
|---|---:|---|
| `--config` | `experimental/config/services/config.yaml` | Server config file path |
| `--log-level` | `INFO` | Python log level |

## Server YAML

| Key | Default | Description |
|---|---:|---|
| `server.backend` | `local` | Stack startup backend |
| `server.install_dir` | unset | Optional override for RL-Insight's managed fallback directory |
| `server.runtime_dir` | `~/.rl-insight/runtime` | Optional directory for rendered Tempo/Grafana runtime configs |
| `server.data_dir` | `~/.rl-insight/data` | Optional directory for Prometheus, Tempo, and Grafana data |
| `server.state_file` | `~/.rl-insight/run/rl-insight-services.json` | Optional PID state file path used by `server stop` |
| `prometheus.min_version` | `2.30.0` | Minimum supported Prometheus version |
| `prometheus.install_version` | `2.54.1` | Prometheus version downloaded by `server install` |
| `prometheus.prometheus_port` | `9090` | Prometheus HTTP port |
| `prometheus.retention_time` | `30d` | Prometheus TSDB retention time |
| `prometheus.config_file` | `prometheus.yml` | Prometheus config file |
| `tempo.min_version` | `2.0.0` | Minimum supported Tempo version |
| `tempo.install_version` | `2.6.1` | Tempo version downloaded by `server install` |
| `tempo.query_port` | `3200` | Tempo query port |
| `tempo.retention_time` | `30d` | Tempo trace block retention time |
| `otel.traces_endpoint` | `http://127.0.0.1:4318/v1/traces` | Trainer trace export endpoint |
| `grafana.min_version` | `9.0.0` | Minimum supported Grafana version |
| `grafana.install_version` | `11.6.3` | Grafana version downloaded by `server install` |
| `grafana.port` | `3000` | Grafana HTTP port |
| `grafana.provisioning_dir` | `provisioning` | Source provisioning directory |
| `grafana.dashboards_dir` | `dashboards` | Dashboard JSON directory |

## Docker Compose for Development

The old Docker Compose stack is kept only as a development helper under `experimental/config/services/docker-compose-dev`.

Use it explicitly from that directory:

```bash
docker compose up -d
docker compose down
```

The default `rl-insight server start` path does not call Docker Compose.

## `insight.init(config)`

| Key | Default | Description |
|---|---:|---|
| `namespace` | `rl_insight_monitor` | Metrics and trace namespace |
| `backend.type` | `ray` | Currently only `ray` is supported |
| `prometheus.metrics_report_port` | `9092` | Monitor hub `/metrics` port |
| `prometheus.prometheus_port` | `9090` | Prometheus HTTP port used for reload |
| `prometheus.config_file` | bundled absolute path | Prometheus config file to rewrite |
| `prometheus.reload.mode` | `ray` | `ray` or `none` |
| `otel.traces_endpoint` | `http://127.0.0.1:4318/v1/traces` | Trainer trace export endpoint |
