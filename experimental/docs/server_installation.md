# Server Installation

RL-Insight server depends on Prometheus, Tempo, and Grafana. The recommended
path is to let RL-Insight download known-good Linux binaries into a user-managed
directory:

```bash
rl-insight server install
rl-insight server start
```

`server install` does not use Docker. It downloads release archives, extracts
the binaries under `~/.rl-insight/services`, and records the managed paths for
later `server start` runs.

## Versions

| Service | Required version | Installer version |
|---|---:|---:|
| Prometheus | `>= 2.30.0` | `2.54.1` |
| Tempo | `>= 2.0.0` | `2.6.1` |
| Grafana | `>= 9.0.0` | `11.6.3` |

System packages are fine when they satisfy the required version. If your distro
package is older, use `rl-insight server install`.

These requirements are intentionally low. The current server stack only uses
basic Prometheus scraping/reload, Tempo OTLP HTTP with local storage, and Grafana
datasource/dashboard provisioning.

## Linux Support

| Area | Supported values |
|---|---|
| OS family | Ubuntu/Debian, CentOS/RHEL/Rocky/Alma |
| CPU architecture | `amd64`/`x86_64`, `arm64`/`aarch64` |

Automatic installation is Linux-only. Windows and macOS can still run the
training-side Python APIs, but RL-Insight does not manage local server services
there yet.

## Recommended Install

From the Python environment where `rl-insight` is installed:

```bash
rl-insight server install
```

Use a custom managed directory only when needed:

```bash
rl-insight server install --install-dir /opt/rl-insight/services
```

Then start the local services:

```bash
rl-insight server start
```

Foreground mode stops Prometheus, Tempo, and Grafana when you press `Ctrl+C`.
Detached mode can be stopped explicitly:

```bash
rl-insight server start --detach
rl-insight server stop
```

## Data Persistence

RL-Insight keeps collected server data on disk. By default, data is stored under
`~/.rl-insight/data`:

| Service | Persistent data |
|---|---|
| Prometheus | `~/.rl-insight/data/prometheus` TSDB blocks |
| Tempo | `~/.rl-insight/data/tempo/traces` and `~/.rl-insight/data/tempo/wal` |
| Grafana | `~/.rl-insight/data/grafana` data, logs, and plugins |

`Ctrl+C` and `rl-insight server stop` stop processes only. They do not delete the
data directory, so previous RL training metrics and traces remain available on
the next start.

Prometheus and Tempo retain data for `30d` by default:

```yaml
prometheus:
  retention_time: 30d
tempo:
  retention_time: 30d
```

Use a custom persistent directory when needed:

```yaml
server:
  data_dir: /path/to/rl-insight/data
```

## Package Managers

Package manager installs are useful in company-managed environments. After
installing, check versions:

```bash
prometheus --version
tempo --version
grafana-server --version
```

On Ubuntu/Debian, Prometheus may be available from the distro repository:

```bash
sudo apt-get update
sudo apt-get install -y prometheus
```

Grafana is usually installed from Grafana's official APT/RPM repositories. Tempo
is commonly installed from its release archive or package artifact. Keep these
installs only if their versions meet the table above.

## Manual Binary Fallback

Use this only when the RL-Insight installer cannot reach the release endpoints.

```bash
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ARCH=amd64 ;;
  aarch64|arm64) ARCH=arm64 ;;
  *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac
```

Prometheus:

```bash
PROMETHEUS_VERSION=2.54.1
curl -fL -O "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-${ARCH}.tar.gz"
tar -xzf "prometheus-${PROMETHEUS_VERSION}.linux-${ARCH}.tar.gz"
sudo install -m 0755 "prometheus-${PROMETHEUS_VERSION}.linux-${ARCH}/prometheus" /usr/local/bin/prometheus
prometheus --version
```

Tempo:

```bash
TEMPO_VERSION=2.6.1
curl -fL -O "https://github.com/grafana/tempo/releases/download/v${TEMPO_VERSION}/tempo_${TEMPO_VERSION}_linux_${ARCH}.tar.gz"
tar -xzf "tempo_${TEMPO_VERSION}_linux_${ARCH}.tar.gz"
sudo install -m 0755 tempo /usr/local/bin/tempo
tempo --version
```

Grafana:

```bash
GRAFANA_VERSION=11.6.3
curl -fL -O "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-${ARCH}.tar.gz"
tar -xzf "grafana-${GRAFANA_VERSION}.linux-${ARCH}.tar.gz"
sudo mkdir -p /opt/grafana
sudo cp -a "grafana-v${GRAFANA_VERSION}/." /opt/grafana/
sudo ln -sf /opt/grafana/bin/grafana-server /usr/local/bin/grafana-server
grafana-server --version
```

If your network cannot access GitHub release assets or `dl.grafana.com`, mirror
the three archives internally and run the same extract/install steps from that
mirror.
