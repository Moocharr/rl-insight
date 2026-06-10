# Docker Compose Development Startup

`rl-insight server start` uses locally installed Prometheus, Tempo, and Grafana binaries by default.

This directory keeps the previous Docker Compose stack as a development-only helper. Use it only when you explicitly want RL-Insight to start the server stack through Docker.

From this directory:

```bash
docker compose up -d
```

Stop it with:

```bash
docker compose down
```

The compose file uses the default local ports: Prometheus `9090`, Tempo `3200`, OTLP HTTP `4318`, and Grafana `3000`.
