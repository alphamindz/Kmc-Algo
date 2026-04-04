#!/bin/bash
set -e

MODE="${ASTRUM_MODE:-all}"

case "$MODE" in
  server)
    echo "Starting Astrum API server on port 7860..."
    exec python -m uvicorn hypernoa.astrum_env.server:app --host 0.0.0.0 --port 7860
    ;;
  demo)
    echo "Starting Gradio demo on port 7860..."
    cd /app/hf_space
    exec python app.py
    ;;
  all)
    echo "Starting Astrum API server on port 7860..."
    python -m uvicorn hypernoa.astrum_env.server:app --host 0.0.0.0 --port 7860 &
    API_PID=$!

    echo "Starting Gradio demo on port 7861..."
    GRADIO_SERVER_PORT=7861 python /app/hf_space/app.py &
    DEMO_PID=$!

    echo "Both services running: API=:7860  Demo=:7861"
    wait $API_PID $DEMO_PID
    ;;
  train)
    echo "Running training episode comparison..."
    python /app/run_astrum_local.py adaptive greedy_fairness random
    echo "Training complete."
    exec sleep 3d
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Valid modes: server, demo, all, train"
    exit 1
    ;;
esac
