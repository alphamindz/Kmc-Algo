#!/bin/bash
set -e

MODE="${KMC_ALGO_MODE:-all}"

case "$MODE" in
  server)
    echo "Starting KMC-Algo API server on port 7860..."
    exec python -m uvicorn Kmcalgo.kmc_env.server:app --host 0.0.0.0 --port 7860
    ;;
  demo)
    echo "Starting Gradio demo on port 7860..."
    cd /app/Kmc_space
    exec python app.py
    ;;
  all)
    echo "Starting KMC-Algo API server on port 7860..."
    python -m uvicorn Kmcalgo.kmc_env.server:app --host 0.0.0.0 --port 7860 &
    API_PID=$!

    echo "Starting Gradio demo on port 7861..."
    GRADIO_SERVER_PORT=7861 python /app/Kmc_space/app.py &
    DEMO_PID=$!

    echo "Both services running: API=:7860  Demo=:7861"
    wait $API_PID $DEMO_PID
    ;;
  train)
    echo "Running simulation episode comparison..."
    python /app/main.py adaptive greedy_fairness random
    echo "Simulation complete."
    exec sleep 3d
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Valid modes: server, demo, all, train"
    exit 1
    ;;
esac
