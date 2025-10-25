#!/usr/bin/env bash
set -e
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
fi

streamlit run app.py --server.port 8501 --server.address 0.0.0.0