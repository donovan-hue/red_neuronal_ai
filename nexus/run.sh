#!/usr/bin/env bash
# Punto de entrada unico. El sandbox no conserva paquetes instalados: esto los trae.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p runs models data/tokens

need_torch() { python3 -c "import torch" 2>/dev/null || { echo "[deps] instalando torch (cpu)…"; pip install -q --index-url https://download.pytorch.org/whl/cpu torch; }; }
need_tok()   { python3 -c "import tokenizers" 2>/dev/null || { echo "[deps] instalando tokenizers…"; pip install -q tokenizers; }; }

case "${1:-help}" in
  setup)   need_torch; need_tok; [ -f data/tokens/train.bin ] || python3 src/corpus.py ;;
  data)    need_tok; [ -n "$(ls data/*.txt 2>/dev/null)" ] || bash src/fetch_corpus.sh; python3 src/corpus.py ${2:-} ;;
  train)   need_torch; python3 -u src/train.py --steps "${2:-4000}" ;;
  serve)   need_torch; need_tok; python3 -u src/server.py ;;
  eval)    need_torch; need_tok; python3 -u src/eval.py ${2:-} ;;
  ask)     need_torch; need_tok; shift; python3 src/ask.py "$@" ;;
  all)     need_torch; need_tok
           [ -f data/tokens/train.bin ] || { [ -n "$(ls data/*.txt 2>/dev/null)" ] || bash src/fetch_corpus.sh; python3 src/corpus.py; }
           nohup python3 -u src/train.py --steps "${2:-4000}" >> runs/train.log 2>&1 &
           sleep 5; python3 -u src/server.py ;;
  *) echo "uso: ./run.sh {setup|data|train [pasos]|serve|eval [flags]|ask 'pregunta'|all [pasos]}"
     echo "  all  -> entrena en segundo plano y levanta la UI en :8080" ;;
esac
