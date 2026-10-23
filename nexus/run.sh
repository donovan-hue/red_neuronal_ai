#!/usr/bin/env bash
# Punto de entrada NEXUS; aísla dependencias en .venv para no modificar Python del sistema.
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-help}" == "help" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "uso: ./run.sh {setup|data|train [pasos]|serve|eval [flags]|ask 'pregunta'|all [pasos]}"
  echo "  setup descarga corpus y prepara tokenizer/dataset"
  echo "  all prepara el corpus, entrena en segundo plano y levanta la UI en :8080"
  exit 0
fi
case "${1:-}" in setup|data|train|serve|eval|ask|all) ;; *)
  echo "uso: ./run.sh {setup|data|train [pasos]|serve|eval [flags]|ask 'pregunta'|all [pasos]}"
  exit 2;; esac
mkdir -p runs models data/tokens
VENV="$PWD/.venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[deps] creando entorno virtual local en $VENV"
  python3 -m venv "$VENV"
fi
PYTHON="$VENV/bin/python"

need_torch() {
  "$PYTHON" -c "import torch" 2>/dev/null || {
    echo "[deps] instalando PyTorch en .venv…"
    "$PYTHON" -m pip install -q 'torch>=2.2'
  }
}
need_python_deps() {
  "$PYTHON" -c "import numpy, tokenizers, httpx" 2>/dev/null || {
    echo "[deps] instalando numpy, tokenizers y httpx en .venv…"
    "$PYTHON" -m pip install -q 'numpy>=1.26' 'tokenizers>=0.15,<1' 'httpx>=0.27,<1'
  }
}
need_all() { need_torch; need_python_deps; }
have_corpus() { find data -maxdepth 1 -type f -name '*.txt' -print -quit | grep -q .; }
ensure_corpus() {
  if ! have_corpus; then bash src/fetch_corpus.sh; fi
  if ! have_corpus; then echo "[error] no se pudo descargar ningún documento" >&2; exit 1; fi
}
prepare_dataset() {
  ensure_corpus
  "$PYTHON" src/corpus.py "$@"
}
have_dataset() {
  [[ -f data/tokens/train.bin && -f data/tokens/val.bin && -f data/tokens/vocab.json && -f models/tokenizer.json ]]
}
have_checkpoint() { [[ -f "$(python3 -c 'import json; print(json.load(open("config.json"))["train"]["ckpt"])')" ]]; }
is_release_dataset() {
  "$PYTHON" -c 'import json; m=json.load(open("data/tokens/manifest.json")); raise SystemExit(0 if str(m.get("source", "")).startswith("release ") else 1)' 2>/dev/null
}

case "${1:-help}" in
  setup)
    need_all
    ensure_corpus
    if is_release_dataset && have_dataset; then
      echo "[data] preservando tokenizer/dataset asociados al checkpoint publicado"
    elif ! have_dataset || ! is_release_dataset; then
      prepare_dataset
    fi
    ;;
  data)
    need_python_deps
    ensure_corpus
    if have_checkpoint; then
      if ! have_dataset; then
        echo "[error] checkpoint existente sin datos/tokenizer completos; ejecuta ./get_model.sh o restaura los artefactos" >&2
        exit 1
      fi
      echo "[data] checkpoint existente: conservo train.bin/tokenizer para evitar incompatibilidad; corpus listo para RAG"
    else
      "$PYTHON" src/corpus.py --force
    fi
    ;;
  train)
    need_all
    if [[ -n "${2:-}" ]]; then "$PYTHON" -u src/train.py --steps "$2"; else "$PYTHON" -u src/train.py; fi
    ;;
  serve)
    need_all
    "$PYTHON" -u src/server.py
    ;;
  eval)
    need_all
    shift
    "$PYTHON" -u src/eval.py "$@"
    ;;
  ask)
    need_all
    shift
    "$PYTHON" src/ask.py "$@"
    ;;
  all)
    need_all
    if ! have_dataset; then
      if have_checkpoint; then
        echo "[error] hay checkpoint pero faltan tokenizer/datos compatibles; ejecuta ./get_model.sh" >&2
        exit 1
      fi
      prepare_dataset
    elif ! have_corpus; then
      echo "[warn] datos de entrenamiento existen, pero el corpus de búsqueda está vacío"
    fi
    nohup "$PYTHON" -u src/train.py --steps "${2:-3500}" >> runs/train.log 2>&1 &
    echo "[train] proceso iniciado (PID $!)"
    "$PYTHON" -u src/server.py
    ;;
esac
