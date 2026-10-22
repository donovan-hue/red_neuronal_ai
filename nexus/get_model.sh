#!/usr/bin/env bash
# Descarga los artefactos entrenados (fuera del arbol de git) y verifica su integridad.
# Uso: bash nexus/get_model.sh [tag]      (tag por defecto: v0.1)
set -euo pipefail
cd "$(dirname "$0")"   # raiz del proyecto NEXUS (contiene src/, models/, data/)
VER="${1:-v0.1}"
BASE="https://github.com/donovan-hue/red_neuronal_ai/releases/download/${VER}"
SHA_PT="61351f52d975a82e8f8a433e"        # nexus-1.pt, 87525399 B
SHA_TK="44e5884e6fc3a9bae240207c"        # tokenizer.json, 553063 B
SHA_TR="e2f98eb4921725c3f2e7814d"        # train.bin, 15448690 B
mkdir -p models data/tokens
get() { # <destino> <nombre-en-release> <sha esperado>
  local dst="$1" src="$2" want="$3"
  if [ -f "$dst" ]; then
    echo "ya existe: $dst"
  else
    echo "descargando: $src"
    curl -fL --retry 3 -o "$dst.part" "${BASE}/${src}"
    mv "$dst.part" "$dst"
  fi
  local got; got=$(sha256sum "$dst" | cut -c1-24)
  if [ "$got" != "$want" ]; then
    echo "CHECKSUM NO COINCIDE en $dst (esperado $want, obtenido $got)" >&2
    exit 1
  fi
  echo "  ok  sha256 $got"
}
get models/nexus-1.pt nexus-1.pt "$SHA_PT"
get models/tokenizer.json tokenizer.json "$SHA_TK"
get data/tokens/train.bin train.bin "$SHA_TR"
echo
echo "Listo. Motor local disponible:"
echo "  bash run.sh ask \"cuanto es 21*43\"   # agente con el motor entrenado"
echo "  bash run.sh serve                    # UI web local"
echo "  bash run.sh train 2000               # o seguir entrenando (usa --resume)"
