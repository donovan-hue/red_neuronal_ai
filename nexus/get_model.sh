#!/usr/bin/env bash
# Descarga el checkpoint publicado, valida SHA-256 completo y crea el split de validación.
set -euo pipefail
cd "$(dirname "$0")"
VER="${1:-v0.1}"
BASE="https://github.com/donovan-hue/red_neuronal_ai/releases/download/${VER}"
SHA_PT="61351f52d975a82e8f8a433e5ba6abf1629a65a97340203e388cf26250df6400"
SHA_TK="44e5884e6fc3a9bae240207caf838178beca5132f1830b5e2c85fc06b1e9d184"
SHA_TR="e2f98eb4921725c3f2e7814d91ad45560b78d23b24adc7489383be0dc01f6077"
mkdir -p models data/tokens

get_asset() {
  local destination="$1" asset="$2" expected="$3"
  if [[ ! -f "$destination" ]]; then
    local temporary="${destination}.part"
    rm -f "$temporary"
    echo "descargando: $asset"
    curl -fL --retry 3 -o "$temporary" "${BASE}/${asset}"
    local actual
    actual=$(sha256sum "$temporary" | awk '{print $1}')
    if [[ "$actual" != "$expected" ]]; then
      rm -f "$temporary"
      echo "SHA-256 inválido para $asset" >&2
      exit 1
    fi
    mv "$temporary" "$destination"
  fi
  local actual
  actual=$(sha256sum "$destination" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 inválido para $destination" >&2
    exit 1
  fi
  echo "verificado: $destination"
}

get_asset models/nexus-1.pt nexus-1.pt "$SHA_PT"
get_asset models/tokenizer.json tokenizer.json "$SHA_TK"

if [[ ! -f data/tokens/train.bin || ! -f data/tokens/val.bin || ! -f data/tokens/vocab.json || ! -f data/tokens/.source_train.sha256 ]]; then
  raw="data/tokens/train.release.part"
  rm -f "$raw"
  echo "descargando: train.bin"
  curl -fL --retry 3 -o "$raw" "${BASE}/train.bin"
  actual=$(sha256sum "$raw" | awk '{print $1}')
  if [[ "$actual" != "$SHA_TR" ]]; then
    rm -f "$raw"
    echo "SHA-256 inválido para train.bin" >&2
    exit 1
  fi
  NEXUS_RELEASE_TAG="$VER" python3 - "$raw" <<'PY'
import array, json, pathlib, sys
import hashlib
raw = pathlib.Path(sys.argv[1])
root = pathlib.Path("data/tokens")
tokenizer = json.loads(pathlib.Path("models/tokenizer.json").read_text(encoding="utf-8"))
vocab = tokenizer.get("model", {}).get("vocab")
if not isinstance(vocab, dict) or not vocab:
    raise SystemExit("tokenizer release no contiene un vocabulario verificable")
ids = list(vocab.values()) + [item.get("id", -1) for item in tokenizer.get("added_tokens", [])]
if not ids or any(not isinstance(token_id, int) for token_id in ids):
    raise SystemExit("ids del tokenizer inválidos")
vocab_size = max(ids) + 1
if sys.byteorder != "little":
    raise SystemExit("la partición de este artefacto uint16 requiere una máquina little-endian")
tokens = array.array("H")
with raw.open("rb") as f:
    tokens.fromfile(f, raw.stat().st_size // tokens.itemsize)
if len(tokens) < 1000:
    raise SystemExit("train.bin demasiado pequeño")
val_count = max(1, len(tokens) // 100)
train = tokens[:-val_count]
valid = tokens[-val_count:]
for name, values in (("train.bin", train), ("val.bin", valid)):
    with (root / name).open("wb") as f:
        values.tofile(f)
(root / "vocab.json").write_text(json.dumps({"vocab": vocab_size}), encoding="utf-8")
manifest = {"train_tokens": len(train), "val_tokens": len(valid), "vocab": vocab_size,
            "source": "release " + __import__("os").environ["NEXUS_RELEASE_TAG"],
            "source_train_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
for name in ("train.bin", "val.bin", "vocab.json"):
    manifest[name + "_sha256"] = hashlib.sha256((root / name).read_bytes()).hexdigest()
(root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
(root / ".source_train.sha256").write_text(manifest["source_train_sha256"], encoding="ascii")
PY
  rm -f "$raw"
fi
python3 - "$SHA_TR" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path("data/tokens")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("source_train_sha256") != sys.argv[1]:
    raise SystemExit("manifest no corresponde al train.bin publicado")
for name in ("train.bin", "val.bin", "vocab.json"):
    digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if digest != manifest.get(name + "_sha256"):
        raise SystemExit("artefacto corrupto: " + name)
PY

echo "Listo. Para usar el motor local: ./run.sh serve"
echo "Para continuar entrenamiento: ./run.sh train 4000"
