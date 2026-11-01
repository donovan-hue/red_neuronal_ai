#!/usr/bin/env bash
# Modo individual local: no modifica configuración ni genera credenciales sin permiso.
set -euo pipefail
cd "$(dirname "$0")"
umask 077
private_dir="${NEXUS_PRIVATE_DIR:-$HOME/.nexus-private}"
mkdir -p "$private_dir"
chmod 700 "$private_dir"
export NEXUS_AUTH_FILE="${NEXUS_AUTH_FILE:-$private_dir/users.json}"
export NEXUS_RATE_DB="${NEXUS_RATE_DB:-$private_dir/limits.sqlite}"
export NEXUS_PRIVATE_MODE=1
if [[ ! -f "$NEXUS_AUTH_FILE" ]]; then
  echo "Falta crear tu usuario. Ejecuta primero:" >&2
  echo "python3 src/auth.py --file \"\$HOME/.nexus-private/users.json\" --user propietario --role admin" >&2
  echo "Si usas NEXUS_PRIVATE_DIR o NEXUS_AUTH_FILE, usa esa ruta en su lugar." >&2
  exit 1
fi
exec ./run.sh serve
