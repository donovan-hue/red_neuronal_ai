#!/usr/bin/env bash
# Descarga el corpus literario bilingue (Project Gutenberg, dominio publico en EE.UU.)
# que se us6 para entrenar. Idempotente: solo baja lo que falta.
set -u
D="$(cd "$(dirname "$0")/.." && pwd)/data"; mkdir -p "$D"; cd "$D"
IDS="11 2701 98 74 76 1400 345 5200 2542 161 730 64317 1268 4300 1727 219 205 514 1080 158 1342 84 1661 2000"
for i in $IDS; do
  f="gb_$i.txt"; [ "$i" = "2000" ] && f="quijote.txt"
  if [ ! -s "$f" ]; then
    echo "[corpus] descargando $f"
    timeout 60 curl -sL -f -o "$f" "https://www.gutenberg.org/cache/epub/$i/pg$i.txt" || { echo "  fallo $i"; rm -f "$f"; }
  fi
done
echo "[corpus] $(cat *.txt 2>/dev/null | wc -c) bytes en $(ls *.txt 2>/dev/null | wc -l) archivos"
