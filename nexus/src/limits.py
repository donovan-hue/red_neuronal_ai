"""Limitador compartido por procesos en un archivo SQLite local confiable."""
import json
import sqlite3
import time


class LimitStoreError(Exception):
    pass


def allow_request(db_path, identity, limit, *, now=None):
    """Consume una cuota de una ventana móvil de 60 s con transacción exclusiva.

    Requiere un filesystem local con bloqueo SQLite fiable; no es un limitador
    distribuido entre hosts con discos independientes.
    """
    now = time.time() if now is None else now
    key = json.dumps(identity, ensure_ascii=True, separators=(',', ':'))
    connection = None
    try:
        connection = sqlite3.connect(db_path, timeout=2, isolation_level=None)
        connection.execute('PRAGMA busy_timeout=2000')
        connection.execute('BEGIN IMMEDIATE')
        connection.execute('CREATE TABLE IF NOT EXISTS requests (identity TEXT NOT NULL, stamp REAL NOT NULL)')
        connection.execute('CREATE INDEX IF NOT EXISTS requests_identity_stamp ON requests(identity, stamp)')
        connection.execute('CREATE INDEX IF NOT EXISTS requests_stamp ON requests(stamp)')
        connection.execute('DELETE FROM requests WHERE stamp <= ?', (now - 60,))
        count = connection.execute('SELECT COUNT(*) FROM requests WHERE identity = ?', (key,)).fetchone()[0]
        total = connection.execute('SELECT COUNT(*) FROM requests').fetchone()[0]
        allowed = count < limit and total < 100_000
        if allowed:
            connection.execute('INSERT INTO requests VALUES (?, ?)', (key, now))
        connection.execute('COMMIT')
        return allowed
    except (sqlite3.Error, OSError, ValueError) as exc:
        if connection is not None and connection.in_transaction:
            connection.rollback()
        raise LimitStoreError('No se pudo comprobar el límite de solicitudes') from exc
    finally:
        if connection is not None:
            connection.close()


def reserve_daily_call(db_path, limit, *, now=None):
    """Reserva una llamada al proveedor por día UTC, incluso si falla o expira."""
    import datetime
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('El límite diario debe ser un entero positivo')
    timestamp = time.time() if now is None else now
    day = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc).date().isoformat()
    connection = None
    try:
        connection = sqlite3.connect(db_path, timeout=2, isolation_level=None)
        connection.execute('PRAGMA busy_timeout=2000')
        connection.execute('BEGIN IMMEDIATE')
        connection.execute('CREATE TABLE IF NOT EXISTS provider_calls (day TEXT PRIMARY KEY, count INTEGER NOT NULL)')
        # Solo se conservan el día actual y anteriores recientes; nunca se eliminan
        # días futuros para no resetear cuotas por retrocesos del reloj.
        cutoff = (datetime.date.fromisoformat(day) - datetime.timedelta(days=30)).isoformat()
        connection.execute('DELETE FROM provider_calls WHERE day < ?', (cutoff,))
        row = connection.execute('SELECT count FROM provider_calls WHERE day = ?', (day,)).fetchone()
        count = row[0] if row else 0
        allowed = count < limit
        if allowed:
            connection.execute('INSERT INTO provider_calls(day, count) VALUES (?, 1) '
                               'ON CONFLICT(day) DO UPDATE SET count = count + 1', (day,))
        connection.execute('COMMIT')
        return allowed
    except (sqlite3.Error, OSError, ValueError, OverflowError) as exc:
        if connection is not None and connection.in_transaction:
            connection.rollback()
        raise LimitStoreError('No se pudo comprobar la cuota diaria') from exc
    finally:
        if connection is not None:
            connection.close()
