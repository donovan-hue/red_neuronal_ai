"""Claves individuales aleatorias y roles; nunca se almacenan tokens en claro."""
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import tempfile
import time
from pathlib import Path

DEFAULT_TTL_DAYS = 90


class AuthConfigError(Exception):
    pass


def load_users(path):
    try:
        with open(path, encoding='utf-8') as stream:
            data = json.load(stream)
        users = data['users']
        if not isinstance(users, list) or len(users) > 1000:
            raise ValueError('usuarios inválidos')
        seen = set()
        for user in users:
            if (not isinstance(user, dict) or not isinstance(user.get('name'), str)
                or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', user['name'])
                or user.get('role') not in ('user', 'admin')
                or not isinstance(user.get('sha256'), str)
                or not re.fullmatch(r'[0-9a-f]{64}', user['sha256'])
                or user['name'] in seen):
                raise ValueError('entrada inválida')
            expiry = user.get('expires_at')
            if expiry is not None and (isinstance(expiry, bool) or not isinstance(expiry, (int, float))
                                       or not math.isfinite(expiry) or expiry <= 0):
                raise ValueError('caducidad inválida')
            seen.add(user['name'])
        return users
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise AuthConfigError('Archivo de autenticación inválido o inaccesible') from exc


def authenticate(path, authorization):
    users = load_users(path)
    prefix = 'Bearer '
    if not authorization.startswith(prefix):
        return None
    token = authorization[len(prefix):]
    if not re.fullmatch(r'[0-9a-f]{64}', token):
        return None
    digest = hashlib.sha256(token.encode('ascii')).hexdigest()
    found = None
    now = time.time()
    for user in users:
        if hmac.compare_digest(user['sha256'], digest):
            # Compatibilidad con archivos anteriores: las claves sin fecha no caducan.
            if user.get('expires_at') is None or now < user['expires_at']:
                found = {'name': user['name'], 'role': user['role']}
    return found


def _save_users(path, users):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.nexus-auth-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump({'users': users}, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _new_key(ttl_days):
    if isinstance(ttl_days, bool) or not isinstance(ttl_days, int) or not 1 <= ttl_days <= 3650:
        raise ValueError('La vigencia debe ser entre 1 y 3650 días')
    token = secrets.token_hex(32)
    return token, hashlib.sha256(token.encode('ascii')).hexdigest(), int(time.time()) + ttl_days * 86400


def add_user(path, name, role, ttl_days=DEFAULT_TTL_DAYS):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', name) or role not in ('user', 'admin'):
        raise ValueError('Nombre o rol inválido')
    path = Path(path)
    users = load_users(path) if path.exists() else []
    if len(users) >= 1000 or any(user['name'] == name for user in users):
        raise ValueError('Usuario duplicado o límite de usuarios alcanzado')
    token, digest, expires_at = _new_key(ttl_days)
    users.append({'name': name, 'role': role, 'sha256': digest, 'expires_at': expires_at})
    _save_users(path, users)
    return token


def rotate_user(path, name, ttl_days=DEFAULT_TTL_DAYS):
    users = load_users(path)
    user = next((item for item in users if item['name'] == name), None)
    if user is None:
        raise ValueError('Usuario no encontrado')
    token, digest, expires_at = _new_key(ttl_days)
    user.update(sha256=digest, expires_at=expires_at)
    _save_users(path, users)
    return token


def revoke_user(path, name):
    users = load_users(path)
    remaining = [user for user in users if user['name'] != name]
    if len(remaining) == len(users):
        raise ValueError('Usuario no encontrado')
    _save_users(path, remaining)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Administra claves; guarda los tokens de forma segura.')
    parser.add_argument('--file', required=True)
    parser.add_argument('--user', required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--rotate', action='store_true')
    action.add_argument('--revoke', action='store_true')
    parser.add_argument('--role', choices=['user', 'admin'])
    parser.add_argument('--ttl-days', type=int, default=DEFAULT_TTL_DAYS)
    args = parser.parse_args()
    if args.revoke:
        if args.role or args.ttl_days != DEFAULT_TTL_DAYS:
            parser.error('--revoke no admite --role ni --ttl-days')
        revoke_user(args.file, args.user)
        print('Usuario revocado')
    elif args.rotate:
        if args.role:
            parser.error('--rotate conserva el rol existente')
        print(rotate_user(args.file, args.user, args.ttl_days))
    else:
        if not args.role:
            parser.error('--role es obligatorio para crear un usuario')
        print(add_user(args.file, args.user, args.role, args.ttl_days))
