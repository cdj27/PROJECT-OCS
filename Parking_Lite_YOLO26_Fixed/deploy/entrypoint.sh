#!/bin/sh
set -eu
python /app/deploy/verify_models.py
exec "$@"
