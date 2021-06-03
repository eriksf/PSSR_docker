#!/bin/bash --login

set -euo pipefail

conda activate pssr

exec "$@"
