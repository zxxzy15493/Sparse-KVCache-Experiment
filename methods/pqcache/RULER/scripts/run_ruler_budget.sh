#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

"$script_dir/run_ruler_no_drop_lb_budget.sh"
"$script_dir/run_ruler_no_drop_lb_topp_budget.sh"
"$script_dir/run_ruler_pqcache_budget.sh"
