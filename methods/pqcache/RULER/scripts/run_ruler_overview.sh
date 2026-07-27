#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

"$script_dir/run_ruler_original_overview.sh"
"$script_dir/run_ruler_no_drop_lb_overview.sh"
"$script_dir/run_ruler_no_drop_lb_topp_overview.sh"
"$script_dir/run_ruler_pqcache_overview.sh"
