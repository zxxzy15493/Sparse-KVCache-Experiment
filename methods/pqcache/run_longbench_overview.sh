#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

"$script_dir/run_longbench_original.sh"
"$script_dir/run_longbench_no_drop_lb.sh"
"$script_dir/run_longbench_no_drop_lb_topp.sh"
"$script_dir/run_longbench_pqcache.sh"
