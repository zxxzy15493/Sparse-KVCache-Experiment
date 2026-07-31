
BUDGET_RATIO=0.018
ESTIMATE_RATIO=0.023
RATIO_OR_FIXED=1
RECALL=0
FIXED_OUTPUT_LENGTH=0
MEASURE_TIME=0
DEVICE=auto
DTYPE=bf16


#  fix  mode
if [ "$RATIO_OR_FIXED" -eq 1 ]; then
    MODE="fixed"
elif [ "$RATIO_OR_FIXED" -eq 0 ]; then
    MODE="ratio"
else
    # ， mode 
    echo "$RATIO_OR_FIXED"
fi
if [ "$RECALL" -eq 1 ]; then
    RECALL="--recall"
else
    RECALL=""
fi
if [ "$MEASURE_TIME" -eq 1 ]; then
    MEASURE_TIME="--measure_time"
else
    MEASURE_TIME=""
fi