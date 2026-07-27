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