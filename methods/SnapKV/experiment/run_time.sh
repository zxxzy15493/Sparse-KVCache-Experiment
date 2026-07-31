#!/bin/bash

Llama="meta-llama/Llama-3.1-8B-Instruct"

python breaktime.py --model "$Llama" --dataset "../../../benchmarks/myinput.txt"


