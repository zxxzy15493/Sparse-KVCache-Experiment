# #!/bin/bash

# # AccuracyOverview
# # Usage: bash Accuracy.sh <model> <attn_type> <benchmark> <task> <num_examples> <budget>
# #   model:       llama3.1-8b, qwen2.5-7b, glm-4-9b-chat-1m
# #   attn_type:   Full_Flash_Attn, RetroInfer
# #   benchmark:   LongBench, Synthetic
# #   num_examples: -1 (all)
# #   budget:      1024 (default for RetroInfer)



# # ============================================================
# # LongBench -- llama3.1-8b
# # ============================================================
bash Accuracy.sh llama3.1-8b RetroInfer LongBench narrativeqa -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer LongBench qasper -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer LongBench 2wikimqa -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer LongBench musique -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer LongBench gov_report -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer LongBench multi_news -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer LongBench triviaqa -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer LongBench samsum -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer LongBench passage_count -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer LongBench passage_retrieval_en -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer LongBench lcc -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer LongBench repobench-p -1 1024

# # ============================================================
# # LongBench -- qwen2.5-7b
# # ============================================================
bash Accuracy.sh qwen2.5-7b RetroInfer LongBench narrativeqa -1 1024
# bash Accuracy.sh qwen2.5-7b RetroInfer LongBench qasper -1 1024
# bash Accuracy.sh qwen2.5-7b RetroInfer LongBench 2wikimqa -1 1024
# bash Accuracy.sh qwen2.5-7b RetroInfer LongBench musique -1 1024
# bash Accuracy.sh qwen2.5-7b RetroInfer LongBench gov_report -1 1024
# bash Accuracy.sh qwen2.5-7b RetroInfer LongBench multi_news -1 1024
# bash Accuracy.sh qwen2.5-7b RetroInfer LongBench triviaqa -1 1024
# bash Accuracy.sh qwen2.5-7b RetroInfer LongBench samsum -1 1024
# bash Accuracy.sh qwen2.5-7b RetroInfer LongBench passage_count -1 1024
# bash Accuracy.sh qwen2.5-7b RetroInfer LongBench passage_retrieval_en -1 1024
# bash Accuracy.sh qwen2.5-7b RetroInfer LongBench lcc -1 1024
# bash Accuracy.sh qwen2.5-7b RetroInfer LongBench repobench-p -1 1024

# # ============================================================
# # Synthetic (RULER) -- llama3.1-8b
# # ============================================================
bash Accuracy.sh llama3.1-8b RetroInfer Synthetic niah_single_1 -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer Synthetic niah_single_2 -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer Synthetic niah_single_3 -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer Synthetic niah_multikey_1 -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer Synthetic niah_multikey_2 -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer Synthetic niah_multikey_3 -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer Synthetic niah_multiquery -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer Synthetic niah_multivalue -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer Synthetic vt -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer Synthetic cwe -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer Synthetic fwe -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer Synthetic qa_1 -1 1024
# bash Accuracy.sh llama3.1-8b RetroInfer Synthetic qa_2 -1 1024

# # ============================================================
# # Synthetic (RULER) -- qwen2.5-7b
# # ============================================================
# bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic niah_single_1 -1 1024
# bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic niah_single_2 -1 1024
# bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic niah_single_3 -1 1024
# bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic niah_multikey_1 -1 1024
# bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic niah_multikey_2 -1 1024
# bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic niah_multikey_3 -1 1024
# bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic niah_multiquery -1 1024
# bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic niah_multivalue -1 1024
# bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic vt -1 1024
# bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic cwe -1 1024
# bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic fwe -1 1024
# bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic qa_1 -1 1024
# bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic qa_2 -1 1024

# echo ""
# echo "All tasks completed."
# echo "Results directory: ./results/pred/<model>/<attn_type>/"
