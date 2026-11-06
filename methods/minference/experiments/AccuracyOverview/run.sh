# length: 
# 4096
# 8192
# 16384
# 32768
# 65536
# 131072
# 196608

# LongBench tasks
# bash Accuracy.sh llama-3.1-8b LongBench narrativeqa
# bash Accuracy.sh llama-3.1-8b LongBench qasper
# bash Accuracy.sh llama-3.1-8b LongBench 2wikimqa
# bash Accuracy.sh llama-3.1-8b LongBench musique
# bash Accuracy.sh llama-3.1-8b LongBench gov_report
# bash Accuracy.sh llama-3.1-8b LongBench multi_news
# bash Accuracy.sh llama-3.1-8b LongBench triviaqa
# bash Accuracy.sh llama-3.1-8b LongBench samsum
# bash Accuracy.sh llama-3.1-8b LongBench passage_count
# bash Accuracy.sh llama-3.1-8b LongBench passage_retrieval_en
# bash Accuracy.sh llama-3.1-8b LongBench lcc
# bash Accuracy.sh llama-3.1-8b LongBench repobench-p




# Synthetic tasks
bash Accuracy.sh llama-3.1-8b Synthetic niah_single_1 4096
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_2
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_3
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_1
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_2
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_3
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multiquery
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multivalue
# bash Accuracy.sh llama-3.1-8b Synthetic vt
# bash Accuracy.sh llama-3.1-8b Synthetic cwe
# bash Accuracy.sh llama-3.1-8b Synthetic fwe
# bash Accuracy.sh llama-3.1-8b Synthetic qa_1
# bash Accuracy.sh llama-3.1-8b Synthetic qa_2

# qwen-2.5-7b
# bash Accuracy.sh qwen-2.5-7b LongBench narrativeqa
# bash Accuracy.sh qwen-2.5-7b LongBench qasper
# bash Accuracy.sh qwen-2.5-7b LongBench 2wikimqa
# bash Accuracy.sh qwen-2.5-7b LongBench musique
# bash Accuracy.sh qwen-2.5-7b LongBench gov_report
# bash Accuracy.sh qwen-2.5-7b LongBench multi_news
# bash Accuracy.sh qwen-2.5-7b LongBench triviaqa
# bash Accuracy.sh qwen-2.5-7b LongBench samsum
# bash Accuracy.sh qwen-2.5-7b LongBench passage_count
# bash Accuracy.sh qwen-2.5-7b LongBench passage_retrieval_en
# bash Accuracy.sh qwen-2.5-7b LongBench lcc
# bash Accuracy.sh qwen-2.5-7b LongBench repobench-p


# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_1
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_2
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_3
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_1
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_2
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_3
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multiquery
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multivalue
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic vt
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic cwe
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic fwe
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_1
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_2

# glm-4-9b-chat-1m 
# bash Accuracy.sh glm-4-9b-chat-1m LongBench narrativeqa
# bash Accuracy.sh glm-4-9b-chat-1m LongBench qasper
# bash Accuracy.sh glm-4-9b-chat-1m LongBench 2wikimqa
# bash Accuracy.sh glm-4-9b-chat-1m LongBench musique
# bash Accuracy.sh glm-4-9b-chat-1m LongBench gov_report
# bash Accuracy.sh glm-4-9b-chat-1m LongBench multi_news
# bash Accuracy.sh glm-4-9b-chat-1m LongBench triviaqa
# bash Accuracy.sh glm-4-9b-chat-1m LongBench samsum
# bash Accuracy.sh glm-4-9b-chat-1m LongBench passage_count
# bash Accuracy.sh glm-4-9b-chat-1m LongBench passage_retrieval_en
# bash Accuracy.sh glm-4-9b-chat-1m LongBench lcc
# bash Accuracy.sh glm-4-9b-chat-1m LongBench repobench-p

# Qwen3 commands: uncomment the tasks to run.
# Optional GPU / allocator settings:
# export CUDA_VISIBLE_DEVICES=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# For a one-sample smoke test, use 32768 1 as the fourth and fifth arguments.

# qwen3-4b LongBench
bash Accuracy.sh qwen3-4b LongBench narrativeqa
# bash Accuracy.sh qwen3-4b LongBench qasper
# bash Accuracy.sh qwen3-4b LongBench 2wikimqa
# bash Accuracy.sh qwen3-4b LongBench musique
# bash Accuracy.sh qwen3-4b LongBench gov_report
# bash Accuracy.sh qwen3-4b LongBench multi_news
# bash Accuracy.sh qwen3-4b LongBench triviaqa
# bash Accuracy.sh qwen3-4b LongBench samsum
# bash Accuracy.sh qwen3-4b LongBench passage_count
# bash Accuracy.sh qwen3-4b LongBench passage_retrieval_en
# bash Accuracy.sh qwen3-4b LongBench lcc
# bash Accuracy.sh qwen3-4b LongBench repobench-p

# qwen3-4b RULER / Synthetic (32K)
# bash Accuracy.sh qwen3-4b Synthetic niah_single_1 32768
# bash Accuracy.sh qwen3-4b Synthetic niah_single_2 32768
# bash Accuracy.sh qwen3-4b Synthetic niah_single_3 32768
# bash Accuracy.sh qwen3-4b Synthetic niah_multikey_1 32768
# bash Accuracy.sh qwen3-4b Synthetic niah_multikey_2 32768
# bash Accuracy.sh qwen3-4b Synthetic niah_multikey_3 32768
# bash Accuracy.sh qwen3-4b Synthetic niah_multiquery 32768
# bash Accuracy.sh qwen3-4b Synthetic niah_multivalue 32768
# bash Accuracy.sh qwen3-4b Synthetic vt 32768
# bash Accuracy.sh qwen3-4b Synthetic cwe 32768
# bash Accuracy.sh qwen3-4b Synthetic fwe 32768
# bash Accuracy.sh qwen3-4b Synthetic qa_1 32768
# bash Accuracy.sh qwen3-4b Synthetic qa_2 32768

# qwen3-32b LongBench
# bash Accuracy.sh qwen3-32b LongBench narrativeqa
# bash Accuracy.sh qwen3-32b LongBench qasper
# bash Accuracy.sh qwen3-32b LongBench 2wikimqa
# bash Accuracy.sh qwen3-32b LongBench musique
# bash Accuracy.sh qwen3-32b LongBench gov_report
# bash Accuracy.sh qwen3-32b LongBench multi_news
# bash Accuracy.sh qwen3-32b LongBench triviaqa
# bash Accuracy.sh qwen3-32b LongBench samsum
# bash Accuracy.sh qwen3-32b LongBench passage_count
# bash Accuracy.sh qwen3-32b LongBench passage_retrieval_en
# bash Accuracy.sh qwen3-32b LongBench lcc
# bash Accuracy.sh qwen3-32b LongBench repobench-p

# qwen3-32b RULER / Synthetic (32K)
# bash Accuracy.sh qwen3-32b Synthetic niah_single_1 32768
# bash Accuracy.sh qwen3-32b Synthetic niah_single_2 32768
# bash Accuracy.sh qwen3-32b Synthetic niah_single_3 32768
# bash Accuracy.sh qwen3-32b Synthetic niah_multikey_1 32768
# bash Accuracy.sh qwen3-32b Synthetic niah_multikey_2 32768
# bash Accuracy.sh qwen3-32b Synthetic niah_multikey_3 32768
# bash Accuracy.sh qwen3-32b Synthetic niah_multiquery 32768
# bash Accuracy.sh qwen3-32b Synthetic niah_multivalue 32768
# bash Accuracy.sh qwen3-32b Synthetic vt 32768
# bash Accuracy.sh qwen3-32b Synthetic cwe 32768
# bash Accuracy.sh qwen3-32b Synthetic fwe 32768
# bash Accuracy.sh qwen3-32b Synthetic qa_1 32768
# bash Accuracy.sh qwen3-32b Synthetic qa_2 32768
