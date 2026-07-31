# "Usage: $0 <model> $1{prefill_method}  $2 <attn_type> $3 <budget> $4 {input_max_token} $5{fixed_output_length}"

# VRAMOverview -- llama
# bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 64 1024 4096
# bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 512 1024 4096
# bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 64 1024 2
# bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 512 1024 2
bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 512 65536 2
bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 8192 65536 2

# # VRAMOverview -- qwen2.5-7b-1m
# bash VRAMOverview.sh qwen-2.5-7b-1m Full_Flash_Attn RetroInfer 64 1024 4096
# bash VRAMOverview.sh qwen-2.5-7b-1m Full_Flash_Attn RetroInfer 512 1024 4096
# bash VRAMOverview.sh qwen-2.5-7b-1m Full_Flash_Attn RetroInfer 64 1024 2
# bash VRAMOverview.sh qwen-2.5-7b-1m Full_Flash_Attn RetroInfer 512 1024 2
# bash VRAMOverview.sh qwen-2.5-7b-1m Full_Flash_Attn RetroInfer 512 65536 2
# bash VRAMOverview.sh qwen-2.5-7b-1m Full_Flash_Attn RetroInfer 8192 65536 2

# bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 1024 4096 2
# bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 1024 8192 2
# bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 1024 16384 2
# bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 1024 32768 2
# bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 1024 65536 2
# bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 1024 131072 2
# bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 1024 262144 2



# bash VRAMOverview.sh qwen-2.5-7b-1m Full_Flash_Attn RetroInfer 1024 4096 2
# bash VRAMOverview.sh qwen-2.5-7b-1m Full_Flash_Attn RetroInfer 1024 8192 2
# bash VRAMOverview.sh qwen-2.5-7b-1m Full_Flash_Attn RetroInfer 1024 16384 2
# bash VRAMOverview.sh qwen-2.5-7b-1m Full_Flash_Attn RetroInfer 1024 32768 2
# bash VRAMOverview.sh qwen-2.5-7b-1m Full_Flash_Attn RetroInfer 1024 65536 2
# bash VRAMOverview.sh qwen-2.5-7b-1m Full_Flash_Attn RetroInfer 1024 262144 2