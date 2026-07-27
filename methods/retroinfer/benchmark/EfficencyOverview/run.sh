
# EfficencyOverview
# input_length: 4k, 8k, 16k, 32k, 64k, 128k
bash efficencyOverview.sh llama3.1-8b RetroInfer 0.018 0.232 1024 1 0 0 32 Full_Flash_Attn

bash efficencyOverview.sh qwen2.5-7b RetroInfer 0.018 0.232 1024 1 0 0 32 Full_Flash_Attn


# # # EfficencyBudget -- llama
# # # SISO(4k, 32) budget 128, 256, 512, 1024
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 128 1 0 0 32 Full_Flash_Attn 4096
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 256 1 0 0 32 Full_Flash_Attn 4096
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 512 1 0 0 32 Full_Flash_Attn 4096
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 1024 1 0 0 32 Full_Flash_Attn 4096

# # SISO(64k, 32) budget 128, 384, 1024, 4096, 16k
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 128 1 0 0 32 Full_Flash_Attn 65536
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 384 1 0 0 32 Full_Flash_Attn 65536
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 1024 1 0 0 32 Full_Flash_Attn 65536
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 4096 1 0 0 32 Full_Flash_Attn 65536
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 16384 1 0 0 32 Full_Flash_Attn 65536

# # EfficencyBudget -- qwen
# # SISO(4k, 32) budget 128, 256, 512, 1024
bash efficencyBudget.sh qwen2.5-7b RetroInfer 0.018 0.232 128 1 0 0 32 Full_Flash_Attn 4096
bash efficencyBudget.sh qwen2.5-7b RetroInfer 0.018 0.232 256 1 0 0 32 Full_Flash_Attn 4096
bash efficencyBudget.sh qwen2.5-7b RetroInfer 0.018 0.232 512 1 0 0 32 Full_Flash_Attn 4096
bash efficencyBudget.sh qwen2.5-7b RetroInfer 0.018 0.232 1024 1 0 0 32 Full_Flash_Attn 4096

# # SISO(64k, 32) budget 128, 384, 1024, 4096, 16k
bash efficencyBudget.sh qwen2.5-7b RetroInfer 0.018 0.232 128 1 0 0 32 Full_Flash_Attn 65536
bash efficencyBudget.sh qwen2.5-7b RetroInfer 0.018 0.232 384 1 0 0 32 Full_Flash_Attn 65536
bash efficencyBudget.sh qwen2.5-7b RetroInfer 0.018 0.232 1024 1 0 0 32 Full_Flash_Attn 65536
bash efficencyBudget.sh qwen2.5-7b RetroInfer 0.018 0.232 4096 1 0 0 32 Full_Flash_Attn 65536
bash efficencyBudget.sh qwen2.5-7b RetroInfer 0.018 0.232 16384 1 0 0 32 Full_Flash_Attn 65536

