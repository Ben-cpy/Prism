# AGNET

+ this is a jetson device with CUDA
+ models are in the folder /models (this is the root path, not relative path)
➜  /models ls, for fast dev use the 1.7B model
Qwen3-8B-Q4_K_M.gguf  Qwen3-8B-Q5_K_M.gguf  Qwen3-8B-Q8_0.gguf         Qwen_Qwen3-4B-Q8_0.gguf
Qwen3-8B-Q5_0.gguf    Qwen3-8B-Q6_K.gguf    Qwen_Qwen3-1.7B-Q8_0.gguf
+ use cpu/CUDA to run, example
./build-cpu/bin/llama-cli -m /models/Qwen3-8B-Q8_0.gguf -p "Hello, I come from US, where are you come from ?" -n 8 --temp 1.0 --seed 1 -st 
GGML_CUDA=ON ./build/bin/llama-cli -m /models/Qwen3-8B-Q8_0.gguf -p "Hello, I come from US, where are you come from ?" -n 8 --temp 1.0 --seed 1 -st 