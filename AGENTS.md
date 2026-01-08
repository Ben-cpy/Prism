# AGNET
+ this is my whole plan, i would implement this gradualy @Prism/tasks/whole_plan.md
+ this is a jetson device with CUDA, with 12 core, when you build, use the full core
+ models are in the folder /models (this is the root path, not relative path)
➜  /models ls, for fast dev use the 1.7B model
Qwen3-8B-Q4_K_M.gguf  Qwen3-8B-Q5_K_M.gguf  Qwen3-8B-Q8_0.gguf         Qwen_Qwen3-4B-Q8_0.gguf
Qwen3-8B-Q5_0.gguf    Qwen3-8B-Q6_K.gguf    Qwen_Qwen3-1.7B-Q8_0.gguf
+ use cpu/CUDA to run, example
./build-cpu/bin/llama-cli -m /models/Qwen3-8B-Q8_0.gguf -p "Hello, I come from US, where are you come from ?" -n 8 --temp 1.0 --seed 1 -st 
GGML_CUDA=ON ./build/bin/llama-cli -m /models/Qwen3-8B-Q8_0.gguf -p "Hello, I come from US, where are you come from ?" -n 8 --temp 1.0 --seed 1 -st 
+ you could ref to @Prism/tasks/guide.md when you want to modify some data structure
+ current impementation just focus on the traditional attention, not consider the flash-attention now.
+ When the build process is terminated due to failure, particularly one stemming from linkage issues within the llama-gguf / libllama compilation targets, the preferred remedial action is the prioritized removal of the residual object files followed by a complete reconstruction.