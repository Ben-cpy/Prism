下面是面向“注意力调度/计算、KV cache 管理、多头注意力、attention softmax”的导航式说明文档（尽量简洁、便于定位改动点）。我按模块分组，并指出关键函数/类与它们的作用，便于你快速切入修改。

1) 入口与总体配置（注意力开关/调度参数）
文件：include/llama.h

struct llama_context_params：注意力相关的全局配置入口（如 flash_attn_type、kv_unified、offload_kqv 等），会影响图构建/运行路径与 KV 缓存行为。建议先核对/修改这里是否需要新增配置或开关。

2) 注意力图构建与多头注意力核心逻辑
文件：src/llama-graph.h

llm_graph_context::build_attn_mha：核心多头注意力构建入口（Q/K/V、mask、softmax/flash、输出 reshape 等）。若你要改注意力计算流程、算子替换或融合，这里是首要入口。

build_attn(...) 系列重载：分别处理无 KV cache、KV cache、ISWA KV cache、交叉注意力等路径。你若要改变注意力管线或统一实现，可从这些 overload 开始。

llm_graph_input_attn_* 输入类（如 llm_graph_input_attn_no_cache / llm_graph_input_attn_kv / llm_graph_input_attn_kv_iswa）：这些类负责为注意力图准备 mask 与 KV 索引输入，是调度/数据准备的一部分。

文件：src/llama-graph.cpp

llm_graph_context::build_attn_mha(...)：

Flash Attention 路径：ggml_flash_attn_ext（输入需注意 kq_b 不支持、类型 cast、sinks 等）。

非 Flash 路径：ggml_mul_mat 计算 KQ，ggml_soft_max_ext 执行 softmax（可注入 bias、mask、softcap），再 ggml_mul_mat 得到 KQV。
这是你调度/替换注意力计算、softmax、或改参数的主要位置。

llm_graph_input_attn_kv::set_input / llm_graph_input_attn_kv_iswa::set_input：KV cache 相关输入 mask/索引的实际设置路径（调度/输入准备会影响 attention mask 与索引形状）。

3) KV Cache 管理（含 SWA/统一缓存）
文件：src/llama-kv-cache.h

class llama_kv_cache：KV cache 主类（管理 cache 结构、slot info、K/V view、copy、mask 输入等）。
关键接口：

prepare / update / find_slot / apply_ubatch：KV slot 分配与更新流程。

build_input_k_idxs / build_input_v_idxs / set_input_kq_mask：为注意力构建 mask 与索引输入。

文件：src/llama-kv-cache.cpp

set_input_kq_mask(...)：核心 attention mask 构建（因果/非因果、SWA、ALiBi、2D pos 等），是注意力 mask 行为的关键位置。

set_input_k_idxs / set_input_v_idxs：将 KV cache slot index 写入图输入。

seq_rm / seq_cp / seq_keep / seq_add：KV cache 清理/复制/维护等操作。

文件：src/llama-kv-cache-iswa.h

llama_kv_cache_iswa：SWA 场景下的双 KV cache（base + SWA）组合。若涉及 SWA/混合注意力，重点查看这块。

4) Softmax / Flash Attention 原子算子接口
文件：ggml/include/ggml.h

ggml_soft_max / ggml_soft_max_ext 及其 _inplace/_back：softmax 的标准入口（含 mask、scale、ALiBi bias）。

ggml_flash_attn_ext 及相关 setter：Flash Attention 接口，注意其输入 tensor 维度要求与 mask 广播规则。

5) 各模型注意力“调用点”（模型图构建入口）
文件：src/models/llama.cpp

llm_build_llama 中的 build_attn(...) 调用：这是 LLaMA 系列模型具体组装 Q/K/V、RoPE、norm、attn 的模板。你要修改“注意力计算流程/调度”时，可从这一具体模型实现验证效果。

其他模型（如 src/models/qwen2.cpp、src/models/gemma3.cpp 等）也会调用 build_attn(...)，如果你要改“多模型共享的注意力策略”，建议先改 llm_graph_context::build_attn_mha，再回到各模型确认需要的特殊逻辑。

建议的修改切入点（按你的目标）
注意力调度/计算流程：优先从 llm_graph_context::build_attn_mha（src/llama-graph.cpp）修改，统一改变 Flash/非 Flash 注意力路径。

KV cache 管理/索引/Mask 规则：查看 llama_kv_cache::set_input_kq_mask 及 set_input_k_idxs/set_input_v_idxs。

多头注意力/softmax 算子层：ggml_soft_max_ext 与 ggml_flash_attn_ext 的接口定义和参数语义在 ggml/include/ggml.h，修改或替换算子时先确认这里的契约。

模型层调用链：以 src/models/llama.cpp 为模板查看 Q/K/V 生成、RoPE、norm、build_attn 调用顺序。

Commands I ran
rg -n "attention|attn|kv cache|kvcache|softmax|multihead|mha" src include ggml common examples

rg -n "kv cache|kv_cache|kv-cache|kv\\s+cache|kvcache" src include ggml

rg -n "build_attn|attn" src/llama-graph.cpp src/llama-graph.h src/llama.cpp

rg -n "soft_max" ggml/include/ggml.h

nl -ba ... | sed -n ... to inspect the listed files and ranges.