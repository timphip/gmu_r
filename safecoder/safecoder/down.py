from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
# 指定你想要保存的路径
save_path = "/root/autodl-tmp"

# 1. 下载分词器并保存到指定路径
print(f"正在下载分词器到 {save_path}...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    cache_dir=save_path
)

# 2. 下载模型权重并保存到指定路径
print(f"正在下载模型权重到 {save_path}（这可能需要一段时间）...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    cache_dir=save_path,
    low_cpu_mem_usage=True, 
    device_map="cpu" 
)

print(f"下载并保存完成！路径为: {save_path}")
# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"

# # 1. 下载分词器
# print("正在下载分词器...")
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# # 2. 下载模型权重
# # 使用 device_map="cpu" 可以确保在下载时不占用 GPU 显存
# # 如果只想下载不加载到内存，可以不写这行，但在 transformers 中通常通过加载来触发下载
# print("正在下载模型权重（这可能需要一段时间）...")
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, 
#     low_cpu_mem_usage=True, 
#     device_map="cpu" 
# )

# print("下载完成！")