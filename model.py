from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

streamer = TextStreamer(tokenizer, skip_special_tokens=True)

while True:
    prompt = input(">>> ")
    if not prompt:
        break
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    model.generate(**inputs, max_new_tokens=1024, streamer=streamer)