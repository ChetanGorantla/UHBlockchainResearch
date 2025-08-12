import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model and tokenizer for Qwen2.5-3B-Instruct
model_name = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the updated messages (prompt)
qwen_prompt = "What is 2+2?"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": qwen_prompt}
]

# Prepare input text using Qwen's chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Define the generation arguments
generation_args = { 
    "max_new_tokens": 512, 
}

# Function to test Qwen2.5-3B-Instruct on a node with all performance metrics
def test_qwen_model(node_url, model_inputs, retries=3, timeout=10):
    attempt = 0
    while attempt < retries:
        try:
            start_time = time.time()  # Start time for the entire process

            # Generate the response from Qwen2.5-3B-Instruct
            generated_ids = model.generate(**model_inputs, **generation_args)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            end_time = time.time()  # End time for the entire process
            latency = end_time - start_time  # Total time taken for the process
            
            # Return all relevant metrics and the generated response
            return {
                'node_url': node_url,
                'response': generated_text,
                'latency': latency,
                'response_time': latency,  # For local models, latency equals response time
                'model': 'Qwen2.5-3B-Instruct'
            }

        except Exception as e:
            print(f"Request failed: {e}. Attempt {attempt + 1} of {retries}. Retrying...")
            attempt += 1
            time.sleep(1)  # Brief pause before retrying

    # If all attempts fail
    return {
        'node_url': node_url,
        'response': None,
        'latency': None,
        'response_time': None,
        'model': 'Qwen2.5-3B-Instruct'
    }

# Define the nodes' URLs
node0_url = 'http://127.0.0.1:26657'
node1_url = 'http://127.0.0.1:26667'

# Test the Qwen2.5-3B-Instruct model on both nodes
node0_metrics = test_qwen_model(node0_url, model_inputs)
node1_metrics = test_qwen_model(node1_url, model_inputs)

# Print the metrics for both nodes
print("Node 0 Metrics (Qwen2.5-3B-Instruct):", node0_metrics)
print("Node 1 Metrics (Qwen2.5-3B-Instruct):", node1_metrics)
