import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the model and tokenizer for SmolLM
checkpoint = "HuggingFaceTB/SmolLM-360M"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Define the prompt
smol_prompt = 'What is the color of the sky? Answer in one word.'

# Function to test SmolLM on a node with all performance metrics
def test_smol_model(node_url, prompt, retries=3, timeout=10):
    attempt = 0
    while attempt < retries:
        try:
            start_time = time.time()  # Start time for the entire process

            # Encode the prompt and generate text with SmolLM
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
            outputs = model.generate(inputs, max_length=50)
            generated_text = tokenizer.decode(outputs[0])

            end_time = time.time()  # End time for the entire process
            latency = end_time - start_time  # Total time taken for the process
            
            # Return all relevant metrics and the generated response
            return {
                'node_url': node_url,
                'response': generated_text,
                'latency': latency,
                'response_time': latency,  # For local models, latency equals response time
                'model': 'SmolLM-360M'
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
        'model': 'SmolLM-360M'
    }

# Define the nodes' URLs
node0_url = 'http://127.0.0.1:26657'
node1_url = 'http://127.0.0.1:26667'

# Test the SmolLM model on both nodes
node0_metrics = test_smol_model(node0_url, smol_prompt)
node1_metrics = test_smol_model(node1_url, smol_prompt)

# Print the metrics for both nodes
print("Node 0 Metrics (SmolLM):", node0_metrics)
print("Node 1 Metrics (SmolLM):", node1_metrics)
