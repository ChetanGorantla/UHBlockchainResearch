import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the smol llama model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("BEE-spoke-data/smol_llama-101M-GQA-python", use_fast=False)

# Option 1: Set eos_token as pad_token
tokenizer.pad_token = tokenizer.eos_token

# Option 2: Alternatively, add a new pad_token (if needed)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model.resize_token_embeddings(len(tokenizer))

model = AutoModelForCausalLM.from_pretrained("BEE-spoke-data/smol_llama-101M-GQA-python", device_map="auto")

# Define the input prompt
prompt = "What is the answer to 2+2?"

# Function to generate output on a node and return performance metrics
def test_smol_llama_model(node_url, prompt, retries=3, timeout=10):
    attempt = 0
    while attempt < retries:
        try:
            start_time = time.time()  # Start time for the process

            # Encode the input prompt with attention mask
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
            output = model.generate(
                inputs['input_ids'], 
                attention_mask=inputs['attention_mask'],  # Add attention mask
                max_length=100,         # Limit the response length
                temperature=0.7,        # Lower temperature for more coherence
                do_sample=True          # Sampling enabled
            )

            # Decode the output
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            end_time = time.time()  # End time for the process
            latency = end_time - start_time  # Total time taken for the process
            
            # Return performance metrics and generated response
            return {
                'node_url': node_url,
                'response': generated_text,
                'latency': latency,
                'response_time': latency,  # For local models, latency equals response time
                'model': 'smol_llama-101M-GQA-python'
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
        'model': 'smol_llama-101M-GQA-python'
    }

# Define the nodes' URLs
node0_url = 'http://127.0.0.1:26657'
node1_url = 'http://127.0.0.1:26667'

# Test the smol_llama model on both nodes
node0_metrics = test_smol_llama_model(node0_url, prompt)
node1_metrics = test_smol_llama_model(node1_url, prompt)

# Print the metrics for both nodes
print("Node 0 Metrics (smol_llama-101M-GQA-python):", node0_metrics)
print("Node 1 Metrics (smol_llama-101M-GQA-python):", node1_metrics)
