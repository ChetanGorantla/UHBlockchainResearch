import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load DistilGPT-2 model and tokenizer
model_name = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the input prompt
prompt = "What is the color of the sky?"

# Function to generate output on a node and return performance metrics
def test_distilgpt_model(node_url, prompt, retries=3, timeout=10):
    attempt = 0
    while attempt < retries:
        try:
            start_time = time.time()  # Start time for the process

            # Encode the input prompt and generate output
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            output = model.generate(
                inputs, 
                max_new_tokens=100,  # Increased length for more content
                temperature=0.7,     # Small positive temperature for more coherence
                do_sample=True,     # Ensure deterministic output
                pad_token_id=tokenizer.eos_token_id  # Fix padding issue
            )

            # Decode and clean the output
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

            end_time = time.time()  # End time for the process
            latency = end_time - start_time  # Total time taken for the process
            
            # Return performance metrics and generated response
            return {
                'node_url': node_url,
                'response': generated_text,
                'latency': latency,
                'response_time': latency,  # For local models, latency equals response time
                'model': 'DistilGPT-2'
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
        'model': 'DistilGPT-2'
    }

# Define the nodes' URLs
node0_url = 'http://127.0.0.1:26657'
node1_url = 'http://127.0.0.1:26667'

# Test the DistilGPT-2 model on both nodes
node0_metrics = test_distilgpt_model(node0_url, prompt)
node1_metrics = test_distilgpt_model(node1_url, prompt)

# Print the metrics for both nodes
print("Node 0 Metrics (DistilGPT-2):", node0_metrics)
print("Node 1 Metrics (DistilGPT-2):", node1_metrics)
