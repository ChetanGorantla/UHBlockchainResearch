import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load an alternative model (e.g., GPT-Neo 125M)
model_name = "EleutherAI/gpt-neo-125M"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the input prompt
prompt = "What is the color of the sky?"

# Function to generate output on a node and return performance metrics
def test_model(node_url, prompt, retries=3, timeout=10):
    attempt = 0
    while attempt < retries:
        try:
            start_time = time.time()  # Start time for the process

            # Encode the input prompt
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            # Generate output
            output = model.generate(
                inputs, 
                max_length=50, 
                temperature=0,  # Use deterministic generation
                do_sample=False   # No sampling for consistent output
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
                'response_time': latency,
                'model': model_name
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
        'model': model_name
    }

# Define the nodes' URLs
node0_url = 'http://127.0.0.1:26657'
node1_url = 'http://127.0.0.1:26667'

# Test the model on both nodes
node0_metrics = test_model(node0_url, prompt)
node1_metrics = test_model(node1_url, prompt)

# Print the metrics for both nodes
print("Node 0 Metrics:", node0_metrics)
print("Node 1 Metrics:", node1_metrics)
