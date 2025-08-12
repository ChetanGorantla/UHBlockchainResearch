import time
from transformers import ReformerModelWithLMHead, AutoTokenizer

# Load Reformer model and tokenizer
model_name = "google/reformer-enwik8"
model = ReformerModelWithLMHead.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the input prompt
prompt = "What is the color of the sky?"

# Function to generate output on a node and return performance metrics
def test_reformer_model(node_url, prompt, retries=3, timeout=10):
    attempt = 0
    while attempt < retries:
        try:
            start_time = time.time()  # Start time for the process

            # Encode the input prompt and generate output
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(
                inputs.input_ids, 
                max_length=100,         # Increase the response length for Reformer
                temperature=0.7,        # Adjust temperature for more diverse output
                do_sample=True,         # Enable sampling to introduce variability
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
                'model': 'Reformer'
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
        'model': 'Reformer'
    }

# Define the nodes' URLs
node0_url = 'http://127.0.0.1:26657'
node1_url = 'http://127.0.0.1:26667'

# Test the Reformer model on both nodes
node0_metrics = test_reformer_model(node0_url, prompt)
node1_metrics = test_reformer_model(node1_url, prompt)

# Print the metrics for both nodes
print("Node 0 Metrics (Reformer):", node0_metrics)
print("Node 1 Metrics (Reformer):", node1_metrics)
