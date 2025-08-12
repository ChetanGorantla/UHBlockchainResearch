import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Nano Mistral model and tokenizer
model_name = "crumb/nano-mistral"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Define the input prompt
prompt = "What is the answer to 2+2?"

# Function to generate output on a node and return performance metrics
def test_nano_mistral_model(node_url, prompt, retries=3, timeout=10):
    attempt = 0
    while attempt < retries:
        try:
            start_time = time.time()  # Start time for the process

            # Tokenize the input prompt and move it to the model's device
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in dict(inputs).items()}

            # Generate the output with controlled settings
            outputs = model.generate(
                inputs['input_ids'], 
                max_new_tokens=128,        # Limit response length
                temperature=0.1,           # Adjust temperature for varied output
                top_k=20,                  # Set top_k for sampling
                do_sample=True             # Enable sampling
            )

            # Decode the output
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            end_time = time.time()  # End time for the process
            latency = end_time - start_time  # Total time taken for the process
            
            # Return performance metrics and generated response
            return {
                'node_url': node_url,
                'response': generated_text,
                'latency': latency,
                'response_time': latency,  # For local models, latency equals response time
                'model': 'Nano-Mistral'
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
        'model': 'Nano-Mistral'
    }

# Define the nodes' URLs
node0_url = 'http://127.0.0.1:26657'
node1_url = 'http://127.0.0.1:26667'

# Test the Nano Mistral model on both nodes
node0_metrics = test_nano_mistral_model(node0_url, prompt)
node1_metrics = test_nano_mistral_model(node1_url, prompt)

# Print the metrics for both nodes
print("Node 0 Metrics (Nano-Mistral):", node0_metrics)
print("Node 1 Metrics (Nano-Mistral):", node1_metrics)
