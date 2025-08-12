import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load Flan-T5-small model and tokenizer
model_name = "google/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the prompt
prompt = "What is the color of the sky?"

# Function to generate output on a node and return performance metrics
def test_flan_model(node_url, prompt, retries=3, timeout=10):
    attempt = 0
    while attempt < retries:
        try:
            start_time = time.time()  # Start time for the process

            # Tokenize the input and generate output
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(inputs.input_ids, max_length=50, temperature=0.0, do_sample=False)

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
                'model': 'Flan-T5-small'
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
        'model': 'Flan-T5-small'
    }

# Define the nodes' URLs
node0_url = 'http://127.0.0.1:26657'
node1_url = 'http://127.0.0.1:26667'

# Test the Flan-T5-small model on both nodes
node0_metrics = test_flan_model(node0_url, prompt)
node1_metrics = test_flan_model(node1_url, prompt)

# Print the metrics for both nodes
print("Node 0 Metrics (Flan-T5-small):", node0_metrics)
print("Node 1 Metrics (Flan-T5-small):", node1_metrics)
