import time
import keras
import keras_nlp
import numpy as np

# Define the model for Gemma from keras-nlp
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma2_instruct_2b_en")

# Define the prompt components
start_of_turn_user = "<start_of_turn>user\n"
start_of_turn_model = "<start_of_turn>model\n"
end_of_turn = "<end_of_turn>\n"

# Define the prompt
prompt = start_of_turn_user + "What is the color of the sky?" + end_of_turn + start_of_turn_model

# Function to test Gemma model on a node with all performance metrics
def test_gemma_model(node_url, prompt, retries=3, timeout=10):
    attempt = 0
    while attempt < retries:
        try:
            start_time = time.time()  # Start time for the entire process

            # Generate the response using Gemma
            response = gemma_lm.generate(prompt, max_length=30)

            end_time = time.time()  # End time for the entire process
            latency = end_time - start_time  # Total time taken for the process
            
            # Return all relevant metrics and the generated response
            return {
                'node_url': node_url,
                'response': response,
                'latency': latency,
                'response_time': latency,  # For local models, latency equals response time
                'model': 'Gemma2-Instruct-27B'
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
        'model': 'Gemma2-Instruct-27B'
    }

# Define the nodes' URLs
node0_url = 'http://127.0.0.1:26657'
node1_url = 'http://127.0.0.1:26667'

# Test the Gemma model on both nodes
node0_metrics = test_gemma_model(node0_url, prompt)
node1_metrics = test_gemma_model(node1_url, prompt)

# Print the metrics for both nodes
print("Node 0 Metrics (Gemma2-Instruct-27B):", node0_metrics)
print("Node 1 Metrics (Gemma2-Instruct-27B):", node1_metrics)
