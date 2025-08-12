import time
from transformers import pipeline

# Initialize the DistilBERT model with a fill-mask pipeline
bert_filler = pipeline('fill-mask', model='distilbert-base-uncased')

# Define the prompt with a masked token
bert_prompt = 'What is the color of the sky?'

# Function to test DistilBERT on a node with all performance metrics
def test_distilbert_model(node_url, prompt, retries=3, timeout=10):
    attempt = 0
    while attempt < retries:
        try:
            start_time = time.time()  # Start time for the entire process

            # Generate the top prediction for the masked token
            response = bert_filler(prompt, top_k=1)  # Limit to the top 1 prediction

            end_time = time.time()  # End time for the entire process
            latency = end_time - start_time  # Total time taken for the process
            best_response = response[0]['sequence']

            # Return all relevant metrics and the generated response
            return {
                'node_url': node_url,
                'response': best_response,
                'latency': latency,
                'response_time': latency,  # For local models, latency equals response time
                'model': 'distilbert-base-uncased'
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
        'model': 'distilbert-base-uncased'
    }

# Define the nodes' URLs
node0_url = 'http://127.0.0.1:26657'
node1_url = 'http://127.0.0.1:26667'

# Test the DistilBERT model on both nodes
node0_metrics = test_distilbert_model(node0_url, bert_prompt)
node1_metrics = test_distilbert_model(node1_url, bert_prompt)

# Print the metrics for both nodes
print("Node 0 Metrics (DistilBERT):", node0_metrics)
print("Node 1 Metrics (DistilBERT):", node1_metrics)
