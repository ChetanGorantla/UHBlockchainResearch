import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

# Define the model and tokenizer for Phi-3.5-mini-instruct
model_name = "microsoft/Phi-3.5-mini-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the messages (prompt)
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What is 2+2?"},
]

# Create a text generation pipeline for Phi-3.5-mini-instruct
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Define the generation arguments
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

# Function to test Phi-3.5-mini-instruct on a node with all performance metrics
def test_phi_model(node_url, messages, retries=3, timeout=10):
    attempt = 0
    while attempt < retries:
        try:
            start_time = time.time()  # Start time for the entire process

            # Generate the response from Phi-3.5-mini-instruct
            output = pipe(messages, **generation_args)
            generated_text = output[0]['generated_text']

            end_time = time.time()  # End time for the entire process
            latency = end_time - start_time  # Total time taken for the process
            
            # Return all relevant metrics and the generated response
            return {
                'node_url': node_url,
                'response': generated_text,
                'latency': latency,
                'response_time': latency,  # For local models, latency equals response time
                'model': 'Phi-3.5-mini-instruct'
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
        'model': 'Phi-3.5-mini-instruct'
    }

# Define the nodes' URLs
node0_url = 'http://127.0.0.1:26657'
node1_url = 'http://127.0.0.1:26667'

# Test the Phi-3.5-mini-instruct model on both nodes
node0_metrics = test_phi_model(node0_url, messages)
node1_metrics = test_phi_model(node1_url, messages)

# Print the metrics for both nodes
print("Node 0 Metrics (Phi-3.5-mini-instruct):", node0_metrics)
print("Node 1 Metrics (Phi-3.5-mini-instruct):", node1_metrics)
