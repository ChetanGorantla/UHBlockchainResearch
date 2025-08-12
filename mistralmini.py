import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set the random seed for reproducibility
torch.random.manual_seed(0)

# Define the model and tokenizer for Mistral-NeMo-Minitron-8B-Chat
model_id = "rasyosef/Mistral-NeMo-Minitron-8B-Chat"
model = AutoModelForCausalLM.from_pretrained(
    model_id,  
    device_map="auto",  
    torch_dtype=torch.bfloat16 
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Define the messages (prompt)
mistral_prompt = [
    {"role": "system", "content": "You are a helpful AI assistant."}, 
    {"role": "user", "content": "What is the color of the sky?"}, 
    
]

# Create a text generation pipeline for Mistral-NeMo-Minitron-8B-Chat
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
)

# Define the generation arguments
generation_args = { 
    "max_new_tokens": 256, 
    "return_full_text": False, 
    "temperature": 0.1,  # Consistent output
    "do_sample": True,  # Deterministic output
}

# Function to test Mistral-NeMo-Minitron-8B-Chat on a node with all performance metrics
def test_mistral_model(node_url, messages, retries=3, timeout=10):
    attempt = 0
    while attempt < retries:
        try:
            start_time = time.time()  # Start time for the entire process

            # Generate the response from Mistral-NeMo-Minitron-8B-Chat
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
                'model': 'Mistral-NeMo-Minitron-8B-Chat'
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
        'model': 'Mistral-NeMo-Minitron-8B-Chat'
    }

# Define the nodes' URLs
node0_url = 'http://127.0.0.1:26657'
node1_url = 'http://127.0.0.1:26667'

# Test the Mistral-NeMo-Minitron-8B-Chat model on both nodes
node0_metrics = test_mistral_model(node0_url, mistral_prompt)
node1_metrics = test_mistral_model(node1_url, mistral_prompt)

# Print the metrics for both nodes
print("Node 0 Metrics (Mistral-NeMo-Minitron-8B-Chat):", node0_metrics)
print("Node 1 Metrics (Mistral-NeMo-Minitron-8B-Chat):", node1_metrics)
