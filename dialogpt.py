from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Load DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Function to chat on a node and return performance metrics
def test_dialogpt_model(node_url, step_count=5, retries=3):
    attempt = 0
    chat_history_ids = None

    while attempt < retries:
        try:
            start_time = time.time()  # Start time for the process

            for step in range(step_count):
                # Take user input
                new_user_input = input(f"Node {node_url} >> User:")
                new_user_input_ids = tokenizer.encode(new_user_input + tokenizer.eos_token, return_tensors='pt')

                # Append new user input tokens to the chat history
                bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

                # Generate a response while limiting the total chat history to 1000 tokens
                chat_history_ids = model.generate(
                    bot_input_ids, 
                    max_length=1000, 
                    pad_token_id=tokenizer.eos_token_id
                )

                # Print bot response
                bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
                print(f"Node {node_url} DialoGPT: {bot_response}")

            end_time = time.time()  # End time for the process
            latency = end_time - start_time  # Total time taken for the process

            # Return performance metrics and generated response
            return {
                'node_url': node_url,
                'response': bot_response,
                'latency': latency,
                'response_time': latency,  # For local models, latency equals response time
                'model': 'DialoGPT-small'
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
        'model': 'DialoGPT-small'
    }

# Define the nodes' URLs
node0_url = 'http://127.0.0.1:26657'
node1_url = 'http://127.0.0.1:26667'

# Test the DialoGPT model on both nodes
node0_metrics = test_dialogpt_model(node0_url)
node1_metrics = test_dialogpt_model(node1_url)

# Print the metrics for both nodes
print("Node 0 Metrics (DialoGPT-small):", node0_metrics)
print("Node 1 Metrics (DialoGPT-small):", node1_metrics)
