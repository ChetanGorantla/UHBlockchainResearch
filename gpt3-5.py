import requests
import time

# Define the nodes' URLs
node0_url = 'http://127.0.0.1:26657'  # Localhost for node0
node1_url = 'http://127.0.0.1:26667'  # Localhost for node1

# Define the GPT-4 API endpoint and headers
gpt_api_url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": ""
}

# Define the model ID and prompt
gpt_model_id = "gpt-3.5-turbo"
prompt = 'What is the color of the sky?'

# Function to test the model on a node with retries and timeout handling
def test_model(node_url, retries=3, timeout=10):
    attempt = 0
    while attempt < retries:
        try:
            # Start the timer for latency measurement
            start_time = time.time()

            # Make the request to GPT-4
            response = requests.post(
                gpt_api_url,
                headers=headers,
                json={
                    "model": gpt_model_id,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0,
                    "max_tokens": 100,
                    "n": 1,
                    "stop": None
                },
                timeout=timeout
            )

            # End the timer for latency measurement
            end_time = time.time()
            latency = end_time - start_time
            response_time = response.elapsed.total_seconds()

            # Attempt to parse the JSON response from GPT-4
            try:
                response_json = response.json()
                

                # Check if 'choices' key exists in the response
                if 'choices' in response_json:
                    return {
                        'status_code': response.status_code,
                        'response': response_json['choices'][0]['message']['content'],  # Extracting the GPT-4 response content
                        'latency': latency,
                        'response_time': response_time
                    }
                else:
                    return {
                        'status_code': response.status_code,
                        'response': response_json,  # Return the entire response if 'choices' is missing
                        'latency': latency,
                        'response_time': response_time
                    }

            except ValueError:
                print(f"Failed to parse JSON response from {node_url}: {response.text}")
                return {
                    'status_code': response.status_code,
                    'response': response.text,  # Raw text if JSON parsing fails
                    'latency': latency,
                    'response_time': response_time
                }

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Attempt {attempt + 1} of {retries}. Retrying...")
            attempt += 1
            time.sleep(1)  # Brief pause before retrying

    # If all attempts fail
    return {
        'status_code': None,
        'response': None,
        'latency': None,
        'response_time': None
    }


# Test the model on both nodes (though they both use the same GPT-4 API)
node0_metrics = test_model(node0_url)
node1_metrics = test_model(node1_url)

# Print the metrics
print("Node 0 Metrics:", node0_metrics)
print("Node 1 Metrics:", node1_metrics)
