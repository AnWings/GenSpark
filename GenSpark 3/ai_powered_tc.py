import openai
import logging

# Setup logging for observations
logging.basicConfig(filename='ai_evaluation.log', level=logging.INFO)

# Function to generate responses from the AI model
def generate_response(prompt, temperature=0.7, max_tokens=100, top_p=1.0):
    try:
        response = openai.Completion.create(
            engine="gpt-4",  # or "gpt-3.5-turbo" based on your model choice
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return f"Error: {e}"

# Function to test with various types of prompts
def test_prompts():
    prompts = {
        "Creative Story": "Write a short story about a dragon who befriends a young knight.",
        "Summarize Climate Change": "Summarize the following paragraph: Climate change is one of the most pressing issues of our time. It refers to the long-term changes in temperature, precipitation, and other atmospheric conditions that have been primarily driven by human activities such as burning fossil fuels and deforestation.",
        "Explain Quantum Physics": "Explain quantum physics in simple terms for a 10-year-old."
    }

    # Parameters to experiment with
    temperatures = [0.2, 0.5, 0.7, 1.0]
    max_tokens_values = [50, 100, 150, 200]
    top_p_values = [0.8, 1.0]

    # Iterate over each prompt and different parameters
    for prompt_name, prompt_text in prompts.items():
        for temp in temperatures:
            for max_tokens in max_tokens_values:
                for top_p in top_p_values:
                    print(f"Testing {prompt_name} with parameters: temperature={temp}, max_tokens={max_tokens}, top_p={top_p}")
                    response = generate_response(prompt_text, temperature=temp, max_tokens=max_tokens, top_p=top_p)
                    log_observation(prompt_name, temp, max_tokens, top_p, response)
                    print(response)
                    print("-" * 50)

# Function to log observations
def log_observation(prompt_name, temp, max_tokens, top_p, response):
    log_message = f"Prompt: {prompt_name}\n" \
                  f"Parameters: temperature={temp}, max_tokens={max_tokens}, top_p={top_p}\n" \
                  f"Response: {response}\n" \
                  f"{'=' * 50}\n"
    logging.info(log_message)

# Main function to run the tests
def main():
    # Ensure you've set your OpenAI API key before running
    openai.api_key = 'your-api-key-here'  # Replace with your actual API key

    print("Running AI Model Evaluation...\n")
    test_prompts()

if __name__ == "__main__":
    main()
