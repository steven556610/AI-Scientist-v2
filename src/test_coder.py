import openai

client = openai.OpenAI(
    api_key="none",
    base_url="http://localhost:8004/v1"
)

print("\nTesting local/qwen3-coder-30b-a3b-instruct integration...")

try:
    response = client.chat.completions.create(
        model="local/qwen3-coder-30b-a3b-instruct",
        messages=[
            {"role": "user", "content": "Write a python function to print hello world. Return ONLY the code."}
        ],
        temperature=0.1,
        max_tokens=100
    )

    print("\nSUCCESS!")
    print(f"Model Response: {response.choices[0].message.content}")

except Exception as e:
    print(f"\nERROR: Failed to connect or generate text: {e}")
