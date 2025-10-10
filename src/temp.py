from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
  extra_body={},
  model="openai/gpt-oss-20b:free",
  messages=[
    {
      "role": "user",
      "content": 
       "hello",
      
    }
  ]
)
print(completion.choices[0].message.content)