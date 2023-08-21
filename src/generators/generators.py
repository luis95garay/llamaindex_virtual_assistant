import openai
import os


class OpenAIGenerator:
    def __init__(
          self,
          model: str = "gpt-3.5-turbo",
          temperature: int = 0
          ) -> None:

        self.model = model
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        return response.choices[0].message["content"]
