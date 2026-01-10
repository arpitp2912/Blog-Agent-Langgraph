from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

class GroqLLM:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        # self.model = ChatGroq(api_key=self.api_key, model="openai/gpt-oss-120b")
        self.model = ChatGroq(api_key=self.api_key, model="llama-3.3-70b-versatile")

    def get_llm(self):
        """
        Return the Groq LLM model instance.
        """
        return self.model

    def generate_response(self, prompt):
        """
        Generate a response from the Groq LLM model.
        :param prompt: The input prompt string.
        """
        try:
            response = self.model.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Could not generate response."