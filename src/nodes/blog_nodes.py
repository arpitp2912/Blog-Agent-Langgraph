from ..states.blog_state import BlogState

class BlogNode:
    """
    A node to generate content by the blogging agent.
    """

    def __init__(self, llm):
        self.llm = llm

    def title_generation(self, state:BlogState):
        """
        Generate a blog title based on the topic in the state.
        """
        if "topic" not in state:
            raise ValueError("State must contain a 'topic' key.")
        else:
            topic = state["topic"]
            prompt = """Role: You are an expert blog writer and SEO specialist.

Task: Generate one perfect blog title based on the user-provided topic: {topic}.

Critical Guidelines for the Title:

Catchy & Engaging: Use power words, curiosity gaps, or clear benefit-driven language to immediately grab attention.

Concise: Aim for 6-12 words. It must be easily scannable.

SEO Friendly:

Naturally include a primary keyword or key phrase related to the {topic}.

Place important keywords near the front.

Ensure it matches likely user search intent (informational, how-to, listicle, etc.).

Format: Provide only the title. Do not use quotation marks, colons, or introductory text.
"""
            
            system_message = prompt.format(topic=topic)

            response = self.llm.invoke(system_message)

            return {"blog": {"title": response.content}}
        
    def content_generation(self, state:BlogState):
        """
        Generate blog content based on the title in the state.
        """
        if "blog" not in state or "title" not in state["blog"]:
            raise ValueError("State must contain a 'blog' key with a 'title'.")
        else:
            title = state["blog"]["title"]
            prompt = """You are an expert blog writer that is SEO optimized. Use markdown formatting.
            Your task is to generate a detailed and engaging blog post based on the title: {title}.
            The content should be well-structured, informative, and provide value to the readers."""
            
            system_message = prompt.format(title=title)

            response = self.llm.invoke(system_message)

            return {"blog": {"content": response.content, "title": state["blog"]["title"]}}