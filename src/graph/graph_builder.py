from langgraph.graph import StateGraph, START, END
from src.llm.groqllm import GroqLLM
from src.states.blog_state import BlogState
from src.nodes.blog_nodes import BlogNode

class GraphBuilder:

    def __init__(self, llm):
        self.llm = llm
        self.graph = StateGraph(BlogState)

    def build_topic_graph(self):
        """
        Build a graph to generate a blog based on a topic.
        """

        self.blog_node_obj = BlogNode(self.llm)

        self.graph.add_node("Title Generation", self.blog_node_obj.title_generation)
        self.graph.add_node("Content Generation", self.blog_node_obj.content_generation)

        self.graph.add_edge(START, "Title Generation")
        self.graph.add_edge("Title Generation", "Content Generation")
        self.graph.add_edge("Content Generation", END)

        return self.graph
    

# Below code is for langsmith langgraph studio
llm = GroqLLM().get_llm()

# Graph
graph_builder = GraphBuilder(llm)
blog_graph = graph_builder.build_topic_graph().compile()