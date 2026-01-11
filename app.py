import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.graph.graph_builder import GraphBuilder
from src.llm.groqllm import GroqLLM
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

@app.post("/generate_blog")
async def generate_blog(request: Request):
    """
    Endpoint to generate a blog post based on a given topic.
    Expects a JSON payload with a 'topic' key.
    """
    data = await request.json()
    topic = data.get("topic")
    if not topic:
        return JSONResponse(content={"error": "Topic is required."}, status_code=400)
    
    groq_llm = GroqLLM()
    llm = groq_llm.get_llm()
    
    graph = GraphBuilder(llm).build_topic_graph().compile()

    output = graph.invoke({"topic": topic})

    return {"response": output}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
