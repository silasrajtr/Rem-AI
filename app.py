from fastapi import FastAPI, HTTPException  # Import FastAPI and HTTP exception handling
from pydantic import BaseModel  # Import BaseModel for data validation
from langgraph.graph import StateGraph  # Import StateGraph from LangGraph
from langchain_core.messages import HumanMessage  # Import HumanMessage to construct user messages
import uvicorn  # Import Uvicorn for running the FastAPI server
import requests  # Import requests for potential HTTP requests (if needed)
from sample import graph  # Import the graph object from a module (replace with actual module as needed)

app = FastAPI()  # Initialize the FastAPI application

print(graph)  # Output the graph object to verify it is correctly initialized

# Define a data model for the chat request using Pydantic
class GraphRequest(BaseModel):
    thread_id: str  # Unique identifier for the conversation thread
    user_id: str    # Unique identifier for the user
    input_message: str  # The message input from the user

from fastapi.responses import JSONResponse  # Import JSONResponse for custom JSON responses if needed

# Define the chat endpoint to interact with the graph-based chat bot
@app.post("/chat")
def chat_with_bot(request: GraphRequest):
    try:
        # Prepare configuration with thread and user identifiers for the graph processing
        config = {
            "configurable": {
                "thread_id": request.thread_id, 
                "user_id": request.user_id
            }
        }
        
        # Create a list containing the human message constructed from the input
        input_messages = [HumanMessage(content=request.input_message)]
        
        # Initialize a list to store the results from the streamed response
        results = []
        
        # Stream the graph's response in 'values' mode and collect the latest message content from each chunk
        for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
            results.append(chunk["messages"][-1].content)
        
        # Return a JSON response indicating success and include the collected results
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        # In case of any errors, raise an HTTP 500 error with the error details
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn  # Import Uvicorn server for running the app
    uvicorn.run(app, host='127.0.0.1', port=8001)  # Run the FastAPI app on localhost at port 8001
