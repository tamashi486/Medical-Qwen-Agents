import uvicorn
import os
import sys

# Add the parent directory to sys.path to allow imports from agent package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    uvicorn.run("agent.api.server:app", host="0.0.0.0", port=8081, reload=True)
