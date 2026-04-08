import uvicorn
from agent.config import AGENT_HOST, AGENT_PORT

if __name__ == "__main__":
    uvicorn.run("agent.api:app", host=AGENT_HOST, port=AGENT_PORT, reload=True)
