from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

class WeatherResponse(BaseModel):
    conditions: str

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = init_chat_model(
    "anthropic:claude-3-7-sonnet-latest",
    temperature=0
)

checkpointer = InMemorySaver()

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    checkpointer=checkpointer,
    response_format=WeatherResponse  
)

# Run the agent
config = {"configurable": {"thread_id": "1"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Chicago"}]},
    config  
)

print(f"response: {response['structured_response']}")
