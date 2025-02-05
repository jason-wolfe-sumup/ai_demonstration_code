from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper
from langchain.tools import DuckDuckGoSearchRun

# Initialize the language model
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key="<API-KEY>")

# Create tools for the agents
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="Search",
    func=search.run,
    description="Useful for searching information on the internet"
)

# Create memory for agents
researcher_memory = ConversationBufferMemory(memory_key="chat_history")
writer_memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize the researcher agent
researcher = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=researcher_memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=2,
)

# Initialize the writer agent (without search tool, focuses on writing)
writer = initialize_agent(
    tools=[],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=writer_memory,
    verbose=True,
    handle_parsing_errors=True,
)

# Example of using the multi-agent system
def run_multi_agent_task(topic):
    # First, have the researcher gather information
    research_result = researcher.run(
        f"Find key information about {topic}. Be concise and focus on main points."
    )
    
    # Then, have the writer create content based on the research
    final_content = writer.run(
        f"Based on this research: '{research_result}', write a short, engaging paragraph about {topic}."
    )
    
    return final_content
# Example usage
if __name__ == "__main__":
    topic = "artificial intelligence in healthcare"
    result = run_multi_agent_task(topic)
    print("\nFinal Result:")
    print(result)