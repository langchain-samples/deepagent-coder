# Deepagent Coder 

This application demonstrates how to use the [deepagents](https://github.com/langchain-ai/deepagents) package
to build a coding assistant that can run as a [LangSmith Deployment](https://docs.langchain.com/langsmith/deployments).

`deepagents` gives you a powerful harness for creating agents that can handle complex, multi-step tasks, such as 
software engineering projects. The LangSmith Deployment platform gives you a production-ready server
and managed infrastructure to run such a complicated agent scalably and reliably as you increase the scope 
of its work. This repo shows how you can combine these two LangChain offerings to design and deploy a software
engineering agent that can do useful work in the unique context of your organization. 

## Key Components

### Memory
The agent learns about the user's preferences for code and saves them to LangSmith Deployment's managed
long-term memory store. The store keeps track of separate preferences 
per [Assistant](https://docs.langchain.com/langsmith/assistants)

### Skills
The skills stored in the codebase are uploaded to the sandbox so the agent can use them and run any associated code.
There are some example skills in this repo.

### Sandbox
This agent can write and test code within a [Daytona sandbox](https://www.daytona.io/).
For simplicity, the agent will return the final code it has written as a message to the end user, but it 
can iterate on and test the code before deciding it is done. 

You can modify this code to instead use [Runloop](https://www.runloop.ai/) or [Modal](https://modal.com/), 
or implement your own sandbox.

## How to Run

You will need the following environment variables:

- An API key for the LLM model of your choice e.g. `OPENAI_API_KEY`
- A [Tavily](https://www.tavily.com/) API key
- A [Daytona](https://www.daytona.io/) API key

You can run this locally using [LangSmith Studio](https://docs.langchain.com/langsmith/studio) or deploy this code
to a [LangSmith Deployment](https://docs.langchain.com/langsmith/deployments)
