# Implementing "Building effective agents" from Anthropic

On this repo, I will be implementing the patterns described by Anthropic in the great ["Building effective agents"](https://www.anthropic.com/research/building-effective-agents) publication from Anthropic.

On it, Anthropic describes 3 common patterns for building AI systems:
- Directly calling LLMs: invoking an LLM and getting a response.
- Workflows: systems with pre-defined paths and flow control. "if this/then that". LLMs are just elements within those paths and do not control execution flow.
- Agents: LLMs dynamically control the flow of the execution, potentially invoking other agents, and more.

This is not a comprehensive implementation guide - it's just a set of practical examples that I built as an exercise.

## Use case
To make the examples more practical, I will be implementing a real use case. I will be parsing local pdf files and process them in several ways:
- identifying main topics
- classifying & summarize them them
- etc

I'll be using pdf files I have lying around, ranging from AI newsletters, the ["Building effective agents"](https://www.anthropic.com/research/building-effective-agents) publication from Anthropic that I mentioned earlier, etc.

I will use this to build a local RAG in the future. 
I will process a batch of multiple files at the same time.

## Notebooks
- [Directly calling LLMs](1-DirectCall.ipynb)
- [Workflow: Prompt Chaining](2-Workflows-Prompt-Chaining.ipynb)
- [Workflow: Routing](3-Workflows-Routing.ipynb)

## How to use this repo
Simply open each notebook and run the cells.
Each notebook is auto-contained, including dependencies installation, utils, etc.
In the [Directly calling LLMs](1-DirectCall.ipynb) notebook we define some classes and functions that are later exported into the [utils.py](utils.py) file for convenience.

## Design decisions
There are a couple of design decisions that I took that I think are worth pointing out:

- Forcing schema on LLM outputs: while some may argue that this is not strictly necessary for the "Directly Call LLMs" approach, it is nice to have and even necessary for some of the workflow implementations.
- LangChain: There are three main reasons on why I have chosen to use langchain:
  - Abstract away LLM integrations, messaging formatting and more. For this repo, I will be using a local LLM served using [LM Studio](https://lmstudio.ai/), but others may want to use different LLMs. Using LangChain makes it easier for other people to just change a cell and use the LLM of their choice. 
  - Using `with_structured_output` makes it easy to force schema validation. More info available [here](https://python.langchain.com/docs/how_to/structured_output/).
  - It also abstracts away some of the complexities of building agents.
  - Pydantic: A great library for defining and validating schemas, converting types and more.
- LangGraph: Primarily here for building workflows. While we could manually build workflows with bare Python, LangGraph makes this a lot easier. As a bonus, being able to export worflows diagrams as png files makes explaining this a lot easier.

### A little note about schema validation

LLMs do not inherently return structured output, making it hard to extract information out of their responses. But why do we want to have an structured output? The answer is integrations - what if we want to pass along the response from an LLM response into something else?

### A little note about prompt security
A big disclaimer I want to make upfront: I am not implementing proper input sanitization to prompts. The reasoning is that I have direct control on the input files I am using. I will use Langchain's [PromptTemplates](https://python.langchain.com/docs/concepts/prompt_templates/) to help with this.
