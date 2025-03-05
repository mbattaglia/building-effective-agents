import os, asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from typing import List, Iterator
from pydantic import BaseModel, Field
from dotenv import load_dotenv

def load_llm() -> BaseLanguageModel:
    load_dotenv()

    max_tokens = int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else 8192
    temperature = float(os.getenv("TEMPERATURE")) if os.getenv("TEMPERATURE") else 0.1
    top_p = float(os.getenv("TOP_P")) if os.getenv("TOP_P") else 0.4
    
    print("Loading LLM...")
    print("Parameters:")
    print(f"max_tokens: {max_tokens}, temperature: {temperature}, top_p: {top_p}")

    # LM Studio
    if os.getenv('USE_LMSTUDIO'):
        model = os.getenv("MODEL") if os.getenv("MODEL") else "qwen2.5-7b-instruct-1m"

        from langchain_openai import ChatOpenAI

        print("Using LMStudio")

        llm = ChatOpenAI(
            base_url="http://localhost:1234/v1", # local LLM
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=None,
            max_retries=2
        )
        return llm
    # OpenAI
    elif os.getenv('OPENAI_API_KEY'):
        print()
        from langchain_openai import ChatOpenAI

        model = os.getenv("MODEL") if os.getenv("MODEL") else "gpt-4o-mini-2024-07-18"

        print(f"Using OpenAI. Model: {model}")

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=None,
            max_retries=2
        )
        return llm
    # Anthropic
    elif os.getenv('ANTHROPIC_API_KEY'):
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("MODEL") if os.getenv("MODEL") else "claude-3-7-latest"

        print(f"Using Anthropic. Model: {model}")

        llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=None,
            max_retries=2
        )
        return llm
    else:
        print("Failed to load LLM")
        raise ValueError("Missing LLM configuration")



##### Classes definition
class FileContent(BaseModel):
    path: str = Field(
        description="Full path to the file"
    )
    content: str = Field(
        description="Raw content of the file"
    )

class Section(BaseModel):
    title: str = Field(
        description="Title of the section",
        min_length = 5,
        max_length = 100
    )
    summary: str = Field(
        description="Comprehensive summary of the file content",
        min_length = 100,
        max_length = 1000
    )

class FileSummary(BaseModel):
    path: str = Field(
        description="Full path to the file"
    )
    sections: List[Section] = Field(
        description="List of sections extracted from the file",
        min_items=1
    )
    category: str = Field(
        description="Category for the field. It can be a newsletter, documentation page, etc."
    )

    # Iterator object
    def __iter__(self) -> Iterator[Section]:
        return iter(self.sections)
    
    # To allow splicing
    def __getitem__(self, index: int) -> Section:
        return self.sections[index]

class Topic(BaseModel):
    name: str = Field(
        description="The name of the topic.",
        min_length = 1,
        max_length = 50
    )
    description: str = Field(
        description="The description of the topic.",
        min_length = 10,
        max_length = 100
    )

class TopicList(BaseModel):
    topics: List[Topic] = Field(
        description="List of main topics identified from the file",
        min_items=1
    )

    # Iterator object
    def __iter__(self) -> Iterator[Topic]:
        return iter(self.topics)
    
    # To allow splicing
    def __getitem__(self, index: int) -> Topic:
        return self.topics[index]

##### Functions definition
async def read_files(file_list: List[str]) -> List[FileContent]:
    """
    Asynchronously loads multiple PDF files and extracts text content.
    
    Args:
        file_list (List[str]): The list of file paths to process.

    Returns:
        List[FileContent]: A list of FileContent objects containing extracted text.
    """
    tasks = [read_file_async(path) for path in file_list]
    return await asyncio.gather(*tasks)

async def read_file_async(path: str) -> FileContent:
    """
    Asynchronously loads a PDF file using LangChain's PyPDFLoader and extracts text content.
    
    Args:
        path (str): The file path to process.

    Returns:
        FileContent: An object containing the extracted text content.
    """
    try:
        if not os.path.exists(path):
            print(f"Warning: File not found {path}")
            return FileContent(path=path, content="")
        
        loader = PyPDFLoader(path)
        pages = await asyncio.to_thread(loader.load)  # Run synchronous PDF loading in a thread
        # the line below works well for small files, but I would suggest a different approach for large file
        content = "\n".join(page.page_content for page in pages)
        return FileContent(path=path, content=content)
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        return FileContent(path=path, content="")

def return_summarization_prompt() -> str:
    """
    Returns the prompt used for summarization
    
    Args:
        none

    Returns:
        str: The prompt used for summarization
    """
    template = """

    <text>{content}</text>

    You are an expert at analyzing and summarizing files.
    Analyze the text contained within the <text> tags and identify:
    - individual sections within the file
    - an overall category for the file (newsletter, documentation page, blogpost, etc)
    For each section, create a title and a summary.
    """
    prompt = PromptTemplate(
        input_variables = ["content"],
        template=template
    )

    return prompt

def summarize_files(file_list: List[FileContent], llm: BaseLanguageModel) -> List[FileSummary]:
    """
    Summarize PDF files using an LLM and retuns a list of
    FileSummary objects with the summarize news content.
    
    Args:
        file_list (list): the list of files to summarize
        llm (BaseLanguageModel): an LLM to use for summarization

    Returns:
        List[FileSummary]: List of FileSummary objects with the summarized news
    """

    structured_llm = llm.with_structured_output(FileSummary)

    file_summary_list = []

    prompt = return_summarization_prompt()

    for file in file_list:
        print(f"processing file {file.path}")
        try:           
            file_summary = structured_llm.invoke(prompt.format(content=file.content))
            file_summary.path = file.path
            
        except Exception as e:
            print(f"Error generating title & summary {str(e)}")
            file_summary = FileSummary(path=file.path, sections=[], category='')


        # create new FileContent object with the extracted text
        file_summary_list.append(file_summary)

    return file_summary_list


def return_main_topic_identification_prompt() -> str:
    """
    Returns the prompt used for main topic identification
    
    Args:
        none

    Returns:
        str: The prompt used for main topic identification
    """
    template = """

    <text>{text}</text>

    You are an expert at analyzing text to identify main topics.
    Analyze the text contained within the <text> tags and identify
    the main topics across all text. There could be multiple main
    topics. For each topic provide:
    - its name
    - a brief description (less than 100 characters)
    """

    prompt = PromptTemplate(
        input_variables = ["text"],
        template=template
    )

    return prompt

def identify_main_topics(summary_list: List[FileSummary], llm: BaseLanguageModel) -> List[Topic]:
    """
    Identify the main topics within a list of FileSummary using an LLM and retuns a list of Topic.
    
    Args:
        summary_list: List[FileSummary]: the list of summaries to from which identify main topics
        llm (BaseLanguageModel): an LLM to use for identifying main topics

    Returns:
        List[Topic]: List of the main topics identified on the summaries
    """

    structured_llm = llm.with_structured_output(TopicList)

    prompt = return_main_topic_identification_prompt()

    text = '\n'.join(
        f"{section.title}: {section.summary}"
        for file_summary in summary_list
        for section in file_summary.sections
    )

    main_topics = structured_llm.invoke(prompt.format(text=text))

    return main_topics
