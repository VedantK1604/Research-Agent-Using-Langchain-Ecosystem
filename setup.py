from setuptools import setup, find_packages

setup(
    name="smart_research_assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "langchain-community>=0.0.10",
        "langchain-core>=0.1.10",
        "langgraph>0.0.20",
        "openai>1.3.0",
        "chromadb>0.4.18",
        "beautifulsoup4>4.12.2",
        "lxml>4.9.3",
        "python-dotenv>1.0.0",
        "pydantic>=2.5.2",
        "typing-extensions>=4.8.0",
        "tqdm>=4.66.1",
    ],
    entry_point={
        "console_scripts": ["research-assistant=smart_research_assistant.main:main"],
    },
    author="vedant khapekar",
    author_email="vedantkhapekar16@gmail.com",
    description="AI research Agent using Langchain, and Langgraph.",
    keywords="ai, research, agents",
    python_requires=">=3.9",
)
