# Smart Research Assistant

Smart Research Assistant is a powerful tool for conducting research and summarizing information using advanced language models and vector databases. It leverages Langchain and Langgraph for efficient research workflows and provides a user-friendly interface through a Streamlit app.

## Features

- Research query execution with automated planning and analysis
- Document summarization
- URL-based document loading
- Vector database integration (FAISS and Chroma)
- Streamlit-based web interface for easy interaction

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/smart-research-assistant.git
   cd smart-research-assistant
   ```

2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add your API keys:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

## Usage

### Running the Streamlit App

To start the Streamlit app, run:

```bash
streamlit run streamlit_app.py
```

This will launch a web interface where you can:

- Enter research queries and execute the research process
- Summarize text
- Download research outputs

### Using the Research Assistant Programmatically

You can also use the research assistant in your Python code:

```python
from smart_research_assistant.langchain_module import ResearchAssistantLangchain
from smart_research_assistant.langgraph_module import ResearchAssistantLanggraph

# Initialize the research assistant
assistant = ResearchAssistantLangchain()
graph_assistant = ResearchAssistantLanggraph()

# Execute research
query = "What are the latest advancements in AI?"
result = graph_assistant.execute_research(query)

# Summarize a document
document = "Your long text here..."
summary = assistant.summarize_document(document)
```

## File Structure

```
smart_research_assistant/
├── __init__.py
├── langchain_module.py
├── langchain_temp.py
├── langgraph_module.py
├── langraph_temp.py
streamlit_app.py
```

- `langchain_module.py`: Core research assistant functionality using Langchain
- `langgraph_module.py`: Research workflow management using Langgraph
- `streamlit_app.py`: Streamlit interface for the research assistant

## Dependencies

- Python 3.8+
- Streamlit
- Langchain
- Langgraph
- FAISS
- Chroma
- Google Generative AI
- OpenAI
- Other dependencies listed in `requirements.txt`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For any questions or feedback, please open an issue in the GitHub repository.
