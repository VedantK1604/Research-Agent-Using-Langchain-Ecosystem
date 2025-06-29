import streamlit as st
import os
import json
from dotenv import load_dotenv

from smart_research_assistant.langchain_module import ResearchAssistantLangchain
from smart_research_assistant.langgraph_module import ResearchAssistantLanggraph

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()


class SmartResearchAssistant:

    def __init__(self) -> None:

        self.langchain_module = ResearchAssistantLangchain()
        self.langgraph_module = ResearchAssistantLanggraph()

    def research(self, query: str):
        """Execute the complete research process."""

        result = self.langgraph_module.execute_research(query=query)
        return {"result": result}

    def generate_summary(self, text: str):
        """Generate the summary of the text."""

        summary = self.langchain_module.summarize_document(text)
        return summary

    def url_loader(self, urls: list):
        pass


st.set_page_config(
    page_title="Smart Research Assistant", page_icon=":robot_face:", layout="wide"
)
st.title("Smart Research Assistant")
st.markdown("Powered by Langchaina and Langgraph.")

if "assistant" not in st.session_state:
    st.session_state.assistant = SmartResearchAssistant()

tab1, tab2 = st.tabs(["Research", "Summarization"])

with tab1:
    st.header("Research")
    query = st.text_area("Enter your research query here: ", height=70)
    if st.button("Start Research"):
        if query:
            with st.spinner("Researching..."):
                try:
                    result = st.session_state.assistant.research(query=query)
                    if "result" in result:
                        data = result["result"]
                    else:
                        data = result

                    if data.get("research_plan"):
                        st.subheader("Research Plan")
                        research_plan = data["research_plan"]
                        if isinstance(research_plan, list):
                            for i, step in enumerate(research_plan):
                                st.markdown(f"**Step {i + 1}:** {step}")

                        if data.get("summary"):
                            st.subheader("Summary")
                            st.markdown(data["summary"])

                        if data.get("follow_up_questions"):
                            st.subheader("Follow-up Questions")
                            follow_up_questions = data["follow_up_questions"]
                            if follow_up_questions is not None:
                                for q in follow_up_questions:
                                    st.markdown(f"- {q}")

                        output = {
                            "query": data.get("query", ""),
                            "reseach_plan": data.get("research_plan", ""),
                            "analysis": data.get("analysis", ""),
                            "summary": data.get("summary", ""),
                            "follow_up_questions": data.get("follow_up_questions", []),
                        }

                        st.download_button(
                            label="Download Research Output",
                            data=json.dumps(output, indent=2),
                            file_name="research_output.json",
                            mime="application/json",
                        )
                except Exception as e:
                    st.error(f"An error occurred during research: {e}")
        else:
            st.warning("Please enter a query to start the research.")


with tab2:
    st.header("Summarization")
    text = st.text_area("Enter the text to summarize: ", height=100)
    if st.button("Generate Summary"):
        if text:
            with st.spinner("Generating summary..."):
                try:
                    summary = st.session_state.assistant.generate_summary(text=text)
                    st.subheader("Summary: ")
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"An error occured during summarization: {e}")
