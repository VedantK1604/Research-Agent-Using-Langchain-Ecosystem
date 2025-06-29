import os
from typing import List, Dict, TypedDict, Annotated, Optional, Sequence, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_module import ResearchAssistantLangchain
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class ResearchState(TypedDict):
    """State for Research Assistant workflow."""

    query: str
    research_plan: Optional[List[str]]
    current_step: Optional[int]
    retrieved_information: Optional[List[Dict[str, Any]]]
    analysis: Optional[str]
    summary: Optional[str]
    follow_up_questions: Optional[List[str]]
    final_report: Optional[str]
    messages: List[Dict[str, Any]]
    errors: Optional[List[str]]
    status: str


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


def create_research_plan(state: ResearchState) -> ResearchState:
    """Create a research plan based on the initial query."""

    query = state["query"]

    SYSTEM_MESSAGE = SystemMessage(
        content="""
    You are a research planning assistant. Your job is to break down the research query into specific, actionable steps. Create a structured research plan with 3-5 clear steps."""
    )

    HUMAN_MESSAGE = HumanMessage(
        content=f"""Create a research plan for the following query: {query}."""
    )

    response = llm.invoke([SYSTEM_MESSAGE, HUMAN_MESSAGE])
    if isinstance(response.content, str):
        research_plan = response.content.split("\n")
    elif isinstance(response.content, list):
        research_plan = response.content
    else:
        research_plan = []
    research_plan = [
        step for step in research_plan if isinstance(step, str) and step.strip()
    ]

    new_state = state.copy()
    new_state["research_plan"] = research_plan
    new_state["current_step"] = 0
    new_state["status"] = "planning"
    new_state["messages"].append(
        {
            "role": "assistant",
            "content": f"Research plan created: \n" + "\n".join(research_plan),
        }
    )

    return new_state


def execute_research(state: ResearchState) -> ResearchState:
    """Execute the current step in the research plan."""

    new_state = state.copy()

    if not state.get("research_plan") or state.get("current_step") is None:
        errors = new_state.get("errors") or []
        new_state["errors"] = errors + [
            "No research plan is available or current step is not set."
        ]
        new_state["status"] = "error"
        return new_state

    current_step = state["current_step"]
    research_plan = state["research_plan"]

    if (
        research_plan is not None
        and current_step is not None
        and current_step >= len(research_plan)
    ):
        new_state["status"] = "analyzing"
        return new_state

    if research_plan is not None and current_step is not None:
        step_description = research_plan[current_step]
    else:
        new_state["errors"] = (new_state.get("errors") or []) + [
            "Cannot execute research step: research_plan or current_step is None."
        ]
        new_state["status"] = "error"
        return new_state

    SYSTEM_MESSAGE = SystemMessage(
        content="""
    Conver the research step into specific search queries that would help gather information related to this step. Provide 2-3 search queries."""
    )

    HUMAN_MESSAGE = HumanMessage(content=f"Research step: {step_description}")

    response = llm.invoke([SYSTEM_MESSAGE, HUMAN_MESSAGE])
    if isinstance(response.content, str):
        search_queries = response.content.split("\n")
    elif isinstance(response.content, list):
        search_queries = response.content
    else:
        search_queries = []
    search_queries = [
        q.strip() for q in search_queries if isinstance(q, str) and q.strip()
    ]

    retrieved_information = []
    for query in search_queries:
        try:
            search_result = {
                "query": query,
                "results": f"Placeholder results for query: {query}",
            }
            retrieved_information.append(search_result)
        except Exception as e:
            errors = new_state.get("errors") or []
            new_state["errors"] = errors + [
                f"Error executing query '{query}': {str(e)}"
            ]

    if (
        "retrieved_information" not in new_state
        or new_state["retrieved_information"] is None
    ):
        new_state["retrieved_information"] = []

    new_state["retrieved_information"].extend(retrieved_information)
    new_state["current_step"] = current_step + 1
    new_state["status"] = "researching"
    new_state["messages"].append(
        {
            "role": "assistant",
            "content": f"Completed research step {current_step + 1}: {step_description}",
        }
    )

    return new_state


def analyze_information(state: ResearchState) -> ResearchState:
    """Analyze the retrieved information"""

    new_state = state.copy()
    if not state.get("retrieved_information"):
        errors = new_state.get("errors") or []
        new_state["errors"] = errors + ["No information retrieved to analyze."]
        new_state["status"] = "error"
        return new_state

    all_information = ""
    retrieved_information = state["retrieved_information"] or []
    for info in retrieved_information:
        all_information += f"Query: {info['query']}\nResults: {info['results']}\n\n"

    SYSTEM_MESSAGE = SystemMessage(
        content="""
        Analyze the following research information. Identify key insights, patterns, and potential conclusions. Be objective and thorough in you analysis.
        """
    )

    HUMAN_MESSAGE = HumanMessage(content=f"Research information:\n{all_information}")

    response = llm.invoke([SYSTEM_MESSAGE, HUMAN_MESSAGE])
    analysis_content = response.content
    if isinstance(analysis_content, list):
        analysis_content = "\n".join(str(item) for item in analysis_content)
    new_state["analysis"] = analysis_content
    new_state["status"] = "summarizing"
    new_state["messages"].append(
        {"role": "assistant", "content": f"Analysis completed."}
    )

    return new_state


def generate_summary(state: ResearchState) -> ResearchState:
    """Generate the final summary of the research findings."""

    new_state = state.copy()
    if not state["analysis"]:
        errors = new_state["errors"] or []
        new_state["errors"] = errors + ["No analysis available to summarize."]
        new_state["status"] = "error"
        return new_state

    SYSTEM_MESSAGE = SystemMessage(
        content="""
        Create a comprehensive summary of the research findings. Include key points, insights, and conclusions.
        """
    )

    HUMAN_MESSAGE = HumanMessage(content=f"Research Analysis:\n{state['analysis']}")

    response = llm.invoke([SYSTEM_MESSAGE, HUMAN_MESSAGE])

    QUESTION_SYSTEM_MESSAGE = SystemMessage(
        content="""
        Based on the research analysis, suggest 4-5 follow-up question that could help deepen the understang of the topic or explore related areas."""
    )

    QUESTION_HUMAN_MESSAGE = HumanMessage(
        content=f"Research Analysis:\n{state['analysis']}"
    )

    question_response = llm.invoke([QUESTION_SYSTEM_MESSAGE, QUESTION_HUMAN_MESSAGE])
    # follow_up_questions = question_response.content.split("\n")
    # follow_up_questions = [q.strip() for q in follow_up_questions if q.strip()]
    if isinstance(question_response.content, str):
        follow_up_questions = question_response.content.split("\n")
    elif isinstance(question_response.content, list):
        follow_up_questions = question_response.content
    else:
        follow_up_questions = []
    follow_up_questions = [
        q.strip() for q in follow_up_questions if isinstance(q, str) and q.strip()
    ]

    summary_content = response.content
    if isinstance(summary_content, list):
        summary_content = "\n".join(str(item) for item in summary_content)
    new_state["summary"] = summary_content
    new_state["follow_up_questions"] = follow_up_questions
    new_state["status"] = "completed"
    new_state["messages"].append(
        {
            "role": "assistant",
            "content": f"Research completed. Summary: \n\n{summary_content}\n\n Follow-up Questions: \n\n"
            + "\n".join(follow_up_questions),
        }
    )

    return new_state


def should_continue_research(state: ResearchState) -> str:
    """Determine if the research process should continue based on the current state."""

    if state.get("status") == "error":
        return "generate_summary"

    if state.get("status") == "planning":
        return "execute_research"

    if state.get("status") == "researching":
        current_step = state.get("current_step")
        research_plan = state.get("research_plan")
        if current_step is not None and research_plan:
            if isinstance(current_step, int) and current_step < len(research_plan):
                return "execute_research"
            else:
                return "analyze_information"

    if state.get("status") == "analyzing":
        return "generate_summary"

    if state.get("status") == "summarizing" or state.get("status") == "completed":
        return "end"

    return "end"


def build_research_graph():
    """Build research assistant workflow graph."""

    graph = StateGraph(ResearchState)

    graph.add_node("create_research_plan", create_research_plan)
    graph.add_node("execute_research", execute_research)
    graph.add_node("analyze_information", analyze_information)
    graph.add_node("generate_summary", generate_summary)

    graph.add_conditional_edges(
        "create_research_plan",
        should_continue_research,
        {
            "execute_research": "execute_research",
            "generate_summary": "generate_summary",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "execute_research",
        should_continue_research,
        {
            "execute_research": "execute_research",
            "analyze_information": "analyze_information",
            "generate_summary": "generate_summary",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "analyze_information",
        should_continue_research,
        {"generate_summary": "generate_summary", "end": END},
    )

    graph.add_conditional_edges(
        "generate_summary",
        should_continue_research,
        {"end": END},
    )

    graph.set_entry_point("create_research_plan")

    return graph.compile()


class ResearchAssistantLanggraph(ResearchAssistantLangchain):
    """Research Assistant using langgraph."""

    def __init__(self):

        self.graph = build_research_graph()

    def execute_research(self, query: str) -> ResearchState:
        """Initialize the research process and execute the workflow."""

        initial_state = {
            "query": query,
            "research_plan": None,
            "current_step": None,
            "retrieved_information": None,
            "analysis": None,
            "summary": None,
            "follow_up_questions": None,
            "final_report": None,
            "messages": [{"role": "user", "content": query}],
            "errors": [],
            "status": "planning",
        }

        result = self.graph.invoke(initial_state)  # type: ignore

        # Ensure result matches ResearchState type
        if not isinstance(result, dict):
            raise TypeError(
                "Result is not a dictionary and cannot be cast to ResearchState."
            )
        # Optionally, you could validate keys here if needed
        return result  # type: ignore
