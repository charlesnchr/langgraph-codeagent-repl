import os
import subprocess
import tempfile
from typing import Annotated, Dict, List, Literal

from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
import argparse

# Configuration
REPL_FUNCTION: Literal["langchain", "native"] = "native"
console = Console()

# Initialize Python REPL
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands.",
    func=python_repl.run,
)

def get_graph_figure(
    app, save_path: str | None = None, background_color: str = "f0f2f6"
):
    from langchain_core.runnables.graph import MermaidDrawMethod
    from PIL import Image
    from io import BytesIO

    image_data = app.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API, background_color=background_color
    )

    if save_path:
        with open(save_path, "wb") as f:
            f.write(image_data)

    bytes_io = BytesIO(image_data)
    image = Image.open(bytes_io)

    return image

# State Models
class REPLState(BaseModel):
    """State for the REPL execution process."""

    action: Literal["continue", "complete"] = Field(default="continue")
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)
    code_history: List[str] = Field(default_factory=list)
    code_to_execute: str = Field(default="")
    last_execution_result: str | None = None
    error: str | None = None
    goal: str | None = None
    final_answer: str | None = None


class REPLDecision(BaseModel):
    """Decision on whether to continue execution or complete."""

    action: Literal["continue", "complete"]
    rationale: str


class CodeGeneration(BaseModel):
    """Generated code and explanation."""

    code: str
    explanation: str


class FinalSynthesis(BaseModel):
    """Final answer and explanation."""

    answer: str
    explanation: str


def create_repl_graph():
    """Creates and returns a compiled REPL execution graph."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    builder = StateGraph(REPLState)

    # Define nodes
    def decide_action(state: REPLState) -> Dict:
        """Decide whether to continue execution or complete."""
        console.print(
            Panel(
                f"[bold blue]Goal: {state.goal}\n"
                f"Last Result: {state.last_execution_result or 'No execution yet'}\n"
                f"Steps: {len(state.code_history)}",
                title="Decision Phase",
                border_style="blue",
            )
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a Python coding assistant that decides whether the goal has been achieved.
Choose 'complete' only if the goal is achieved, execution was successful, and output shows expected results.
Choose 'continue' if the goal hasn't been achieved, there were errors, or more calculations are needed.""",
                ),
                (
                    "human",
                    "Goal: {goal}\n\nCurrent state:\nLast result: {last_result}\nHistory:\n{code_history}\n\nShould we continue or complete?",
                ),
            ]
        )

        chain = prompt | llm.with_structured_output(REPLDecision)
        result = chain.invoke(
            {
                "goal": state.goal,
                "last_result": state.last_execution_result or "No execution yet",
                "code_history": "\n".join(
                    f"Step {i + 1}:\n{code}"
                    for i, code in enumerate(state.code_history)
                )
                if state.code_history
                else "No code executed yet",
            }
        )

        console.print(
            Panel(
                f"[yellow]Decision:[/yellow] {result.action}\n"
                f"[yellow]Rationale:[/yellow] {result.rationale}",
                title="Decision Result",
                border_style="blue",
            )
        )

        return {"action": result.action}

    def generate_code(state: REPLState) -> Dict:
        """Generate Python code to address the goal."""
        console.print(
            Panel(
                f"[bold yellow]Goal: {state.goal}\n"
                f"Previous: {state.last_execution_result or 'No previous execution'}",
                title="Code Generation",
                border_style="yellow",
            )
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Generate complete, self-contained Python code to achieve the goal.
Code must be complete, handle errors, and include clear print statements.""",
                ),
                ("human", "Goal: {goal}\n\nPrevious result: {last_result}"),
            ]
        )

        chain = prompt | llm.with_structured_output(CodeGeneration)
        result = chain.invoke(
            {
                "goal": state.goal,
                "last_result": state.last_execution_result or "No previous execution",
            }
        )

        console.print(
            Panel(
                f"[yellow]Explanation:[/yellow] {result.explanation}\n\n"
                f"[yellow]Generated Code:[/yellow]\n{result.code}",
                title="Generation Result",
                border_style="yellow",
            )
        )

        return {"code_to_execute": result.code}

    def execute_code(state: REPLState) -> Dict:
        """Execute code in the REPL."""
        console.print(
            Panel(
                f"[yellow]Executing:[/yellow]\n{state.code_to_execute}",
                title="Execution",
                border_style="green",
            )
        )

        try:
            if REPL_FUNCTION == "langchain":
                result = repl_tool.run(state.code_to_execute)
            else:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False
                ) as tmp_file:
                    tmp_file.write(state.code_to_execute)
                    temp_file_name = tmp_file.name

                try:
                    process_result = subprocess.run(
                        ["python", temp_file_name],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    result = process_result.stdout
                except subprocess.CalledProcessError as e:
                    result = e.stderr
                finally:
                    os.remove(temp_file_name)

            console.print(
                Panel(
                    f"[green]Output:[/green]\n{result}",
                    title="Execution Result",
                    border_style="green",
                )
            )
            return {
                "last_execution_result": result,
                "code_history": state.code_history + [state.code_to_execute],
                "error": None,
            }
        except Exception as e:
            console.print(
                Panel(
                    f"[red]Error:[/red]\n{str(e)}",
                    title="Execution Failed",
                    border_style="red",
                )
            )
            return {
                "last_execution_result": None,
                "code_history": state.code_history + [state.code_to_execute],
                "error": str(e),
            }

    def synthesize_answer(state: REPLState) -> Dict:
        """Generate final answer based on execution results."""
        execution_history = []
        for i, code in enumerate(state.code_history):
            execution_history.extend(
                [
                    f"Step {i + 1}:",
                    f"Code:\n{code}",
                    f"Result: {state.last_execution_result}"
                    if i == len(state.code_history) - 1
                    else "",
                    "",
                ]
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Create a clear, concise answer focusing on WHAT was calculated/computed.
Start with the exact output that answers the query. Be specific and show the actual result.""",
                ),
                (
                    "human",
                    """Goal: {input_goal}
Execution History:
{execution_history}
Final Result: {final_result}""",
                ),
            ]
        )

        chain = prompt | llm.with_structured_output(FinalSynthesis)
        result = chain.invoke(
            {
                "input_goal": state.goal,
                "execution_history": "\n".join(execution_history),
                "final_result": state.last_execution_result,
            }
        )

        console.print(
            Panel(
                f"[yellow]Answer:[/yellow] {result.answer}\n\n"
                f"[yellow]Explanation:[/yellow] {result.explanation}",
                title="Final Result",
                border_style="purple",
            )
        )

        return {"final_answer": result.answer}

    # Add nodes and edges
    builder.add_node("decide_action", decide_action)
    builder.add_node("generate_code", generate_code)
    builder.add_node("execute_code", execute_code)
    builder.add_node("synthesize_answer", synthesize_answer)

    builder.add_edge(START, "decide_action")
    builder.add_conditional_edges(
        "decide_action",
        lambda state, config: state.action,
        {"continue": "generate_code", "complete": "synthesize_answer"},
    )
    builder.add_edge("generate_code", "execute_code")
    builder.add_edge("execute_code", "decide_action")
    builder.add_edge("synthesize_answer", END)

    return builder.compile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute a coding task using LangGraph")
    parser.add_argument(
        "goal", 
        nargs='?',
        default="Calculate the square root of 1.578",
        help="The programming task to execute"
    )
    args = parser.parse_args()
    
    console.print(
        Panel(f"Goal: {args.goal}", title="LangGraph Code Generator", border_style="purple")
    )

    graph = create_repl_graph()
    # Save the graph figure
    get_graph_figure(graph, "workflow_graph.png")

    result = graph.invoke(REPLState(goal=args.goal))