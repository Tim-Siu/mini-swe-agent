#!/usr/bin/env python3
"""
Simple trajectory inspector for browsing agent conversation trajectories.

More information about the usage: [bold green] https://mini-swe-agent.com/latest/usage/inspector/ [/bold green].
"""

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static

from minisweagent.agents.interactive_textual import _messages_to_steps

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def _extract_reasoning_content(message: dict) -> str | None:
    """Extract reasoning/thinking content from message extras."""
    extra = message.get("extra", {})
    if not extra or "response" not in extra:
        return None
    
    response = extra.get("response", {})
    choices = response.get("choices", [])
    if not choices:
        return None
    
    msg_data = choices[0].get("message", {})
    
    if "reasoning_content" in msg_data:
        reasoning = msg_data["reasoning_content"]
        if reasoning:
            return str(reasoning)
    
    provider_fields = msg_data.get("provider_specific_fields", {})
    if "reasoning_content" in provider_fields:
        reasoning = provider_fields["reasoning_content"]
        if reasoning:
            return str(reasoning)
    
    return None


def _extract_tool_calls(message: dict) -> list[dict] | None:
    """Extract tool calls from message extras."""
    extra = message.get("extra", {})
    if not extra or "response" not in extra:
        return None
    
    response = extra.get("response", {})
    choices = response.get("choices", [])
    if not choices:
        return None
    
    msg_data = choices[0].get("message", {})
    tool_calls = msg_data.get("tool_calls")
    
    if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
        return tool_calls
    
    return None


def _format_tool_calls_markdown(tool_calls: list[dict]) -> str:
    """Format tool calls as markdown."""
    lines = []
    for tc in tool_calls:
        if tc.get("type") == "function":
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args = func.get("arguments", "")
            lines.append(f"**Tool Call:** `{name}`")
            lines.append("")
            lines.append("```json")
            lines.append(args)
            lines.append("```")
        else:
            lines.append(f"**Tool Call:** `{tc}`")
    return "\n".join(lines)


def _format_message_content_markdown(content: str) -> str:
    """Format message content as markdown, handling code blocks."""
    # If content already has code blocks, preserve them
    if "```" in content:
        return content
    
    # For output/observation content, wrap in code block if it looks like terminal output
    lines = content.split("\n")
    if len(lines) > 1 and any(tag in content for tag in ["<returncode>", "<output>", "<warning>"]):
        # Clean up XML tags for readability
        formatted = content
        formatted = formatted.replace("<returncode>", "**Return Code:** ")
        formatted = formatted.replace("</returncode>", "")
        formatted = formatted.replace("<output>", "\n```\n")
        formatted = formatted.replace("</output>", "\n```\n")
        formatted = formatted.replace("<warning>", "\n> ⚠️ **Warning:** ")
        formatted = formatted.replace("</warning>", "\n")
        formatted = formatted.replace("<output_head>", "\n```\n")
        formatted = formatted.replace("</output_head>", "\n```\n...")
        formatted = formatted.replace("<elided_chars>", "\n> *")
        formatted = formatted.replace("</elided_chars>", "*\n")
        formatted = formatted.replace("<output_tail>", "\n```\n")
        formatted = formatted.replace("</output_tail>", "\n```")
        return formatted
    
    return content


def _generate_markdown(trajectory_file: Path, messages: list[dict], steps: list[list[dict]]) -> str:
    """Generate markdown representation of a trajectory."""
    lines = []
    
    # Header
    lines.append(f"# Trajectory: {trajectory_file.stem}")
    lines.append("")
    lines.append(f"**Source:** `{trajectory_file}`")
    lines.append(f"**Total Steps:** {len(steps)}")
    lines.append(f"**Total Messages:** {len(messages)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Process each step
    for step_idx, step in enumerate(steps):
        lines.append(f"## Step {step_idx + 1}")
        lines.append("")
        
        for message in step:
            role = message.get("role", "unknown")
            
            # Get content
            if isinstance(message.get("content"), list):
                content = "\n".join([item.get("text", "") for item in message["content"]])
            else:
                content = str(message.get("content", ""))
            
            # Format based on role
            if role == "system":
                lines.append("<details>")
                lines.append("<summary><b>🖥️ SYSTEM</b></summary>")
                lines.append("")
                lines.append("```")
                lines.append(content[:2000] + ("..." if len(content) > 2000 else ""))
                lines.append("```")
                lines.append("")
                lines.append("</details>")
                
            elif role == "user":
                lines.append("<details>")
                lines.append("<summary><b>👤 USER</b></summary>")
                lines.append("")
                lines.append(_format_message_content_markdown(content))
                lines.append("")
                lines.append("</details>")
                
            elif role == "assistant":
                lines.append("### 🤖 ASSISTANT")
                lines.append("")
                
                # Extract and display reasoning
                reasoning = _extract_reasoning_content(message)
                if reasoning and reasoning.strip():
                    lines.append("<details>")
                    lines.append("<summary><b>🧠 Thinking Process</b></summary>")
                    lines.append("")
                    lines.append("```")
                    lines.append(reasoning)
                    lines.append("```")
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")
                
                # Extract and display tool calls
                tool_calls = _extract_tool_calls(message)
                if tool_calls:
                    lines.append("<details open>")
                    lines.append("<summary><b>⚡ Tool Calls</b></summary>")
                    lines.append("")
                    lines.append(_format_tool_calls_markdown(tool_calls))
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")
                
                # Main content
                if content.strip():
                    lines.append("**Response:**")
                    lines.append("")
                    lines.append(_format_message_content_markdown(content))
                    lines.append("")
            
            else:
                lines.append(f"### {role.upper()}")
                lines.append("")
                lines.append(content)
                lines.append("")
            
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


def _export_trajectory_to_markdown(trajectory_file: Path, output_dir: Path | None = None) -> Path:
    """Export a single trajectory to markdown."""
    # Load trajectory
    data = json.loads(trajectory_file.read_text())
    
    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict) and "messages" in data:
        messages = data["messages"]
    else:
        raise ValueError("Unrecognized trajectory format")
    
    steps = _messages_to_steps(messages)
    
    # Generate markdown
    markdown = _generate_markdown(trajectory_file, messages, steps)
    
    # Determine output path
    if output_dir:
        output_path = output_dir / f"{trajectory_file.stem}.md"
    else:
        output_path = trajectory_file.with_suffix(".md")
    
    output_path.write_text(markdown)
    return output_path


class TrajectoryInspector(App):
    BINDINGS = [
        Binding("right,l", "next_step", "Step++"),
        Binding("left,h", "previous_step", "Step--"),
        Binding("0", "first_step", "Step=0"),
        Binding("$", "last_step", "Step=-1"),
        Binding("j,down", "scroll_down", "Scroll down"),
        Binding("k,up", "scroll_up", "Scroll up"),
        Binding("L", "next_trajectory", "Next trajectory"),
        Binding("H", "previous_trajectory", "Previous trajectory"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, trajectory_files: list[Path]):
        css_path = os.environ.get(
            "MSWEA_INSPECTOR_STYLE_PATH", str(Path(__file__).parent.parent.parent / "config" / "mini.tcss")
        )
        self.__class__.CSS = Path(css_path).read_text()

        super().__init__()
        self.trajectory_files = trajectory_files
        self._i_trajectory = 0
        self._i_step = 0
        self.messages = []
        self.steps = []

        if trajectory_files:
            self._load_current_trajectory()

    # --- Basics ---

    @property
    def i_step(self) -> int:
        """Current step index."""
        return self._i_step

    @i_step.setter
    def i_step(self, value: int) -> None:
        """Set current step index, automatically clamping to valid bounds."""
        if value != self._i_step and self.n_steps > 0:
            self._i_step = max(0, min(value, self.n_steps - 1))
            self.query_one(VerticalScroll).scroll_to(y=0, animate=False)
            self.update_content()

    @property
    def n_steps(self) -> int:
        """Number of steps in current trajectory."""
        return len(self.steps)

    @property
    def i_trajectory(self) -> int:
        """Current trajectory index."""
        return self._i_trajectory

    @i_trajectory.setter
    def i_trajectory(self, value: int) -> None:
        """Set current trajectory index, automatically clamping to valid bounds."""
        if value != self._i_trajectory and self.n_trajectories > 0:
            self._i_trajectory = max(0, min(value, self.n_trajectories - 1))
            self._load_current_trajectory()
            self.query_one(VerticalScroll).scroll_to(y=0, animate=False)
            self.update_content()

    @property
    def n_trajectories(self) -> int:
        """Number of trajectory files."""
        return len(self.trajectory_files)

    def _load_current_trajectory(self) -> None:
        """Load the currently selected trajectory file."""
        if not self.trajectory_files:
            self.messages = []
            self.steps = []
            return

        trajectory_file = self.trajectory_files[self.i_trajectory]
        try:
            data = json.loads(trajectory_file.read_text())

            if isinstance(data, list):
                self.messages = data
            elif isinstance(data, dict) and "messages" in data:
                self.messages = data["messages"]
            else:
                raise ValueError("Unrecognized trajectory format")

            self.steps = _messages_to_steps(self.messages)
            self._i_step = 0
        except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
            self.messages = []
            self.steps = []
            self.notify(f"Error loading {trajectory_file.name}: {e}", severity="error")

    @property
    def current_trajectory_name(self) -> str:
        """Get the name of the current trajectory file."""
        if not self.trajectory_files:
            return "No trajectories"
        return self.trajectory_files[self.i_trajectory].name

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with VerticalScroll():
                yield Vertical(id="content")
        yield Footer()

    def on_mount(self) -> None:
        self.update_content()

    def _extract_reasoning_content(self, message: dict) -> str | None:
        """Extract reasoning/thinking content from message extras.
        
        Handles GLM models and other providers that store reasoning content
        in different locations within the response metadata.
        """
        extra = message.get("extra", {})
        if not extra or "response" not in extra:
            return None
        
        response = extra.get("response", {})
        choices = response.get("choices", [])
        if not choices:
            return None
        
        msg_data = choices[0].get("message", {})
        
        # Try different possible locations for reasoning content
        # GLM models: reasoning_content field
        if "reasoning_content" in msg_data:
            reasoning = msg_data["reasoning_content"]
            if reasoning:
                return str(reasoning)
        
        # Some models may store it in provider_specific_fields
        provider_fields = msg_data.get("provider_specific_fields", {})
        if "reasoning_content" in provider_fields:
            reasoning = provider_fields["reasoning_content"]
            if reasoning:
                return str(reasoning)
        
        return None

    def _extract_tool_calls(self, message: dict) -> list[dict] | None:
        """Extract tool calls from message extras.
        
        Handles OpenAI-compatible tool format used by GLM and other models.
        """
        extra = message.get("extra", {})
        if not extra or "response" not in extra:
            return None
        
        response = extra.get("response", {})
        choices = response.get("choices", [])
        if not choices:
            return None
        
        msg_data = choices[0].get("message", {})
        tool_calls = msg_data.get("tool_calls")
        
        if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
            return tool_calls
        
        return None

    def _format_tool_calls(self, tool_calls: list[dict]) -> str:
        """Format tool calls for display."""
        lines = []
        for tc in tool_calls:
            if tc.get("type") == "function":
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                args = func.get("arguments", "")
                lines.append(f"🔧 Tool Call: {name}")
                lines.append(f"   Arguments: {args}")
            else:
                # Handle other tool call formats
                lines.append(f"🔧 Tool Call: {tc}")
        return "\n".join(lines)

    def update_content(self) -> None:
        """Update the displayed content."""
        container = self.query_one("#content", Vertical)
        container.remove_children()

        if not self.steps:
            container.mount(Static("No trajectory loaded or empty trajectory"))
            self.title = "Trajectory Inspector - No Data"
            return

        for message in self.steps[self.i_step]:
            # Extract main content
            if isinstance(message["content"], list):
                content_str = "\n".join([item["text"] for item in message["content"]])
            else:
                content_str = str(message["content"])
            
            message_container = Vertical(classes="message-container")
            container.mount(message_container)
            role = message["role"].replace("assistant", "mini-swe-agent")
            message_container.mount(Static(role.upper(), classes="message-header"))
            
            # For assistant messages, try to extract reasoning and tool calls
            if message.get("role") == "assistant":
                # Display reasoning content if available
                reasoning = self._extract_reasoning_content(message)
                if reasoning and reasoning.strip():
                    reasoning_header = Static("🧠 THINKING", classes="message-header reasoning-header")
                    message_container.mount(reasoning_header)
                    reasoning_text = Text(reasoning, no_wrap=False)
                    message_container.mount(Static(reasoning_text, classes="message-content reasoning-content"))
                
                # Display tool calls if available
                tool_calls = self._extract_tool_calls(message)
                if tool_calls:
                    tool_calls_str = self._format_tool_calls(tool_calls)
                    tool_header = Static("⚡ TOOL CALLS", classes="message-header tool-header")
                    message_container.mount(tool_header)
                    tool_text = Text(tool_calls_str, no_wrap=False)
                    message_container.mount(Static(tool_text, classes="message-content tool-content"))
            
            # Display main content
            message_container.mount(Static(Text(content_str, no_wrap=False), classes="message-content"))

        self.title = (
            f"Trajectory {self.i_trajectory + 1}/{self.n_trajectories} - "
            f"{self.current_trajectory_name} - "
            f"Step {self.i_step + 1}/{self.n_steps}"
        )

    # --- Navigation actions ---

    def action_next_step(self) -> None:
        self.i_step += 1

    def action_previous_step(self) -> None:
        self.i_step -= 1

    def action_first_step(self) -> None:
        self.i_step = 0

    def action_last_step(self) -> None:
        self.i_step = self.n_steps - 1

    def action_next_trajectory(self) -> None:
        self.i_trajectory += 1

    def action_previous_trajectory(self) -> None:
        self.i_trajectory -= 1

    def action_scroll_down(self) -> None:
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y + 15)

    def action_scroll_up(self) -> None:
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y - 15)


@app.command(help=__doc__)
def main(
    path: str = typer.Argument(".", help="Directory to search for trajectory files or specific trajectory file"),
    export_markdown: bool = typer.Option(
        False, 
        "--export-markdown", "-m",
        help="Export trajectories to markdown files instead of launching TUI"
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir", "-o",
        help="Output directory for markdown files (default: same as trajectory location)"
    ),
) -> None:
    path_obj = Path(path)

    if path_obj.is_file():
        trajectory_files = [path_obj]
    elif path_obj.is_dir():
        trajectory_files = sorted(path_obj.rglob("*.traj.json"))
        if not trajectory_files:
            raise typer.BadParameter(f"No trajectory files found in '{path}'")
    else:
        raise typer.BadParameter(f"Error: Path '{path}' does not exist")

    if export_markdown:
        # Export mode: generate markdown files
        console = Console()
        console.print(f"[bold green]Exporting {len(trajectory_files)} trajectory(s) to markdown...[/bold green]")
        
        # Create output directory if specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        exported = []
        for traj_file in trajectory_files:
            try:
                output_path = _export_trajectory_to_markdown(traj_file, output_dir)
                exported.append(output_path)
                console.print(f"  ✓ [cyan]{traj_file.name}[/cyan] → [green]{output_path}[/green]")
            except Exception as e:
                console.print(f"  ✗ [red]{traj_file.name}: {e}[/red]")
        
        console.print(f"\n[bold green]Exported {len(exported)} markdown file(s)[/bold green]")
        if output_dir:
            console.print(f"Output directory: [cyan]{output_dir.absolute()}[/cyan]")
    else:
        # Interactive TUI mode
        inspector = TrajectoryInspector(trajectory_files)
        inspector.run()


if __name__ == "__main__":
    app()
