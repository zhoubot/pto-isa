#!/usr/bin/env python3
"""
PTO Task Graph Visualizer

This tool reads a PTO runtime task dump file and generates a visual
representation of the task dependency graph.

Usage:
    python visualize_taskgraph.py <task_dump_file> [output_pdf]

Examples:
    python visualize_taskgraph.py examples/output_arm64/llama7b/llama_layer_dynamic_task_graph.txt
    python visualize_taskgraph.py examples/output_arm64/fused_softmax/dynamic_softmax_task_graph.txt output.pdf

Requirements:
    - graphviz (pip install graphviz)
    - graphviz system package (brew install graphviz on macOS)
"""

import sys
import os
import re
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional


@dataclass
class Task:
    """Represents a task in the task graph."""
    task_id: int
    name: str
    status: str = "UNKNOWN"
    fanin: int = 0
    fanout: List[int] = field(default_factory=list)
    inputs: List[Dict] = field(default_factory=list)
    outputs: List[Dict] = field(default_factory=list)
    buffer_size_bytes: int = 0       # Estimated InCore buffer size without reuse
    buffer_size_with_reuse: int = 0  # Estimated buffer size with reuse optimization


class TaskGraphParser:
    """Parser for PTO runtime task dump files."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.tasks: Dict[int, Task] = {}
        self.edges: List[tuple] = []  # (from_task, to_task)
        self.summary: Dict[str, int] = {}
        
    def parse(self):
        """Parse the task dump file."""
        with open(self.filepath, 'r') as f:
            content = f.read()
        
        # Parse summary section
        self._parse_summary(content)
        
        # Try simple format first
        self._parse_task_table_simple(content)
        
        # If no tasks found, try verbose format
        if not self.tasks:
            self._parse_task_table_verbose(content)
        
        # Parse dependency graph (may add additional edges)
        self._parse_dependency_graph(content)
        
        return self
    
    def _parse_summary(self, content: str):
        """Parse the SUMMARY section."""
        summary_match = re.search(
            r'SUMMARY\s*[-]+\s*(.*?)(?=\n\n|\nTASK TABLE|\n={10,})',
            content, re.DOTALL
        )
        if summary_match:
            summary_text = summary_match.group(1)
            for line in summary_text.strip().split('\n'):
                match = re.match(r'\s*([^:]+):\s*(\d+)', line)
                if match:
                    key = match.group(1).strip()
                    value = int(match.group(2))
                    self.summary[key] = value
    
    def _parse_task_table_simple(self, content: str):
        """Parse the simple TASK TABLE format."""
        # Match lines like: Task 0: rmsnorm_tile         [READY] fanin=0 fanout=[1,2,3]
        pattern = r'Task\s+(\d+):\s+(\w+)\s+\[(\w+)\]\s+fanin=(\d+)\s+fanout=\[([\d,]*)\]'
        
        for match in re.finditer(pattern, content):
            task_id = int(match.group(1))
            name = match.group(2)
            status = match.group(3)
            fanin = int(match.group(4))
            fanout_str = match.group(5)
            
            fanout = []
            if fanout_str:
                fanout = [int(x) for x in fanout_str.split(',')]
            
            self.tasks[task_id] = Task(
                task_id=task_id,
                name=name,
                status=status,
                fanin=fanin,
                fanout=fanout
            )
            
            # Build edges from fanout
            for consumer_id in fanout:
                self.edges.append((task_id, consumer_id))
    
    def _parse_task_table_verbose(self, content: str):
        """Parse the verbose TASK TABLE format with sections."""
        # Parse task blocks that look like:
        # ----------------
        # TASK N
        # ----------------
        #   Function:     name
        #   ...
        #   without_reuse = X bytes
        #   with_reuse    = Y bytes
        #   ...
        #   fanin = X
        #   ...
        #   fanout[] = [1, 2, 3]
        
        # Find all TASK blocks
        task_pattern = r'TASK\s+(\d+)\s*\n-+\n(.*?)(?=\n-+\nTASK|\n={10,}|$)'
        
        for match in re.finditer(task_pattern, content, re.DOTALL):
            task_id = int(match.group(1))
            block = match.group(2)
            
            # Parse function name
            name_match = re.search(r'Function:\s+(\w+)', block)
            name = name_match.group(1) if name_match else f"task_{task_id}"
            
            # Parse buffer sizes
            buf_match = re.search(r'without_reuse\s*=\s*(\d+)\s*bytes', block)
            buffer_size_bytes = int(buf_match.group(1)) if buf_match else 0
            
            reuse_match = re.search(r'with_reuse\s*=\s*(\d+)\s*bytes', block)
            buffer_size_with_reuse = int(reuse_match.group(1)) if reuse_match else 0
            
            # Parse fanin
            fanin_match = re.search(r'fanin\s*=\s*(\d+)', block)
            fanin = int(fanin_match.group(1)) if fanin_match else 0
            
            # Parse fanout list
            fanout = []
            fanout_match = re.search(r'fanout\[\]\s*=\s*\[([\d,\s]*)\]', block)
            if fanout_match:
                fanout_str = fanout_match.group(1).strip()
                if fanout_str:
                    fanout = [int(x.strip()) for x in fanout_str.split(',') if x.strip()]
            
            # Determine status
            status = 'READY' if fanin == 0 else 'WAITING'
            
            self.tasks[task_id] = Task(
                task_id=task_id,
                name=name,
                status=status,
                fanin=fanin,
                fanout=fanout,
                buffer_size_bytes=buffer_size_bytes,
                buffer_size_with_reuse=buffer_size_with_reuse
            )
            
            # Build edges from fanout
            for consumer_id in fanout:
                edge = (task_id, consumer_id)
                if edge not in self.edges:
                    self.edges.append(edge)
    
    def _parse_dependency_graph(self, content: str):
        """Parse the DEPENDENCY GRAPH section for additional edges."""
        graph_match = re.search(
            r'DEPENDENCY GRAPH.*?={10,}\s*(.*?)(?=\n\n={10,}|\nEND OF DUMP|$)',
            content, re.DOTALL
        )
        if graph_match:
            graph_text = graph_match.group(1)
            
            current_task = None
            for line in graph_text.strip().split('\n'):
                # Match: Task 0 (rmsnorm_tile) [READY]
                task_match = re.match(r'\s*Task\s+(\d+)\s+\((\w+)\)\s+\[(\w+)\]', line)
                if task_match:
                    current_task = int(task_match.group(1))
                    name = task_match.group(2)
                    status = task_match.group(3)
                    
                    # Create task if not exists
                    if current_task not in self.tasks:
                        self.tasks[current_task] = Task(
                            task_id=current_task,
                            name=name,
                            status=status
                        )
                    continue
                
                # Match: └──> Task 1 (linear_tile)
                edge_match = re.search(r'──>\s*Task\s+(\d+)', line)
                if edge_match and current_task is not None:
                    consumer_id = int(edge_match.group(1))
                    edge = (current_task, consumer_id)
                    if edge not in self.edges:
                        self.edges.append(edge)


class TaskGraphVisualizer:
    """Visualizes a task graph using Graphviz.
    
    Layout strategy:
    - Independent tasks (same level) are placed in the same COLUMN (vertically)
    - Dependencies flow from LEFT to RIGHT
    - Level is computed as the longest path from any root to that task
    """
    
    # Color scheme for different function types
    COLORS = {
        'rmsnorm': '#E8F5E9',      # Light green
        'linear': '#E3F2FD',        # Light blue
        'softmax': '#FFF3E0',       # Light orange
        'attention': '#FCE4EC',     # Light pink
        'rope': '#F3E5F5',          # Light purple
        'swiglu': '#FFEBEE',        # Light red
        'residual': '#E0F7FA',      # Light cyan
        'rowmax': '#FFF8E1',        # Light amber
        'rowsum': '#FFF8E1',
        'rowexpand': '#FFFDE7',     # Light yellow
        'tile_': '#F5F5F5',         # Light grey
        'elem_': '#F5F5F5',
        'default': '#FFFFFF'        # White
    }
    
    def __init__(self, parser: TaskGraphParser):
        self.parser = parser
        self.tasks = parser.tasks
        self.edges = parser.edges
        self.task_levels = {}  # task_id -> level (computed)
        self._compute_levels()
    
    def _compute_levels(self):
        """Compute the level of each task.
        
        Level = longest path from any root to this task.
        Tasks at the same level are independent and can run in parallel.
        """
        # Build reverse adjacency list (who are my predecessors?)
        predecessors = {tid: [] for tid in self.tasks}
        for from_id, to_id in self.edges:
            if to_id in predecessors:
                predecessors[to_id].append(from_id)
        
        # Initialize levels: root tasks (fanin=0) are at level 0
        for tid, task in self.tasks.items():
            if task.fanin == 0:
                self.task_levels[tid] = 0
        
        # BFS/topological order to compute levels
        # Level of task = max(level of all predecessors) + 1
        changed = True
        iterations = 0
        max_iterations = len(self.tasks) + 10
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for tid in self.tasks:
                if tid in self.task_levels:
                    continue
                
                # Check if all predecessors have levels
                preds = predecessors[tid]
                if not preds:
                    # No predecessors but not in levels? Set to 0
                    self.task_levels[tid] = 0
                    changed = True
                elif all(p in self.task_levels for p in preds):
                    # All predecessors computed - our level is max + 1
                    self.task_levels[tid] = max(self.task_levels[p] for p in preds) + 1
                    changed = True
        
        # Handle any remaining tasks (cycles or disconnected)
        for tid in self.tasks:
            if tid not in self.task_levels:
                self.task_levels[tid] = 0
    
    def _get_color(self, name: str) -> str:
        """Get color based on function name."""
        name_lower = name.lower()
        for prefix, color in self.COLORS.items():
            if prefix in name_lower:
                return color
        return self.COLORS['default']
    
    def _get_shape(self, status: str) -> str:
        """Get node shape based on status."""
        if status == 'READY':
            return 'box'
        elif status == 'COMPLETE':
            return 'ellipse'
        else:
            return 'box'
    
    def generate_dot(self) -> str:
        """Generate DOT format graph description.
        
        Layout:
        - rankdir=LR: dependencies flow left to right
        - Tasks at same level (independent) are in the same column
        """
        lines = []
        lines.append('digraph TaskGraph {')
        lines.append('    rankdir=LR;')  # Left to right for dependencies
        lines.append('    node [fontname="Helvetica", fontsize=9];')
        lines.append('    edge [fontname="Helvetica", fontsize=7];')
        lines.append('    ranksep=0.5;')   # Space between columns (levels)
        lines.append('    nodesep=0.1;')   # Space between rows (parallel tasks)
        lines.append('')
        
        # Graph title
        total_tasks = len(self.tasks)
        max_level = max(self.task_levels.values()) if self.task_levels else 0
        ready_count = sum(1 for t in self.tasks.values() if t.fanin == 0)
        lines.append(f'    labelloc="t";')
        lines.append(f'    label="PTO Task Graph\\n{total_tasks} tasks, {max_level+1} levels, {ready_count} initially parallel";')
        lines.append('    fontsize=14;')
        lines.append('')
        
        # Define nodes
        lines.append('    // Task nodes')
        for task_id, task in sorted(self.tasks.items()):
            color = self._get_color(task.name)
            shape = self._get_shape(task.status)
            
            level = self.task_levels.get(task_id, 0)
            
            # Format buffer size in KB
            buf_kb = task.buffer_size_with_reuse / 1024.0 if task.buffer_size_with_reuse > 0 else 0
            
            # Shorter label for dense graphs, include buffer size in KB
            if buf_kb > 0:
                label = f"T{task_id}\\n{task.name}\\n{buf_kb:.1f}KB"
            else:
                label = f"T{task_id}\\n{task.name}"
            
            # Style based on status
            if task.fanin == 0:  # Ready to execute (level 0)
                style = 'filled,bold'
                penwidth = '2'
                color = '#90EE90'  # Light green for ready tasks
            elif task.status == 'COMPLETE':
                style = 'filled,dashed'
                penwidth = '1'
            else:
                style = 'filled'
                penwidth = '1'
            
            lines.append(
                f'    task{task_id} [label="{label}", shape={shape}, '
                f'style="{style}", fillcolor="{color}", penwidth={penwidth}];'
            )
        
        lines.append('')
        
        # Define edges
        lines.append('    // Dependencies (Producer -> Consumer)')
        for from_id, to_id in self.edges:
            lines.append(f'    task{from_id} -> task{to_id};')
        
        lines.append('')
        
        # Group tasks by level - tasks at same level go in same column
        self._add_level_constraints(lines)
        
        lines.append('}')
        return '\n'.join(lines)
    
    def _add_level_constraints(self, lines: List[str]):
        """Add rank constraints to put tasks at same level in same column.
        
        This ensures:
        - Independent tasks (same level) appear vertically aligned
        - Dependencies flow horizontally (left to right)
        """
        # Group tasks by level
        level_to_tasks: Dict[int, List[int]] = {}
        for tid, level in self.task_levels.items():
            if level not in level_to_tasks:
                level_to_tasks[level] = []
            level_to_tasks[level].append(tid)
        
        # Sort levels and add rank constraints
        for level in sorted(level_to_tasks.keys()):
            tasks = sorted(level_to_tasks[level])
            if tasks:
                lines.append(f'    // Level {level}: {len(tasks)} parallel tasks')
                lines.append('    { rank=same; ' + ' '.join(f'task{tid};' for tid in tasks) + ' }')
    
    def render(self, output_path: str, format: str = 'pdf'):
        """Render the graph to a file."""
        try:
            import graphviz
        except ImportError:
            raise ImportError(
                "graphviz package not installed. Install with `pip install graphviz` "
                "and ensure Graphviz is available on your system."
            )
        
        dot_source = self.generate_dot()
        
        # Remove extension if present
        if output_path.endswith('.pdf'):
            output_path = output_path[:-4]
        elif output_path.endswith('.png'):
            output_path = output_path[:-4]
            format = 'png'
        
        # Create graph and render
        graph = graphviz.Source(dot_source)
        output_file = graph.render(output_path, format=format, cleanup=True)
        
        return output_file
    
    def save_dot(self, output_path: str):
        """Save the DOT source to a file."""
        dot_source = self.generate_dot()
        with open(output_path, 'w') as f:
            f.write(dot_source)
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Visualize PTO task graph from runtime dump file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s task_dump.txt
  %(prog)s task_dump.txt output.pdf
  %(prog)s task_dump.txt output.png --format png
  %(prog)s task_dump.txt --dot-only output.dot
        """
    )
    
    parser.add_argument('input_file', help='Path to task dump file')
    parser.add_argument('output_file', nargs='?', help='Output file path (default: input_file.pdf)')
    parser.add_argument('--format', '-f', choices=['pdf', 'png', 'svg'], default='pdf',
                        help='Output format (default: pdf)')
    parser.add_argument('--dot-only', '-d', action='store_true',
                        help='Only generate DOT file, do not render')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print verbose output')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Determine output path
    if args.output_file:
        output_path = args.output_file
    else:
        base = os.path.splitext(args.input_file)[0]
        output_path = f"{base}.{args.format}"
    
    # Parse task dump
    print(f"Parsing: {args.input_file}")
    graph_parser = TaskGraphParser(args.input_file)
    graph_parser.parse()
    
    if args.verbose:
        print(f"\nSummary:")
        for key, value in graph_parser.summary.items():
            print(f"  {key}: {value}")
        print(f"\nTasks: {len(graph_parser.tasks)}")
        print(f"Edges: {len(graph_parser.edges)}")
    
    # Create visualizer
    visualizer = TaskGraphVisualizer(graph_parser)
    
    # Generate output
    if args.dot_only:
        if not output_path.endswith('.dot'):
            output_path = output_path.rsplit('.', 1)[0] + '.dot'
        result = visualizer.save_dot(output_path)
        print(f"DOT file saved: {result}")
    else:
        result = visualizer.render(output_path, format=args.format)
        print(f"Graph rendered: {result}")
    
    # Print task summary
    print(f"\nTask Graph Summary:")
    print(f"  Total tasks: {len(graph_parser.tasks)}")
    
    ready_tasks = [t for t in graph_parser.tasks.values() if t.status == 'READY']
    if ready_tasks:
        print(f"  Ready tasks: {len(ready_tasks)} ({', '.join(t.name for t in ready_tasks)})")
    
    # Find critical path (longest path)
    print(f"  Dependencies: {len(graph_parser.edges)}")


if __name__ == '__main__':
    main()
