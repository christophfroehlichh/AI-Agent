"""
Entry point for the expense agent workflow.
Validates CLI input and starts the LangGraph-based processing pipeline.
"""

import sys
from pathlib import Path
from agents.graph_workflow import run_workflow



def main(pdf_path_str: str) -> None:
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    run_workflow(pdf_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <pfad_zum_pdf>")
        sys.exit(1)

    main(sys.argv[1])
