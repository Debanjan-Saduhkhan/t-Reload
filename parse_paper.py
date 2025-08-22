from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

# Prefer PyMuPDF; fall back to pdfminer.six if import fails
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

PARSED_PATH = Path("analysis/parsed_paper.json")
PARSED_PATH.parent.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "title", "authors", "venue", "year", "abstract", "contributions",
    "method_summary", "assumptions", "equations", "algorithms",
    "datasets", "metrics", "baselines", "hyperparams"
]


def parse_pdf(path: Union[str, Path]) -> Dict[str, any]:
    """
    Parse a PDF research paper and extract key information.
    
    Args:
        path: Path to the PDF file
        
    Returns:
        Dictionary containing extracted information
    """
    p = Path(path)
    text = ""
    
    if PYMUPDF_AVAILABLE:
        with fitz.open(p) as doc:
            text = "\n\n".join(page.get_text("text") for page in doc)
    else:
        from pdfminer.high_level import extract_text
        text = extract_text(str(p))

    def find(pattern: str) -> Optional[Union[List[str], str]]:
        """Find matches for a regex pattern in the text."""
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        return matches if matches else None
    
    def find_algorithms_detailed() -> Dict[str, str]:
        """Find detailed algorithm information."""
        algorithms = {}
        
        # Look for algorithm sections with more context
        algorithm_patterns = [
            r"(?s)(Algorithm\s+\d+[:\s].*?)(?=\n\s*(?:Algorithm\s+\d+|$))",
            r"(?s)(Algorithm\s+\d+[:\s].*?)(?=\n\s*[0-9]+\s*[A-Z])",
            r"(?s)(Algorithm\s+\d+[:\s].*?)(?=\n\s*[A-Z][a-z]+)"
        ]
        
        for pattern in algorithm_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if matches:
                for match in matches:
                    # Extract algorithm number and content
                    algo_match = re.match(r"Algorithm\s+(\d+)[:\s]*(.*)", match.strip(), re.IGNORECASE)
                    if algo_match:
                        algo_num = algo_match.group(1)
                        algo_content = algo_match.group(2).strip()
                        algorithms[f"Algorithm_{algo_num}"] = algo_content
        
        # If the above patterns don't work, try to extract the specific algorithms we know exist
        if not algorithms:
            # Algorithm 1: Offline training
            algo1_pattern = r"(?s)Algorithm\s+1\s+Offline\s+training(.*?)(?=\n\s*(?:Algorithm\s+2|Figure|Table|The overall))"
            algo1_match = re.search(algo1_pattern, text, re.IGNORECASE)
            if algo1_match:
                algorithms["Algorithm_1"] = "Offline training" + algo1_match.group(1).strip()
            
            # Algorithm 2: Online training  
            algo2_pattern = r"(?s)Algorithm\s+2\s+Online\s+training(.*?)(?=\n\s*(?:The overall|Figure|Table|$))"
            algo2_match = re.search(algo2_pattern, text, re.IGNORECASE)
            if algo2_match:
                algorithms["Algorithm_2"] = "Online training" + algo2_match.group(1).strip()
        
        return algorithms

    # Extract key information using regex patterns
    data = {
        "title": (find(r"^(.+?)\n") or [p.stem])[0],
        "authors": find(r"(?m)^(?:Authors?|By):?\s*(.+)$"),
        "venue": find(r"(?i)(?:conference|journal|workshop|venue):?\s*(.+)"),
        "year": find(r"\b(20[12]\d)\b"),
        "abstract": (find(r"(?s)abstract\s*[:\n]\s*(.+?)\n\s*\b(?:1\.|introduction)\b") or [None])[0],
        "contributions": find(r"(?s)contributions?\s*[:\n]\s*(.+?)\n\s*\b(?:method|approach|model)\b"),
        "method_summary": find(r"(?s)(?:method|approach|algorithm)\s*[:\n]\s*(.+?)\n\s*\b(?:experiments|evaluation|results)\b"),
        "assumptions": find(r"(?i)assumptions?\s*[:\n]\s*(.+)"),
        "equations": find(r"\$[^$]+\$|\\\[[\s\S]*?\\\]"),
        "algorithms": find_algorithms_detailed(),
        "datasets": find(r"(?i)dataset[s]?:?\s*(.*)"),
        "metrics": find(r"(?i)metric[s]?:?\s*(.*)"),
        "baselines": find(r"(?i)baseline[s]?:?\s*(.*)"),
        "hyperparams": {}
    }

    # Save parsed data
    with open(PARSED_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return data


if __name__ == "__main__":
    # Test parsing
    pdf_path = "../AIMLSystems_tReload.pdf"
    if Path(pdf_path).exists():
        result = parse_pdf(pdf_path)
        print("Parsed paper information:")
        print(json.dumps(result, indent=2))
    else:
        print(f"PDF file not found: {pdf_path}")
