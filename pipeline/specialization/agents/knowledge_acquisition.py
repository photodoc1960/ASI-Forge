"""
Knowledge Acquisition Agent

Builds domain knowledge corpus by searching for relevant papers
and processing reference documents.
"""

import json
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel


class KnowledgeDocument(BaseModel):
    """A single knowledge document for the RAG system."""
    title: str
    source: str  # URL or file path
    design_insight: str
    experimental_trigger_patterns: str
    background: str
    algorithmic_innovation: str
    implementation_guidance: str
    design_ai_instructions: str
    relevance_score: float = 1.0


class KnowledgeCorpus(BaseModel):
    """Complete knowledge corpus for a domain."""
    documents: List[KnowledgeDocument]
    search_queries: List[str]  # Queries used to find documents
    key_papers: List[str]  # Most important references
    experimental_patterns: List[str]  # Common approaches in domain


KNOWLEDGE_EXTRACTION_PROMPT = """# Knowledge Extraction Task

## Context
You are building a knowledge base for an autonomous research system in the domain of:
{domain_name}

## Domain Understanding
{domain_understanding}

## Source Material
{source_content}

## Your Task
Extract structured knowledge from this source material that will help an AI system:
1. Understand the theoretical foundations
2. Know when to apply specific techniques
3. Implement solutions correctly
4. Avoid common pitfalls

## Required Output Format

Extract and format the knowledge as:

### DESIGN_INSIGHT
The key architectural or design idea from this source. What is the main contribution?

### EXPERIMENTAL_TRIGGER_PATTERNS
When should this insight be applied? What performance signatures or symptoms suggest using this approach?

### BACKGROUND
Historical context and theoretical foundations. Why was this approach developed?

### ALGORITHMIC_INNOVATION
The core algorithm or mechanism. What is the technical contribution?

### IMPLEMENTATION_GUIDANCE
Practical guidance for implementing this. What parameters, settings, or configurations work best?

### DESIGN_AI_INSTRUCTIONS
Specific instructions for an AI system implementing this. What should it do and avoid?

Provide detailed, actionable content for each section."""


SEARCH_QUERIES_PROMPT = """# Search Query Generation Task

## Domain
{domain_name}: {domain_description}

## Key Concepts
{domain_concepts}

## Task
Generate 30-50 search queries to find the most relevant and impactful research papers, documentation, and resources for this domain. Be comprehensive and cover all aspects of the field.

The queries should cover:
1. Foundational papers and techniques
2. State-of-the-art methods
3. Benchmarks and evaluation approaches
4. Implementation best practices
5. Recent breakthroughs

Format: One query per line, designed for academic search (Google Scholar, arXiv, etc.)"""


class KnowledgeAcquisitionAgent:
    """
    Agent that builds domain knowledge corpus through web search
    and document processing.
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the agent.

        Args:
            model: LLM model to use
        """
        self.model = model

    # Supported file extensions for document processing
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.json', '.py', '.tex', '.rst', '.html', '.xml'}

    async def acquire(
        self,
        domain_name: str,
        domain_description: str,
        domain_understanding: Dict[str, Any],
        reference_docs: Optional[List[str]] = None,
        reference_folder: Optional[str] = None,
        max_web_results: int = 50
    ) -> KnowledgeCorpus:
        """
        Acquire knowledge for a domain.

        Args:
            domain_name: Name of the domain
            domain_description: Full description
            domain_understanding: Output from DomainUnderstandingAgent
            reference_docs: Optional list of user-provided document paths
            reference_folder: Optional path to folder containing reference documents
            max_web_results: Maximum web search results to process

        Returns:
            KnowledgeCorpus with processed documents
        """
        documents = []
        search_queries = []

        # Generate search queries
        print("    Generating search queries...", flush=True)
        search_queries = await self._generate_search_queries(
            domain_name,
            domain_description,
            domain_understanding
        )
        print(f"    Generated {len(search_queries)} search queries", flush=True)

        # Process user-provided reference folder
        if reference_folder:
            folder_docs = await self._process_reference_folder(
                reference_folder,
                domain_name,
                domain_understanding
            )
            documents.extend(folder_docs)

        # Process user-provided reference documents (individual files)
        if reference_docs:
            print(f"    Processing {len(reference_docs)} reference documents...", flush=True)
            for i, doc_path in enumerate(reference_docs):
                print(f"      [{i+1}/{len(reference_docs)}] {Path(doc_path).name}", flush=True)
                doc = await self._process_reference_doc(
                    doc_path,
                    domain_name,
                    domain_understanding
                )
                if doc:
                    documents.append(doc)
            print(f"    Extracted knowledge from {len(documents)} documents", flush=True)

        # Search web for additional resources
        print(f"    Searching web ({len(search_queries)} queries, this may take a few minutes)...", flush=True)
        web_docs = await self._search_and_process(
            search_queries,
            domain_name,
            domain_understanding,
            max_results=max_web_results
        )
        documents.extend(web_docs)

        # Extract key papers and patterns
        key_papers = [d.title for d in documents if d.relevance_score > 0.8][:10]
        experimental_patterns = self._extract_patterns(documents)

        return KnowledgeCorpus(
            documents=documents,
            search_queries=search_queries,
            key_papers=key_papers,
            experimental_patterns=experimental_patterns
        )

    async def _generate_search_queries(
        self,
        domain_name: str,
        domain_description: str,
        domain_understanding: Dict[str, Any]
    ) -> List[str]:
        """Generate search queries for the domain."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()

            prompt = SEARCH_QUERIES_PROMPT.format(
                domain_name=domain_name,
                domain_description=domain_description,
                domain_concepts=", ".join(domain_understanding.get("domain_concepts", []))
            )

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Generate search queries, one per line."},
                    {"role": "user", "content": prompt}
                ]
            )

            queries = response.choices[0].message.content.strip().split("\n")
            return [q.strip() for q in queries if q.strip()]

        except Exception as e:
            # Fallback: generate queries from domain understanding and description
            # NOT from domain_name which is just the project title
            print(f"  Warning: LLM query generation failed ({e}), using fallback")

            queries = []

            # Use domain concepts if available (from codebase analysis)
            concepts = domain_understanding.get("domain_concepts", [])
            for concept in concepts[:5]:
                queries.append(f"{concept} neural network architecture")
                queries.append(f"{concept} state of the art")

            # Use research areas if available
            research_areas = domain_understanding.get("research_areas", [])
            for area in research_areas[:3]:
                queries.append(f"{area} survey paper")

            # Extract key terms from description as last resort
            if not queries and domain_description:
                # Extract technical terms (longer words, likely domain-specific)
                words = domain_description.replace(",", " ").replace(".", " ").split()
                technical_terms = [w for w in words if len(w) > 6 and w[0].islower()][:5]
                for term in technical_terms:
                    queries.append(f"{term} machine learning")
                    queries.append(f"{term} deep learning research")

            # Absolute fallback - at least search for something reasonable
            if not queries:
                queries = [
                    "neural architecture search survey",
                    "deep learning best practices",
                    "machine learning benchmarks"
                ]

            return queries[:15]

    async def _process_reference_folder(
        self,
        folder_path: str,
        domain_name: str,
        domain_understanding: Dict[str, Any],
        max_documents: int = 50
    ) -> List[KnowledgeDocument]:
        """
        Process all documents in a folder recursively.

        Args:
            folder_path: Path to the folder containing documents
            domain_name: Name of the domain
            domain_understanding: Output from DomainUnderstandingAgent
            max_documents: Maximum number of documents to process

        Returns:
            List of processed KnowledgeDocument objects
        """
        documents = []
        folder = Path(folder_path)

        if not folder.exists() or not folder.is_dir():
            return documents

        # Find all supported files recursively
        supported_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            supported_files.extend(folder.rglob(f"*{ext}"))

        # Sort by modification time (newest first) and limit
        supported_files = sorted(
            supported_files,
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )[:max_documents]

        print(f"  Processing {len(supported_files)} documents from folder...")

        # Process each file
        for i, file_path in enumerate(supported_files):
            try:
                doc = await self._process_reference_doc(
                    str(file_path),
                    domain_name,
                    domain_understanding
                )
                if doc:
                    documents.append(doc)
                    if (i + 1) % 10 == 0:
                        print(f"    Processed {i + 1}/{len(supported_files)} documents...")
            except Exception as e:
                # Skip files that can't be processed
                continue

        print(f"  Successfully extracted knowledge from {len(documents)} documents.")
        return documents

    async def _process_reference_doc(
        self,
        doc_path: str,
        domain_name: str,
        domain_understanding: Dict[str, Any]
    ) -> Optional[KnowledgeDocument]:
        """Process a user-provided reference document."""
        path = Path(doc_path)

        if not path.exists():
            return None

        # Read document content based on file type
        try:
            content = self._read_document_content(path)
            if not content:
                return None
        except Exception:
            return None

        # Truncate if too long
        if len(content) > 10000:
            content = content[:10000] + "\n...[truncated]..."

        return await self._extract_knowledge(
            source=str(path),
            content=content,
            domain_name=domain_name,
            domain_understanding=domain_understanding
        )

    def _read_document_content(self, path: Path) -> Optional[str]:
        """
        Read content from a document file.

        Handles different file types appropriately.
        """
        suffix = path.suffix.lower()

        try:
            if suffix == '.pdf':
                # Try to extract text from PDF
                return self._extract_pdf_text(path)
            elif suffix == '.json':
                # Format JSON nicely
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return json.dumps(data, indent=2)
            else:
                # Plain text files
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception:
            return None

    def _extract_pdf_text(self, path: Path) -> Optional[str]:
        """Extract text content from a PDF file."""
        try:
            # Try using PyPDF2 if available
            import PyPDF2
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text_parts = []
                for page in reader.pages[:20]:  # Limit to first 20 pages
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                return "\n\n".join(text_parts)
        except ImportError:
            # PyPDF2 not available, try pdfplumber
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    text_parts = []
                    for page in pdf.pages[:20]:  # Limit to first 20 pages
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                    return "\n\n".join(text_parts)
            except ImportError:
                # No PDF library available, return filename as placeholder
                return f"[PDF Document: {path.name}]\n\nNote: Install PyPDF2 or pdfplumber to extract PDF content."
        except Exception:
            return None

    async def _search_and_process(
        self,
        queries: List[str],
        domain_name: str,
        domain_understanding: Dict[str, Any],
        max_results: int = 50
    ) -> List[KnowledgeDocument]:
        """Search web and process results."""
        documents = []

        try:
            # Use web search (if available)
            from openai import AsyncOpenAI

            client = AsyncOpenAI()

            # Process top queries
            query_limit = min(len(queries), 50)
            total_queries = query_limit
            for i, query in enumerate(queries[:query_limit]):
                if (i + 1) % 5 == 1 or i == 0:  # Print progress every 5 queries
                    print(f"      Query {i+1}/{total_queries}: {query[:50]}...", flush=True)
                try:
                    # Web search using GPT-4 with browsing or search API
                    # For now, simulate with knowledge cutoff content
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": f"You are a research expert in {domain_name}. Provide a summary of key research findings, techniques, and best practices based on your knowledge."
                            },
                            {
                                "role": "user",
                                "content": f"Summarize the most important research findings for: {query}\n\nProvide detailed technical content that would help an AI system understand and implement solutions in this area."
                            }
                        ]
                    )

                    content = response.choices[0].message.content

                    doc = await self._extract_knowledge(
                        source=f"Research summary: {query}",
                        content=content,
                        domain_name=domain_name,
                        domain_understanding=domain_understanding
                    )

                    if doc:
                        documents.append(doc)

                except Exception as e:
                    print(f"    Warning: Failed to process query '{query[:50]}...': {e}")
                    continue

                if len(documents) >= max_results:
                    break

        except Exception as e:
            print(f"  Error in web search: {e}")

        print(f"  Knowledge acquisition found {len(documents)} documents")
        return documents

    async def _extract_knowledge(
        self,
        source: str,
        content: str,
        domain_name: str,
        domain_understanding: Dict[str, Any]
    ) -> Optional[KnowledgeDocument]:
        """Extract structured knowledge from content."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()

            prompt = KNOWLEDGE_EXTRACTION_PROMPT.format(
                domain_name=domain_name,
                domain_understanding=json.dumps(domain_understanding, indent=2),
                source_content=content
            )

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract structured knowledge. Respond with ONLY valid JSON (no markdown, no explanation) containing these fields: title, design_insight, experimental_trigger_patterns, background, algorithmic_innovation, implementation_guidance, design_ai_instructions, relevance_score (0.0-1.0)"
                    },
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse JSON from response, handling markdown code blocks
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            data = json.loads(content)

            # Convert any lists/dicts to strings (LLM sometimes returns complex structures)
            for key in data:
                if isinstance(data[key], list):
                    data[key] = "\n".join(str(item) for item in data[key])
                elif isinstance(data[key], dict):
                    data[key] = "\n".join(f"{k}: {v}" for k, v in data[key].items())

            data["source"] = source

            return KnowledgeDocument(**data)

        except Exception as e:
            print(f"    Warning: Failed to extract knowledge from '{source[:40]}...': {e}")
            return None

    def _extract_patterns(self, documents: List[KnowledgeDocument]) -> List[str]:
        """Extract common experimental patterns from documents."""
        patterns = []

        for doc in documents:
            if doc.experimental_trigger_patterns:
                # Split into individual patterns
                for pattern in doc.experimental_trigger_patterns.split("\n"):
                    pattern = pattern.strip()
                    if pattern and pattern not in patterns:
                        patterns.append(pattern)

        return patterns[:20]  # Top 20 patterns

    async def save_corpus(
        self,
        corpus: KnowledgeCorpus,
        output_path: str
    ):
        """
        Save knowledge corpus to disk.

        Args:
            corpus: The knowledge corpus to save
            output_path: Directory to save documents
        """
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each document as a JSON file
        saved_count = 0
        for i, doc in enumerate(corpus.documents):
            try:
                # Sanitize title for filename - remove problematic characters
                safe_title = "".join(c if c.isalnum() or c in "_ -" else "_" for c in doc.title[:30])
                filename = f"doc_{i:03d}_{safe_title}.json"
                filepath = path / filename

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(doc.model_dump(), f, indent=2)
                saved_count += 1
            except Exception as e:
                print(f"    Warning: Failed to save document {i} ({doc.title[:30]}): {e}", flush=True)

        print(f"  Saved {saved_count}/{len(corpus.documents)} knowledge documents to {output_path}", flush=True)

        # Save corpus metadata
        metadata = {
            "search_queries": corpus.search_queries,
            "key_papers": corpus.key_papers,
            "experimental_patterns": corpus.experimental_patterns,
            "document_count": len(corpus.documents),
            "created_at": datetime.utcnow().isoformat()
        }

        with open(path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
