import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import re
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class ArxivPaperCrawler:
    def __init__(self, delay=1):
        self.delay = delay
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def crawl_paper(self, url):
        """Crawls all sections of the paper in a structured way."""
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            paper_data = {
                "url": url,
                "crawled_at": datetime.now().isoformat(),
                "title": self._extract_title(soup),
                "authors": self._extract_authors(soup),
                "abstract": self._extract_abstract(soup),
                "sections": self._extract_sections(soup),
                "references": self._extract_references(soup),
                "equations": self._extract_equations(soup),
                "figures": self._extract_figures(soup),
            }

            return paper_data

        except Exception as e:
            print(f"Crawling error: {e}")
            return None


    def _extract_title(self, soup):
        """Extract title."""
        title_selectors = ["h1.ltx_title", "h1", ".ltx_title", "title"]

        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        return "Title not found"

    def _extract_authors(self, soup):
        """Extract author information."""
        authors = []

        # Find authors in multiple ways
        author_containers = soup.find_all("div", class_="ltx_authors") or []

        for container in author_containers:
            author_elems = container.find_all("span", class_="ltx_personname")
            for author in author_elems:
                authors.append(author.get_text().strip())

        # Alternative method
        if not authors:
            author_links = soup.find_all("a", href=re.compile(r"author:"))
            authors = [link.get_text().strip() for link in author_links]

        return authors

    def _extract_abstract(self, soup):
        """Extract abstract."""
        abstract_selectors = ["div.ltx_abstract", ".abstract", "blockquote"]

        for selector in abstract_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        return "Abstract not found"

    def _extract_sections(self, soup):
        """Extract content by section."""
        sections = []

        # Find section headers
        section_headers = soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6"], class_=re.compile(r"ltx_title")
        )

        for i, header in enumerate(section_headers):
            section = {
                "level": int(header.name[1]),
                "title": header.get_text().strip(),
                "content": "",
            }

            # Collect content until the next section
            current = header.find_next_sibling()
            content_parts = []

            while current and (
                i + 1 >= len(section_headers) or current != section_headers[i + 1]
            ):
                if current and current.name not in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    content_parts.append(current.get_text().strip())
                current = current.find_next_sibling() if current else None

                if not current:
                    break

            section["content"] = "\n".join(content_parts)
            sections.append(section)

        return sections

    def _extract_references(self, soup):
        """Extract references."""
        references = []

        # Find reference section
        ref_section = soup.find("section", class_="ltx_bibliography") or soup.find(
            "div", class_="references"
        )

        if ref_section:
            ref_items = ref_section.find_all("li") or ref_section.find_all(
                "div", class_="ltx_bibitem"
            )
            for item in ref_items:
                references.append(item.get_text().strip())

        return references

    def _extract_equations(self, soup):
        """Extract equations."""
        equations = []

        # Find LaTeX equations
        eq_elements = soup.find_all(
            ["div", "span"], class_=re.compile(r"ltx_equation|math")
        )

        for eq in eq_elements:
            equations.append(
                {"type": eq.get("class", []), "content": eq.get_text().strip()}
            )

        return equations

    def _extract_figures(self, soup):
        """Extract figure/table information."""
        figures = []

        # Find figure captions
        fig_elements = soup.find_all("figure") + soup.find_all(
            "div", class_="ltx_figure"
        )

        for fig in fig_elements:
            caption = fig.find("figcaption") or fig.find("div", class_="ltx_caption")
            img = fig.find("img")

            figure_data = {
                "caption": caption.get_text().strip() if caption else "",
                "src": img.get("src") if img else "",
            }
            figures.append(figure_data)

        return figures


class PaperRAGProcessor:
    def __init__(
        self, api_key: str = None, model_name: str = "text-embedding-3-small"
    ):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "OpenAI API key is required. Please set the 'OPENAI_API_KEY' environment variable or pass it as an argument."
            )

        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.embedding_dim = self._get_embedding_dimension()

    def _get_embedding_dimension(self):
        # Dynamically check embedding dimension
        try:
            response = self.client.embeddings.create(
                input=["test"], model=self.model_name
            )
            return len(response.data[0].embedding)
        except Exception as e:
            print(f"Error checking embedding dimension: {e}. Using default value 1536.")
            return 1536

    def _get_openai_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Gets embeddings for a list of texts using the OpenAI embedding model."""
        # The OpenAI API cannot handle empty strings, so we replace them.
        texts = [text.replace("\n", " ").strip() or " " for text in texts]

        try:
            response = self.client.embeddings.create(input=texts, model=self.model_name)
            embeddings = [np.array(embedding.embedding) for embedding in response.data]
            return embeddings
        except Exception as e:
            print(f"Error generating OpenAI embeddings: {e}")
            return [np.zeros(self.embedding_dim) for _ in texts]

    def process_paper(self, paper_data):
        """Processes paper data for RAG."""
        chunks = []

        # 1. Process abstract separately
        if paper_data.get("abstract") and paper_data["abstract"] != "Abstract not found":
            chunks.append(
                {
                    "text": paper_data["abstract"],
                    "type": "abstract",
                    "metadata": {"section": "abstract", "importance": "high"},
                }
            )

        # 2. Structural chunking by section
        for section in paper_data.get("sections", []):
            if section["content"].strip():  # Only if content is not empty
                section_chunks = self._chunk_section(section)
                chunks.extend(section_chunks)

        # 3. Generate embeddings (batch processing)
        texts_to_embed = [chunk["text"] for chunk in chunks]
        if texts_to_embed:
            embeddings = self._get_openai_embeddings(texts_to_embed)
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    chunk["embedding"] = embeddings[i]
                else:
                    # If embedding generation failed
                    chunk["embedding"] = np.zeros(self.embedding_dim)

        return [chunk for chunk in chunks if "embedding" in chunk]

    def _chunk_section(self, section):
        """Chunks a section semantically."""
        full_text = f"{section['title']}\n{section['content']}"

        # First split by paragraph
        paragraphs = full_text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:  # Skip empty paragraphs
                continue

            test_chunk = current_chunk + "\n\n" + para if current_chunk else para

            # Based on 200 words
            if len(test_chunk.split()) <= 200:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(
                        {
                            "text": current_chunk.strip(),
                            "type": "section",
                            "metadata": {"section": section["title"]},
                        }
                    )
                current_chunk = para

        if current_chunk:
            chunks.append(
                {
                    "text": current_chunk.strip(),
                    "type": "section",
                    "metadata": {"section": section["title"]},
                }
            )

        return chunks


class PaperQASystem:
    """
    A system that takes a paper URL, crawls, processes, and performs Q&A.
    """

    def __init__(self, openai_api_key: str = None):
        """
        Args:
            openai_api_key (str, optional): OpenAI API key.
                                            If None, it's loaded from the 'OPENAI_API_KEY' environment variable.
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")

        self.crawler = ArxivPaperCrawler()
        self.processor = PaperRAGProcessor(api_key=self.api_key)
        self.client = openai.OpenAI(api_key=self.api_key)
        self.chunks: List[Dict[str, Any]] = []
        self.paper_info: Dict[str, Any] = {}

    def load_paper(self, url: str) -> bool:
        """
        Crawls and processes a paper from a URL to prepare for Q&A.

        Args:
            url (str): The ArXiv URL of the paper to crawl.

        Returns:
            bool: True if successful, False otherwise.
        """
        print(f"1. Crawling paper from URL: {url}")
        paper_data = self.crawler.crawl_paper(url)
        if not paper_data:
            print("Failed to crawl paper.")
            return False

        self.paper_info = {
            "title": paper_data.get("title", "No Title Found"),
            "authors": paper_data.get("authors", []),
            "url": url,
        }
        print(f"Paper Title: {self.paper_info['title']}")

        print("2. Chunking text and generating embeddings...")
        self.chunks = self.processor.process_paper(paper_data)
        print(f"Total {len(self.chunks)} chunks were generated and embedded.")

        return True

    def answer_question(self, question: str, top_k: int = 3) -> str:
        """
        Answers a question based on the loaded paper.

        Args:
            question (str): The user's question.
            top_k (int, optional): The number of chunks to use for the answer. Defaults to 3.

        Returns:
            str: The generated answer.
        """
        if not self.chunks or not self.paper_info:
            return "Error: Paper not loaded. Please call `load_paper()` first."

        print("3. Starting to generate answer for the question...")

        print("Embedding question...")
        question_embedding = self.processor._get_openai_embeddings([question])[0]
        if question_embedding.sum() == 0:
            return "Failed to generate question embedding."

        print("Searching for relevant paper content...")
        similarities = [
            (
                chunk,
                cosine_similarity(
                    question_embedding.reshape(1, -1), chunk["embedding"].reshape(1, -1)
                )[0][0],
            )
            for chunk in self.chunks
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [item[0] for item in similarities[:top_k]]

        print("Requesting answer generation from GPT-4o-mini model...")
        context = "\n\n---\n\n".join([chunk["text"] for chunk in top_chunks])

        prompt = f"""You are an expert AI researcher who answers questions based on the provided paper content.
The following is some content from the paper '{self.paper_info['title']}'.

--- Start of Paper Content ---
{context}
--- End of Paper Content ---

Based on this content, please provide a clear and detailed answer to the following question in English.
Please use KaTeX syntax for mathematical formulas. Enclose inline formulas with `$` and block formulas with `$$`.

Question: {question}
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert AI researcher."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            print("Answer generation complete.")
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred during answer generation: {e}"


def main():
    try:
        # API key is automatically loaded from environment variables.
        qa_system = PaperQASystem()
    except ValueError as e:
        print(e)
        return

    url = "https://ar5iv.labs.arxiv.org/html/2201.12091"
    if not qa_system.load_paper(url):
        return

    questions = [
        "What is the main contribution of this paper?",
        "Please explain the experimental methodology.",
        "What are the limitations mentioned in the conclusion?",
        "What is the name of the model used in this paper?",
    ]

    for q in questions:
        print(f"\n[Question] {q}")
        answer = qa_system.answer_question(q)
        print(f"\n[Answer]\n{answer}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
