import os
import base64
from typing import List
import fitz  # PyMuPDF
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from backend.llm import get_vision_model
from backend.database import get_vector_store, clear_vector_store

class DocumentProcessor:
    """Ingests and parses documents, splitting them and uploading to Chroma."""
    
    def __init__(self):
        pass

    def process_file(
        self,
        file_path: str,
        original_filename: str,
        setup_mode: str = "automatic",
        splitter_type: str = "RecursiveCharacterTextSplitter",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[str]:
        """Entry point for parsing local documents and indexing them with selected splitter parameters."""
        ext = os.path.splitext(original_filename)[1].lower()

        if ext == ".pdf":
            documents = self._process_pdf(file_path, original_filename)
        elif ext in {".txt", ".md", ".py", ".json", ".csv"}:
            documents = self._process_text(file_path, original_filename)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        if not documents:
            raise ValueError("No text could be extracted from the file.")

        # Resolve the splitter dynamically
        splitter = self.get_splitter(splitter_type, chunk_size, chunk_overlap)

        # Split documents using resolved splitter (handling header splitters differently)
        if hasattr(splitter, "split_documents"):
            chunks = splitter.split_documents(documents)
        else:
            chunks = []
            for doc in documents:
                split_docs = splitter.split_text(doc.page_content)
                for s_doc in split_docs:
                    # Propagate original document metadata
                    s_doc.metadata.update(doc.metadata)
                    chunks.append(s_doc)

        vector_store = get_vector_store()
        vector_store.add_documents(chunks)
        return [original_filename]

    def get_splitter(self, splitter_type: str, chunk_size: int, chunk_overlap: int):
        """Instantiates the correct LangChain text splitter based on selection."""
        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter,
            CharacterTextSplitter,
            TokenTextSplitter,
            MarkdownHeaderTextSplitter,
            HTMLHeaderTextSplitter,
            PythonCodeTextSplitter,
            LatexTextSplitter,
            SpacyTextSplitter,
            NLTKTextSplitter,
            Language
        )

        s_type = splitter_type.lower().strip()
        if s_type == "charactertextsplitter":
            return CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif s_type == "tokentextsplitter":
            return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif s_type == "markdownheadertextsplitter":
            return MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")])
        elif s_type == "htmlheadertextsplitter":
            return HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")])
        elif s_type == "languagetextsplitter":
            return RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif s_type == "pythoncodetextsplitter":
            return PythonCodeTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif s_type == "latextextsplitter":
            return LatexTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif s_type == "spacytextsplitter":
            return SpacyTextSplitter(pipeline="en_core_web_sm", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif s_type == "nltktextsplitter":
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            return NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _process_text(self, file_path: str, filename: str) -> List[Document]:
        """Read standard text/code documents."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": filename})]
        except Exception as e:
            print(f"Error reading text file: {e}")
            return []

    def _process_pdf(self, file_path: str, filename: str) -> List[Document]:
        """Tries pypdf extraction first; falls back to Ollama Vision OCR."""
        documents = []
        try:
            reader = PdfReader(file_path)
            num_pages = len(reader.pages)
        except Exception as e:
            print(f"Error reading PDF with pypdf: {e}")
            return []

        fitz_doc = None

        for page_num in range(num_pages):
            page_text = ""
            try:
                page_text = reader.pages[page_num].extract_text() or ""
            except Exception as e:
                print(f"pypdf extraction failed on page {page_num + 1}: {e}")

            if page_text.strip():
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={"source": filename, "page": page_num + 1, "method": "pypdf"}
                    )
                )
            else:
                print(f"Page {page_num + 1} of {filename} is empty/scanned. Running Ollama OCR...")
                try:
                    if fitz_doc is None:
                        fitz_doc = fitz.open(file_path)
                    
                    fitz_page = fitz_doc.load_page(page_num)
                    pix = fitz_page.get_pixmap(dpi=150)
                    img_data = pix.tobytes("png")
                    b64 = base64.b64encode(img_data).decode("utf-8")

                    llm = get_vision_model()
                    message = HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": (
                                    "Perform OCR on this image. Extract and transcribe "
                                    "all readable text. Preserve formatting where possible. "
                                    "If no text is visible, reply with nothing."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ]
                    )
                    response = llm.invoke([message])
                    ocr_text = response.content
                    if ocr_text and ocr_text.strip():
                        documents.append(
                            Document(
                                page_content=ocr_text,
                                metadata={"source": filename, "page": page_num + 1, "method": "ollama_ocr"},
                            )
                        )
                except Exception as ocr_err:
                    print(f"Ollama OCR failed on page {page_num + 1}: {ocr_err}")

        if fitz_doc is not None:
            fitz_doc.close()

        return documents

    def clear_database(self):
        """Wipe active database store."""
        clear_vector_store()
