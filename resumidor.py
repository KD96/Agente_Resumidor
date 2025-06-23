#!/usr/bin/env python3
"""
PDF Summarization Script using LangChain Map-Reduce Approach
Handles large PDF documents (100-150 pages) efficiently
"""

import os
import asyncio
import time
from typing import List, Dict, Any
from pathlib import Path
import tiktoken

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# For progress tracking
from tqdm.asyncio import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFSummarizer:
    """
    A comprehensive PDF summarization tool using LangChain's map-reduce approach
    Designed to handle large PDF documents efficiently with rate limit handling
    """
    
    def __init__(self, openai_api_key: str = None, model_name: str = "gpt-3.5-turbo", 
                 max_tokens_per_chunk: int = 3000, chunk_overlap: int = 200,
                 max_retries: int = 5, retry_delay: float = 1.0):
        """
        Initialize the PDF summarizer
        
        Args:
            openai_api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model_name: Name of the OpenAI model to use
            max_tokens_per_chunk: Maximum tokens per text chunk
            chunk_overlap: Overlap between chunks for context preservation
            max_retries: Maximum number of retries for rate limit errors
            retry_delay: Base delay between retries (will use exponential backoff)
        """
        # Set up OpenAI API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY env variable")
        
        # Store retry settings
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize the LLM with rate limiting considerations
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.3,
            max_tokens=1000,
            request_timeout=60,  # Increase timeout
            max_retries=max_retries  # Built-in retry handling
        )
        
        # Initialize text splitter for handling large documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens_per_chunk,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize tokenizer for counting tokens
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # Define prompts for map and reduce phases
        self.map_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            You are an expert at summarizing documents. Please provide a comprehensive summary of the following text.
            Focus on the main points, key findings, and important details. Make the summary informative and well-structured.
            
            Text to summarize:
            {text}
            
            SUMMARY:
            """
        )
        
        self.reduce_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            You are an expert at creating final comprehensive summaries from multiple partial summaries.
            Please combine the following summaries into one cohesive, well-structured final summary.
            Organize the information logically and ensure all important points are covered.
            
            Partial summaries to combine:
            {text}
            
            FINAL COMPREHENSIVE SUMMARY:
            """
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))
    
    def _handle_rate_limit_with_retry(self, func, *args, **kwargs):
        """
        Execute a function with exponential backoff retry for rate limits
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function execution
        """
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_message = str(e).lower()
                if "rate limit" in error_message or "429" in str(e):
                    if attempt < self.max_retries - 1:
                        # Calculate delay with exponential backoff
                        delay = self.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limit hit. Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Max retries reached. Rate limit error: {e}")
                        raise
                else:
                    # Non-rate-limit error, re-raise immediately
                    raise
        
        # This should never be reached, but just in case
        raise Exception("Unexpected error in retry logic")
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load PDF document and split into manageable chunks
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Loading PDF: {pdf_path}")
        
        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        logger.info(f"Loaded {len(documents)} pages from PDF")
        
        # Split documents into chunks
        split_documents = self.text_splitter.split_documents(documents)
        
        logger.info(f"Split into {len(split_documents)} chunks")
        
        # Log token counts
        total_tokens = sum(self._count_tokens(doc.page_content) for doc in split_documents)
        logger.info(f"Total tokens in document: {total_tokens:,}")
        
        return split_documents
    
    def create_map_reduce_chain(self):
        """
        Create the map-reduce chain for summarization
        
        Returns:
            MapReduceDocumentsChain object
        """
        # Map chain - summarize individual chunks
        map_chain = LLMChain(llm=self.llm, prompt=self.map_prompt)
        
        # Reduce chain - combine summaries
        reduce_chain = LLMChain(llm=self.llm, prompt=self.reduce_prompt)
        
        # Combine documents chain for the reduce step
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name="text"
        )
        
        # Reduce documents chain
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000,  # Maximum tokens for the reduce step
        )
        
        # Map reduce documents chain
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="text",
            return_intermediate_steps=True,  # To see intermediate summaries
        )
        
        return map_reduce_chain
    
    async def summarize_pdf_async(self, pdf_path: str, save_intermediate: bool = False) -> Dict[str, Any]:
        """
        Summarize PDF asynchronously using map-reduce approach
        
        Args:
            pdf_path: Path to the PDF file
            save_intermediate: Whether to save intermediate summaries
            
        Returns:
            Dictionary containing the final summary and metadata
        """
        # Load and split PDF
        documents = self.load_pdf(pdf_path)
        
        if not documents:
            raise ValueError("No content found in PDF")
        
        # Create map-reduce chain
        chain = self.create_map_reduce_chain()
        
        logger.info("Starting map-reduce summarization...")
        
        # Run the chain
        result = await chain.ainvoke({
            "input_documents": documents
        })
        
        # Prepare output
        summary_result = {
            "final_summary": result["output_text"],
            "intermediate_steps": result.get("intermediate_steps", []),
            "document_count": len(documents),
            "total_pages": len(set(doc.metadata.get("page", 0) for doc in documents)),
            "source_file": pdf_path
        }
        
        # Save intermediate summaries if requested
        if save_intermediate:
            self._save_intermediate_summaries(pdf_path, summary_result)
        
        return summary_result
    
    def summarize_pdf(self, pdf_path: str, save_intermediate: bool = False) -> Dict[str, Any]:
        """
        Synchronous wrapper for PDF summarization with rate limit handling
        
        Args:
            pdf_path: Path to the PDF file
            save_intermediate: Whether to save intermediate summaries
            
        Returns:
            Dictionary containing the final summary and metadata
        """
        return self._handle_rate_limit_with_retry(
            asyncio.run, 
            self.summarize_pdf_async(pdf_path, save_intermediate)
        )
    
    def _save_intermediate_summaries(self, pdf_path: str, result: Dict[str, Any]):
        """Save intermediate summaries to files"""
        base_name = Path(pdf_path).stem
        output_dir = Path(pdf_path).parent / f"{base_name}_summaries"
        output_dir.mkdir(exist_ok=True)
        
        # Save intermediate summaries
        for i, summary in enumerate(result["intermediate_steps"]):
            with open(output_dir / f"intermediate_summary_{i+1}.txt", "w", encoding="utf-8") as f:
                f.write(summary)
        
        # Save final summary
        with open(output_dir / "final_summary.txt", "w", encoding="utf-8") as f:
            f.write(result["final_summary"])
        
        logger.info(f"Summaries saved to: {output_dir}")
    
    def batch_summarize(self, pdf_directory: str, output_file: str = None, delay_between_files: float = 60.0):
        """
        Summarize multiple PDFs in a directory with rate limit management
        
        Args:
            pdf_directory: Directory containing PDF files
            output_file: Optional output file for batch results
            delay_between_files: Delay in seconds between processing files to avoid rate limits
        """
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_directory}")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = {}
        
        for i, pdf_file in enumerate(pdf_files):
            try:
                logger.info(f"Processing: {pdf_file.name} ({i+1}/{len(pdf_files)})")
                result = self.summarize_pdf(str(pdf_file))
                results[pdf_file.name] = result
                logger.info(f"Successfully processed: {pdf_file.name}")
                
                # Add delay between files to avoid rate limits (except for the last file)
                if i < len(pdf_files) - 1:
                    logger.info(f"Waiting {delay_between_files}s before next file to avoid rate limits...")
                    time.sleep(delay_between_files)
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                results[pdf_file.name] = {"error": str(e)}
        
        # Save batch results if output file specified
        if output_file:
            self._save_batch_results(results, output_file)
        
        return results
    
    def _save_batch_results(self, results: Dict, output_file: str):
        """Save batch processing results to file"""
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Batch results saved to: {output_file}")


def main():
    """
    Main function to demonstrate usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Summarize large PDF documents using LangChain")
    parser.add_argument("pdf_path", help="Path to the PDF file to summarize")
    parser.add_argument("--api-key", help="OpenAI API key (optional if set in environment)")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate summaries")
    parser.add_argument("--output", help="Output file for the summary")
    parser.add_argument("--batch", action="store_true", help="Process all PDFs in the directory")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retries for rate limit errors")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="Base delay between retries (seconds)")
    parser.add_argument("--batch-delay", type=float, default=60.0, help="Delay between files in batch mode (seconds)")
    
    args = parser.parse_args()
    
    try:
        # Initialize summarizer
        summarizer = PDFSummarizer(
            openai_api_key=args.api_key,
            model_name=args.model,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay
        )
        
        if args.batch:
            # Batch processing
            results = summarizer.batch_summarize(args.pdf_path, args.output, args.batch_delay)
            print(f"Processed {len(results)} files. Check logs for details.")
        else:
            # Single file processing
            result = summarizer.summarize_pdf(args.pdf_path, args.save_intermediate)
            
            print("\n" + "="*80)
            print("PDF SUMMARIZATION COMPLETE")
            print("="*80)
            print(f"Source: {result['source_file']}")
            print(f"Pages: {result['total_pages']}")
            print(f"Chunks processed: {result['document_count']}")
            print("\nFINAL SUMMARY:")
            print("-" * 50)
            print(result['final_summary'])
            
            # Save to output file if specified
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(f"PDF Summary: {Path(args.pdf_path).name}\n")
                    f.write("="*80 + "\n\n")
                    f.write(result['final_summary'])
                print(f"\nSummary saved to: {args.output}")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
