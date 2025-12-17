#!/usr/bin/env python3
"""
Privacy Policy AI Server - Standalone FastAPI Application
Run this script directly with: python server.py
Uses TinyLLama-based chatbot only (lightweight for 4GB VRAM)
"""

import torch
import requests
import gc
import time
import json
import asyncio
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, AsyncGenerator
import uvicorn
from datasets import Dataset
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# Firebase imports
from firebase_config import get_firebase_db, check_cache, save_to_cache

# ============================================================================
# SCRAPER FUNCTIONS
# ============================================================================

def scrape_webpage(url, use_javascript=False, wait_time=10, headless=True):
    """
    Scrapes content from a webpage.
    
    Parameters:
    -----------
    url : str
        The URL of the webpage to scrape
    use_javascript : bool, optional
        If True, uses Selenium to render JavaScript-heavy pages (default: False)
        Set to True for pages that load content dynamically via JavaScript
    wait_time : int, optional
        Maximum time to wait for page elements to load in seconds (default: 10)
    headless : bool, optional
        If True, runs browser in headless mode (no visible window). Only used when use_javascript=True (default: True)
    
    Returns:
    --------
    str : The HTML content of the webpage
    None : If scraping fails
    
    Examples:
    ---------
    # Scrape a static webpage
    >>> content = scrape_webpage("https://example.com")
    >>> print(content[:100])
    
    # Scrape a JavaScript-rendered webpage
    >>> content = scrape_webpage("https://example.com", use_javascript=True, wait_time=15)
    >>> print(content[:100])
    """
    
    if not use_javascript:
        # Use requests for static HTML pages (faster and simpler)
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error scraping webpage with requests: {e}")
            return None
    
    else:
        # Use Selenium for JavaScript-rendered pages
        try:
            chrome_options = Options()
            if headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait to allow JavaScript to execute
            time.sleep(2)
            
            # Get page source
            page_content = driver.page_source
            driver.quit()
            
            return page_content
        
        except Exception as e:
            print(f"Error scraping webpage with Selenium: {e}")
            try:
                driver.quit()
            except:
                pass
            return None


def extract_text_from_html(html_content):
    """
    Extracts plain text from HTML content.
    
    Parameters:
    -----------
    html_content : str
        The HTML content to parse
    
    Returns:
    --------
    str : Plain text extracted from the HTML
    
    Example:
    --------
    >>> html = "<html><body><h1>Hello</h1><p>World</p></body></html>"
    >>> text = extract_text_from_html(html)
    >>> print(text)
    """
    if not html_content:
        return None
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        print(f"Error extracting text from HTML: {e}")
        return None


def extract_clean_content(html_content, article_tag=True, remove_extra_spans=True):
    """
    Extracts clean, readable content from HTML by removing redundant tags and formatting.
    Useful for pages with excessive nested HTML like privacy policies.
    
    Parameters:
    -----------
    html_content : str
        The HTML content to parse
    article_tag : bool, optional
        If True, tries to extract content from article or main content container first (default: True)
    remove_extra_spans : bool, optional
        If True, removes redundant nested span tags that don't add value (default: True)
    
    Returns:
    --------
    str : Clean, formatted text content
    None : If extraction fails
    
    Example:
    --------
    >>> html = "<html><body><article><span><span><p>Privacy Policy</p></span></span></article></body></html>"
    >>> content = extract_clean_content(html)
    >>> print(content)
    """
    if not html_content:
        return None
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style, noscript, and meta elements
        for element in soup(["script", "style", "noscript", "meta", "link", "iframe"]):
            element.decompose()
        
        # Try to find main content area with better specificity
        content_area = None
        if article_tag:
            # Priority order for content detection:
            # 1. article tag
            # 2. main tag
            # 3. div with specific content classes
            # 4. Find the largest section/article
            # 5. Find the largest div with significant text
            
            content_area = soup.find('article')
            
            if not content_area:
                content_area = soup.find('main')
            
            if not content_area:
                # Look for more specific content containers
                for class_name in ['page-content', 'post-content', 'entry-content', 'content-main', 
                                  'main-content', 'region-content', 'document-content', 'privacy-policy']:
                    content_area = soup.find('div', class_=class_name)
                    if content_area:
                        break
            
            if not content_area:
                # Try section tags
                content_area = soup.find('section')
            
            if not content_area:
                # Fall back to generic content div but ensure it's large enough
                content_divs = soup.find_all('div', class_=lambda x: x and any(keyword in (x if isinstance(x, str) else ' '.join(x)) 
                                           for keyword in ['content', 'body', 'main', 'article', 'post']))
                if content_divs:
                    # Find the largest content div (most likely the actual content)
                    content_area = max(content_divs, key=lambda d: len(d.get_text()))
            
            if not content_area:
                # Find the largest section by text content
                all_sections = soup.find_all(['section', 'article', 'div'], class_=lambda x: True)
                if all_sections:
                    # Filter out tiny elements (likely navigation, ads, etc.) - only consider with significant content
                    significant_elements = [e for e in all_sections if len(e.get_text(strip=True)) > 300]
                    if significant_elements:
                        content_area = max(significant_elements, key=lambda e: len(e.get_text()))
            
            if not content_area:
                # Last resort: get all divs and find the largest one with substantial content
                all_divs = soup.find_all('div')
                if all_divs:
                    # Find divs with substantial text content (avoid headers, footers, navs)
                    large_divs = [d for d in all_divs if len(d.get_text(strip=True)) > 500 
                                 and not any(skip in str(d.get('class', [])).lower() 
                                           for skip in ['header', 'footer', 'nav', 'sidebar', 'ads', 'modal'])]
                    if large_divs:
                        content_area = max(large_divs, key=lambda d: len(d.get_text()))
        
        if not content_area:
            # Use body as fallback
            content_area = soup.find('body')
        
        if not content_area:
            # Ultimate fallback: use entire document
            content_area = soup
        
        # Remove navigation, header, footer, and sidebar elements that snuck through
        for element in content_area.find_all(['nav', 'header', 'footer', 'aside', 'form'], recursive=True):
            element.decompose()
        
        # Remove common ad/tracker containers
        for element in content_area.find_all(['div', 'section'], class_=lambda x: x and any(skip in (x if isinstance(x, str) else ' '.join(x)).lower() 
                                             for skip in ['ads', 'advertisement', 'tracker', 'modal', 'popup'])):
            element.decompose()
        
        if remove_extra_spans:
            # Remove redundant nested spans - keep only spans with meaningful attributes or content
            for span in content_area.find_all('span'):
                # If span has no class/id and only contains text or other spans, unwrap it
                if not span.get('class') and not span.get('id') and not span.get('style'):
                    span.unwrap()
        
        # Extract text with proper spacing
        text = content_area.get_text(separator='\n', strip=True)
        
        # Clean up excessive whitespace and empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned_text = '\n'.join(lines)
        
        print(f"[DEBUG] Extracted {len(cleaned_text)} characters from page")
        
        return cleaned_text
    
    except Exception as e:
        print(f"Error extracting clean content from HTML: {e}")
        return None


def scrape_and_extract(url, use_javascript=False, extract_text=True, wait_time=10):
    """
    Convenience function that scrapes a webpage and optionally extracts text.
    
    Parameters:
    -----------
    url : str
        The URL of the webpage to scrape
    use_javascript : bool, optional
        If True, renders JavaScript content (default: False)
    extract_text : bool, optional
        If True, extracts plain text from HTML (default: True)
    wait_time : int, optional
        Maximum time to wait for page elements to load (default: 10)
    
    Returns:
    --------
    str : Either HTML content or plain text depending on extract_text parameter
    None : If scraping fails
    
    Example:
    --------
    >>> content = scrape_and_extract("https://example.com", extract_text=True)
    >>> print(content)
    """
    html_content = scrape_webpage(url, use_javascript=use_javascript, wait_time=wait_time)
    
    if html_content is None:
        return None
    
    if extract_text:
        return extract_text_from_html(html_content)
    else:
        return html_content


def scrape_and_extract_clean(url, use_javascript=False, wait_time=10):
    """
    Scrapes a webpage and extracts clean, readable content (no HTML garbage).
    Perfect for privacy policies, terms of service, and other text-heavy pages.
    
    Parameters:
    -----------
    url : str
        The URL of the webpage to scrape
    use_javascript : bool, optional
        If True, renders JavaScript content (default: False)
    wait_time : int, optional
        Maximum time to wait for page elements to load (default: 10)
    
    Returns:
    --------
    str : Clean, formatted text content without HTML
    None : If scraping or extraction fails
    
    Example:
    --------
    >>> content = scrape_and_extract_clean("https://blinkit.com/privacy", use_javascript=True)
    >>> print(content)
    """
    html_content = scrape_webpage(url, use_javascript=use_javascript, wait_time=wait_time)
    
    if html_content is None:
        return None
    
    return extract_clean_content(html_content, article_tag=True, remove_extra_spans=True)


def extract_clean_content_from_file(html_file_path):
    """
    Extracts clean, readable content from an HTML file.
    Useful for processing previously saved HTML files or extracted content.
    
    Parameters:
    -----------
    html_file_path : str
        Path to the HTML file to extract content from
    
    Returns:
    --------
    str : Clean, formatted text content without HTML
    None : If extraction fails
    
    Example:
    --------
    >>> content = extract_clean_content_from_file("extracted_html.txt")
    >>> print(content)
    """
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return extract_clean_content(html_content, article_tag=True, remove_extra_spans=True)
    except Exception as e:
        print(f"Error reading HTML file: {e}")
        return None
    
def extract_privacy_policy_url_from_play_store(html_content):
    """
    Extracts the actual privacy policy URL from Google Play Store Data Safety page.
    
    Parameters:
    -----------
    html_content : str
        The HTML content of the Play Store Data Safety page
    
    Returns:
    --------
    str : The privacy policy URL if found
    None : If no privacy policy link is found
    
    Example:
    --------
    >>> html = scrape_webpage('https://play.google.com/store/apps/datasafety?id=com.grofers.customerapp')
    >>> policy_url = extract_privacy_policy_url_from_play_store(html)
    >>> print(policy_url)
    """
    if not html_content:
        return None
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all <a> tags with class 'GO2pB'
        privacy_links = soup.find_all('a', class_='GO2pB')
        
        # Search through links for privacy/policy keywords
        for link in privacy_links:
            href = link.get('href')
            link_text = link.get_text().lower()
            
            if href and ('privacy' in href.lower() or 'privacy' in link_text or 'policy' in link_text):
                # If href doesn't start with http, prepend Play Store domain
                if href.startswith('http'):
                    return href
                else:
                    return f'https://play.google.com{href}'
        
        return None
    
    except Exception as e:
        print(f"Error extracting privacy policy URL from Play Store: {e}")
        return None

def scrape_play_store_privacy_policy(package_name, use_javascript=True, wait_time=15):
    """
    Scrapes a privacy policy from Google Play Store Data Safety page.
    
    This function:
    1. Builds the Play Store Data Safety URL from package name
    2. Scrapes the Play Store page
    3. Extracts the actual privacy policy link
    4. Scrapes and cleans the actual privacy policy content
    
    Parameters:
    -----------
    package_name : str
        The Android app package name (e.g., 'com.grofers.customerapp')
    use_javascript : bool, optional
        If True, renders JavaScript content (default: True)
    wait_time : int, optional
        Maximum time to wait for page elements to load (default: 15)
    
    Returns:
    --------
    dict : Dictionary with keys:
        - 'success': bool indicating if scraping was successful
        - 'content': Clean privacy policy content if successful
        - 'policy_url': The actual privacy policy URL found
        - 'error': Error message if not successful
    
    Example:
    --------
    >>> result = scrape_play_store_privacy_policy('com.grofers.customerapp')
    >>> if result['success']:
    ...     print(result['content'])
    ... else:
    ...     print(f"Error: {result['error']}")
    """
    try:
        # Step 1: Build Play Store URL
        play_store_url = f"https://play.google.com/store/apps/datasafety?id={package_name}&hl=en_US"
        print(f"[PLAY_STORE] Scraping Play Store page: {play_store_url}")
        
        # Step 2: Scrape Play Store page
        play_store_html = scrape_webpage(play_store_url, use_javascript=use_javascript, wait_time=wait_time)
        
        if not play_store_html:
            return {
                'success': False,
                'error': f'Failed to scrape Play Store page for package: {package_name}',
                'policy_url': None,
                'content': None
            }
        
        print(f"[PLAY_STORE] Play Store page scraped, extracting privacy policy link...")
        
        # Step 3: Extract actual privacy policy URL
        policy_url = extract_privacy_policy_url_from_play_store(play_store_html)
        
        if not policy_url:
            return {
                'success': False,
                'error': f'No privacy policy link found on Play Store page for package: {package_name}',
                'policy_url': None,
                'content': None
            }
        
        print(f"[PLAY_STORE] Found privacy policy URL: {policy_url}")
        
        # Step 4: Scrape actual privacy policy
        # Many privacy policy pages are JavaScript-heavy, so force JS rendering
        print(f"[PLAY_STORE] Scraping actual privacy policy from: {policy_url}")
        print(f"[PLAY_STORE] Using JavaScript rendering (forced for privacy policies)")
        policy_content = scrape_and_extract_clean(policy_url, use_javascript=True, wait_time=20)
        
        if not policy_content or len(policy_content.strip()) < 200:
            # If we got very little content, try without JS rendering as fallback
            print(f"[PLAY_STORE] Content too short ({len(policy_content) if policy_content else 0} chars), trying without JS...")
            policy_content = scrape_and_extract_clean(policy_url, use_javascript=False, wait_time=10)
        
        if not policy_content:
            return {
                'success': False,
                'error': f'Failed to scrape or extract privacy policy from: {policy_url}',
                'policy_url': policy_url,
                'content': None
            }
        
        print(f"[PLAY_STORE] Successfully scraped privacy policy ({len(policy_content)} characters)")
        
        return {
            'success': True,
            'error': None,
            'policy_url': policy_url,
            'content': policy_content
        }
    
    except Exception as e:
        print(f"[PLAY_STORE] Error during Play Store scraping: {str(e)}")
        return {
            'success': False,
            'error': f'Exception during Play Store scraping: {str(e)}',
            'policy_url': None,
            'content': None
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def chunk_text(text, max_length=1000):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > max_length:
            chunks.append(' '.join(current_chunk[:-1]))
            current_chunk = [word]
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summarize_chunks_sequentially(chunks, summarizer):
    """Summarize chunks using the BART summarizer with batch processing"""
    if not chunks:
        return []
    
    # Filter out empty chunks
    valid_chunks = [chunk for chunk in chunks if chunk.strip()]
    
    if not valid_chunks:
        return []
    
    # Create a dataset for batch processing (eliminates the sequential warning)
    chunk_dataset = Dataset.from_dict({"text": valid_chunks})
    
    # Process in batches for efficiency
    batch_size = 8  # Adjust based on VRAM - 8 is safe for 4GB
    
    def summarize_batch(batch):
        """Summarize a batch of texts"""
        summaries = []
        for text in batch["text"]:
            try:
                result = summarizer(
                    text,
                    max_length=100,
                    min_length=30,
                    do_sample=False
                )
                summaries.append(result[0]['summary_text'])
            except Exception as e:
                # Fallback for texts that are too short or cause issues
                print(f"[WARNING] Summarization failed for chunk, using fallback: {str(e)[:50]}")
                summaries.append(text[:100])  # Use first 100 chars as fallback
        
        return {"summary": summaries}
    
    # Apply summarization to batches
    summarized_dataset = chunk_dataset.map(
        summarize_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"]
    )
    
    # Extract summaries
    summaries = summarized_dataset["summary"]
    
    return summaries

def get_embeddings(texts, model, tokenizer, device="cpu"):
    """Generate embeddings for texts using TinyLLama"""
    embeddings = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        
        embeddings.append(embedding[0])
    
    return np.array(embeddings)

# ============================================================================
# COMPRESSION METRICS & STREAMING
# ============================================================================

class CompressionMetrics:
    """Track compression ratio to predict if next iteration will be final"""
    def __init__(self, history_size=5):
        self.history = []  # List of (input_length, output_length) tuples
        self.history_size = history_size
    
    def add_iteration(self, input_length, output_length):
        """Record an iteration's compression"""
        self.history.append((input_length, output_length))
        if len(self.history) > self.history_size:
            self.history.pop(0)
    
    def get_avg_compression_ratio(self):
        """Calculate average compression ratio"""
        if not self.history:
            return 0.5  # Conservative default: 50% reduction
        
        ratios = [output / input_val for input_val, output in self.history if input_val > 0]
        return sum(ratios) / len(ratios) if ratios else 0.5
    
    def predict_final_iteration(self, current_length, threshold=1000):
        """
        Predict if next iteration will be final
        
        Args:
            current_length: Length of current combined summaries
            threshold: Target length threshold in words
        
        Returns:
            tuple: (is_likely_final: bool, predicted_length: float)
        """
        ratio = self.get_avg_compression_ratio()
        predicted_length = current_length * ratio
        
        # Add safety margin (20%) to account for variance
        safety_margin = 1.2
        conservative_prediction = predicted_length * safety_margin
        
        is_likely_final = conservative_prediction <= threshold
        
        return is_likely_final, predicted_length

async def generate_stream_event(event_type: str, data: dict, iteration: int = None) -> str:
    """Generate a server-sent event formatted as JSON"""
    event = {
        "type": event_type,
        "data": data,
        "timestamp": time.time()
    }
    if iteration is not None:
        event["iteration"] = iteration
    
    return f"data: {json.dumps(event)}\n\n"

async def summarize_with_streaming(text: str, max_length: int = 200, min_length: int = 75) -> AsyncGenerator[str, None]:
    """
    Summarize text with chunk-by-chunk streaming via Server-Sent Events
    
    Yields server-sent events for each chunk summary and iteration progress
    """
    metrics = CompressionMetrics()
    iteration = 0
    
    try:
        # Initial setup
        text = text.strip()
        words = text.split()
        initial_length = len(' '.join(words))
        
        yield await generate_stream_event(
            "start",
            {
                "message": "Starting summarization",
                "initial_length": initial_length,
                "initial_words": len(words)
            }
        )
        
        # Split into initial chunks
        chunks = chunk_text(text, max_length=1000)
        yield await generate_stream_event(
            "chunking_complete",
            {
                "num_chunks": len(chunks),
                "message": f"Split into {len(chunks)} chunks"
            },
            iteration=iteration
        )
        
        # Main summarization loop
        while True:
            iteration += 1
            chunk_summaries = []
            combined_length_before = len(' '.join(chunks))
            
            # Stream each chunk summary as it's generated
            for i, chunk in enumerate(chunks):
                try:
                    result = summarizer(
                        chunk,
                        max_length=100,
                        min_length=30,
                        do_sample=False
                    )
                    summary = result[0]['summary_text']
                    chunk_summaries.append(summary)
                    
                    # Stream individual chunk summary
                    yield await generate_stream_event(
                        "chunk_summary",
                        {
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "summary": summary,
                            "chunk_length": len(chunk.split()),
                            "summary_length": len(summary.split())
                        },
                        iteration=iteration
                    )
                    
                    # Small delay to allow client to process
                    await asyncio.sleep(0.01)
                
                except Exception as e:
                    chunk_summaries.append(chunk[:100])
                    yield await generate_stream_event(
                        "chunk_summary_error",
                        {
                            "chunk_index": i,
                            "error": str(e)[:100],
                            "fallback": "Used first 100 chars"
                        },
                        iteration=iteration
                    )
            
            # Calculate compression metrics
            combined_length_after = len(' '.join(chunk_summaries))
            metrics.add_iteration(combined_length_before, combined_length_after)
            
            combined_length_words = len(' '.join(chunk_summaries).split())
            
            yield await generate_stream_event(
                "iteration_complete",
                {
                    "combined_length": combined_length_after,
                    "combined_words": combined_length_words,
                    "compression_ratio": combined_length_after / combined_length_before if combined_length_before > 0 else 0,
                    "avg_compression_ratio": metrics.get_avg_compression_ratio()
                },
                iteration=iteration
            )
            
            # Check if we're done
            if combined_length_words <= 1000:
                # Final iteration
                summary = '\n'.join(chunk_summaries)
                yield await generate_stream_event(
                    "complete",
                    {
                        "final_summary": summary,
                        "final_length": len(summary.split()),
                        "total_iterations": iteration,
                        "compression_rate": combined_length_after / initial_length if initial_length > 0 else 0
                    }
                )
                break
            
            # Predict if next iteration will be final
            is_likely_final, predicted_length = metrics.predict_final_iteration(
                combined_length_words,
                threshold=1000
            )
            
            yield await generate_stream_event(
                "prediction",
                {
                    "current_length": combined_length_words,
                    "predicted_next_length": predicted_length,
                    "likely_final_iteration": is_likely_final,
                    "confidence": "high" if abs(predicted_length - combined_length_words) < 200 else "medium"
                },
                iteration=iteration
            )
            
            # Prepare for next iteration
            full_text = ' '.join(chunk_summaries)
            chunks = chunk_text(full_text, max_length=1000)
            
            yield await generate_stream_event(
                "rechunking",
                {
                    "new_num_chunks": len(chunks),
                    "message": f"Rechunked into {len(chunks)} chunks for next iteration"
                },
                iteration=iteration
            )
            
            await asyncio.sleep(0.01)  # Brief pause between iterations
    
    except Exception as e:
        yield await generate_stream_event(
            "error",
            {
                "error_message": str(e),
                "iteration": iteration
            }
        )

# ============================================================================
# CHATBOT CLASS
# ============================================================================

class TinyLLamaChatbot:
    def __init__(self, knowledge_base, embed_model, embed_tokenizer, llm_pipeline, device="cpu"):
        """Initialize chatbot with TinyLLama"""
        self.knowledge_base = knowledge_base
        self.embed_model = embed_model
        self.embed_tokenizer = embed_tokenizer
        self.llm = llm_pipeline
        self.device = device
        self.query_cache = {}
        
    def embed_query(self, query):
        """Embed query with TinyLLama (CPU)"""
        if query in self.query_cache:
            return self.query_cache[query]
        
        inputs = self.embed_tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
            query_embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
        
        self.query_cache[query] = query_embedding
        return query_embedding
    
    def retrieve_context(self, query, top_k=2):
        """Retrieve top-k relevant chunks"""
        query_embedding = self.embed_query(query)
        similarities = cosine_similarity([query_embedding], self.knowledge_base['embeddings'])[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        context_chunks = [self.knowledge_base['chunks'][i] for i in top_indices]
        scores = [similarities[i] for i in top_indices]
        
        return context_chunks, scores
    
    def answer_question(self, question, top_k=2):
        """Generate answer using TinyLLama"""
        context_chunks, scores = self.retrieve_context(question, top_k=top_k)
        context_str = "\n".join([f"[Ref {i+1}] {chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Create prompt for TinyLLama
        prompt = f"""<|im_start|>system
You are a helpful assistant for privacy policies. Answer questions based on the provided context only.
<|im_end|>
<|im_start|>user
Context from privacy policy:
{context_str}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""
        
        response = self.llm(
            prompt,
            max_new_tokens=120,
            min_new_tokens=30,
            do_sample=False,
            temperature=0.0
        )
        
        # Extract answer
        answer = response[0]['generated_text'].split("<|im_start|>assistant")[-1].strip()
        
        return {
            'answer': answer,
            'context_chunks': context_chunks,
            'relevance_scores': scores
        }

# ============================================================================
# INITIALIZATION - Load Models and Create Knowledge Bases
# ============================================================================

print("\n" + "="*70)
print("PRIVACY POLICY AI SERVER - INITIALIZATION")
print("="*70)

# Load summarizer for chunked summarization
print("\n1. Loading BART summarizer...")
summarizer = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    device=0,
)

# Load TinyLLama for embeddings on CPU
print("2. Loading TinyLLama 1.1B for embeddings (CPU)...")
embed_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
embed_model = AutoModel.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype=torch.float32
)

# Extract and chunk privacy policy
print("3. Processing privacy policy...")
full_text = extract_clean_content_from_file("privacy_policy_clean.txt")
policy_chunks = chunk_text(full_text, max_length=400)
print(f"   Created {len(policy_chunks)} chunks")

# Generate embeddings
print("4. Generating embeddings for all chunks (CPU)...")
chunk_embeddings = get_embeddings(policy_chunks, embed_model, embed_tokenizer, device="cpu")

knowledge_base = {
    'chunks': policy_chunks,
    'embeddings': chunk_embeddings,
    'num_chunks': len(policy_chunks)
}

# Load TinyLLama for text generation with 4-bit quantization
print("5. Loading 4-bit quantized TinyLLama for text generation (GPU)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

llm_pipeline = pipeline(
    task="text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_kwargs={"quantization_config": bnb_config}
)

llm_pipeline.tokenizer.pad_token = llm_pipeline.tokenizer.eos_token

# Initialize chatbot
print("6. Initializing chatbot...")
chatbot = TinyLLamaChatbot(
    knowledge_base=knowledge_base,
    embed_model=embed_model,
    embed_tokenizer=embed_tokenizer,
    llm_pipeline=llm_pipeline,
    device="cpu"
)

print("\n✓ All models loaded successfully!")

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Privacy Policy AI API", version="1.0.0")

# Request/Response models
class SummarizeRequest(BaseModel):
    """Request model for summarization"""
    text: str
    max_length: int = 200
    min_length: int = 75

class SummarizeResponse(BaseModel):
    """Response model for summarization"""
    summary: str
    length: int

class ChatRequest(BaseModel):
    """Request model for chatbot Q&A"""
    question: str
    top_k: int = 3

class ChatResponse(BaseModel):
    """Response model for chatbot"""
    answer: str
    context_chunks: List[str]
    relevance_scores: List[float]

class ScrapeRequest(BaseModel):
    """Request model for web scraping and summarization"""
    url: str = None  # Optional - use package_name instead for Play Store
    package_name: str = None  # Optional - for Play Store scraping
    use_javascript: bool = True
    wait_time: int = 15

class ScrapeResponse(BaseModel):
    """Response model for scraping"""
    policy_content: str = None
    summary: str = None
    content_length: int = None
    summary_length: int = None
    source_url: str = None  # URL that was actually scraped

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    available_chatbots: List[str]

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "available_chatbots": ["TinyLLama 1.1B (CPU embeddings + 4-bit quantized generation)"]
    }

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_and_summarize(request: ScrapeRequest):
    """
    Scrape a website (including JavaScript-rich pages), extract clean content, and summarize it.
    
    Supports two modes:
    1. Direct URL scraping: Pass a URL directly
    2. Play Store scraping: Pass a package_name, and the API will:
       - Fetch the Play Store Data Safety page
       - Extract the actual privacy policy link
       - Scrape and summarize the actual policy
    
    Args:
        url: The URL to scrape (can be Play Store URL or direct privacy policy URL)
        package_name: Android package name (if scraping from Play Store)
        use_javascript: Whether to render JavaScript content (default: True)
        wait_time: Maximum time to wait for page load in seconds (default: 15)
    
    Returns:
        Cleaned policy content, corresponding summary, and source URL
    """
    try:
        source_url = None
        policy_content = None
        
        # Check if this is a Play Store request
        if request.package_name:
            print(f"\n[SCRAPE] Play Store mode: package={request.package_name}")
            result = scrape_play_store_privacy_policy(
                request.package_name,
                use_javascript=request.use_javascript,
                wait_time=request.wait_time
            )
            
            if not result['success']:
                raise HTTPException(status_code=400, detail=result['error'])
            
            policy_content = result['content']
            source_url = result['policy_url']
            
            print(f"[SCRAPE] Play Store scraping successful")
        
        else:
            # Direct URL scraping
            if not request.url.strip():
                raise HTTPException(status_code=400, detail="URL or package_name must be provided")
            
            print(f"\n[SCRAPE] Direct URL mode: {request.url}")
            print(f"[SCRAPE] JavaScript rendering: {request.use_javascript}")
            
            # Scrape and extract clean content
            print(f"[SCRAPE] Scraping webpage...")
            policy_content = scrape_and_extract_clean(
                request.url,
                use_javascript=request.use_javascript,
                wait_time=request.wait_time
            )
            
            source_url = request.url
        
        if not policy_content:
            raise HTTPException(status_code=400, detail="Failed to scrape or extract content")
        
        if len(policy_content.strip()) < 50:
            raise HTTPException(status_code=400, detail="Scraped content too short (minimum 50 characters)")
        
        print(f"[SCRAPE] Scraped content: {len(policy_content)} characters")
        
        # Summarize the scraped content
        print(f"[SCRAPE] Starting summarization...")
        text = policy_content.strip()
        words = text.split()
        
        print(f"[SCRAPE] Content length: {len(words)} words")
        
        # Split into initial chunks
        chunks = chunk_text(text, max_length=1000)
        print(f"[SCRAPE] Initial chunks: {len(chunks)}")
        
        # Summarize each chunk
        print(f"[SCRAPE] Summarizing chunks...")
        chunk_summaries = summarize_chunks_sequentially(chunks, summarizer)
        
        print(f"[SCRAPE] Generated {len(chunk_summaries)} summaries")
        
        # Iteratively summarize until combined summary is under 1000 words
        iteration = 1
        while len(' '.join(chunk_summaries).split()) > 1000:
            print(f"[SCRAPE] Iteration {iteration}: Combined summaries {len(' '.join(chunk_summaries).split())} words, reducing...")
            
            full_text = ' '.join(chunk_summaries)
            chunks = chunk_text(full_text, max_length=1000)
            print(f"[SCRAPE] Reprocessing {len(chunks)} chunks")
            
            chunk_summaries = summarize_chunks_sequentially(chunks, summarizer)
            iteration += 1
        
        # Join final summaries
        summary = '\n'.join(chunk_summaries)
        
        print(f"[SCRAPE] Final summary: {len(summary.split())} words")
        print(f"[SCRAPE] Scrape and summarization complete!")
        
        return {
            "policy_content": policy_content,
            "summary": summary,
            "content_length": len(policy_content.split()),
            "summary_length": len(summary.split()),
            "source_url": source_url
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SCRAPE] Scrape error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scrape error: {str(e)}")

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_policy(request: SummarizeRequest):
    """
    Summarize a privacy policy or text document using iterative chunking
    
    Args:
        text: The policy text to summarize
        max_length: Maximum length of summary (tokens) - for final pass
        min_length: Minimum length of summary (tokens) - for final pass
    
    Returns:
        Summary text and its length
    """
    try:
        if len(request.text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Text too short to summarize (minimum 50 characters)")
        
        text = request.text.strip()
        words = text.split()
        
        print(f"Summarization request: {len(words)} words")
        
        # Split into initial chunks
        chunks = chunk_text(text, max_length=1000)
        print(f"Initial chunks: {len(chunks)}")
        
        # Summarize each chunk
        print("Summarizing chunks...")
        chunk_summaries = summarize_chunks_sequentially(chunks, summarizer)
        
        print(f"Generated {len(chunk_summaries)} summaries")
        
        # Iteratively summarize until combined summary is under 1000 words
        iteration = 1
        while len(' '.join(chunk_summaries).split()) > 1000:
            print(f"\nIteration {iteration}: Combined summaries {len(' '.join(chunk_summaries).split())} words, reducing...")
            
            full_text = ' '.join(chunk_summaries)
            chunks = chunk_text(full_text, max_length=1000)
            print(f"Reprocessing {len(chunks)} chunks")
            
            chunk_summaries = summarize_chunks_sequentially(chunks, summarizer)
            iteration += 1
        
        # Join final summaries without additional summarization
        summary = ' '.join(chunk_summaries)
        
        print(f"Final summary: {len(summary.split())} words")
        
        return {
            "summary": summary,
            "length": len(summary.split())
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_question(request: ChatRequest):
    """
    Ask a question about the privacy policy using TinyLLama chatbot
    
    Args:
        question: The question to ask
        top_k: Number of context chunks to retrieve (1-5)
    
    Returns:
        Answer with context chunks and relevance scores
    """
    try:
        # Validate inputs
        if len(request.question.strip()) < 5:
            raise HTTPException(status_code=400, detail="Question too short (minimum 5 characters)")
        
        if request.top_k < 1 or request.top_k > 5:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 5")
        
        # Get answer from chatbot
        response = chatbot.answer_question(request.question, top_k=request.top_k)
        
        return {
            "answer": response['answer'],
            "context_chunks": response['context_chunks'],
            "relevance_scores": response['relevance_scores']
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/summarize-stream")
async def summarize_policy_stream(request: SummarizeRequest):
    """
    Summarize a privacy policy with real-time chunk streaming via Server-Sent Events
    
    Returns a stream of events:
    - start: Initialization with content metrics
    - chunking_complete: Initial chunks created
    - chunk_summary: Individual chunk summary generated
    - iteration_complete: All chunks in iteration summarized
    - prediction: Prediction for next iteration
    - rechunking: Chunks prepared for next iteration
    - complete: Final summary ready
    - error: If something goes wrong
    
    Client should listen to stream and parse JSON events
    """
    try:
        if len(request.text.strip()) < 50:
            async def error_stream():
                yield await generate_stream_event("error", {"error_message": "Text too short (minimum 50 characters)"})
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        
        return StreamingResponse(
            summarize_with_streaming(request.text, request.max_length, request.min_length),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        async def error_stream():
            yield await generate_stream_event("error", {"error_message": str(e)})
        
        return StreamingResponse(error_stream(), media_type="text/event-stream")

@app.post("/scrape-stream")
async def scrape_and_summarize_stream(request: ScrapeRequest):
    """
    Scrape a website and summarize with real-time streaming
    Checks Firebase cache first to avoid redundant scraping
    
    Returns a stream of events:
    - cache_hit: Found in Firebase (skips scraping)
    - cache_miss: Not in cache (proceeds with scraping)
    - scrape_start: Scraping begins
    - scrape_complete: HTML extracted
    - extraction_complete: Clean content extracted
    - policy_content: Full cleaned policy
    - (then same as /summarize-stream events)
    - cache_save_complete: Successfully saved to Firebase
    """
    async def scrape_summarize_stream():
        try:
            source_url = None
            policy_content = None
            from_cache = False
            
            # Check Firebase cache first if package_name is provided
            if request.package_name:
                print(f"[CACHE] Checking Firebase for: {request.package_name}")
                cached_data = check_cache(request.package_name)
                
                if cached_data:
                    print(f"[CACHE] Cache HIT for {request.package_name}")
                    yield await generate_stream_event("cache_hit", {
                        "package_name": request.package_name,
                        "message": "Found in Firebase cache"
                    })
                    
                    policy_content = cached_data.get('policy')
                    source_url = cached_data.get('source_url')
                    from_cache = True
                    
                    # Stream the cached policy content
                    yield await generate_stream_event("policy_content", {
                        "content": policy_content,
                        "word_count": len(policy_content.split()),
                        "char_count": len(policy_content),
                        "message": "Cached privacy policy content",
                        "from_cache": True
                    })
                    
                    # Stream the cached summary as chunk_summary event
                    yield await generate_stream_event("chunk_summary", {
                        "chunk_number": 1,
                        "summary": cached_data.get('summary'),
                        "words": len(cached_data.get('summary', '').split()),
                        "from_cache": True
                    })
                    
                    yield await generate_stream_event("complete", {
                        "message": "Policy summary retrieved from cache",
                        "from_cache": True
                    })
                    return
                
                else:
                    print(f"[CACHE] Cache MISS for {request.package_name}")
                    yield await generate_stream_event("cache_miss", {
                        "package_name": request.package_name,
                        "message": "Not in Firebase cache, proceeding with scraping"
                    })
            
            # If we get here, not in cache - proceed with scraping
            if request.package_name:
                yield await generate_stream_event("scrape_start", {
                    "mode": "play_store",
                    "package_name": request.package_name
                })
                
                result = scrape_play_store_privacy_policy(
                    request.package_name,
                    use_javascript=request.use_javascript,
                    wait_time=request.wait_time
                )
                
                if not result['success']:
                    yield await generate_stream_event("error", {"error_message": result['error']})
                    return
                
                policy_content = result['content']
                source_url = result['policy_url']
            
            else:
                # Direct URL
                if not request.url or not request.url.strip():
                    yield await generate_stream_event("error", {"error_message": "URL required"})
                    return
                
                yield await generate_stream_event("scrape_start", {
                    "mode": "direct_url",
                    "url": request.url,
                    "use_javascript": request.use_javascript
                })
                
                policy_content = scrape_and_extract_clean(
                    request.url,
                    use_javascript=request.use_javascript,
                    wait_time=request.wait_time
                )
                source_url = request.url
            
            if not policy_content or len(policy_content.strip()) < 50:
                yield await generate_stream_event("error", {"error_message": "Failed to extract content"})
                return
            
            yield await generate_stream_event("scrape_complete", {
                "content_length": len(policy_content),
                "content_words": len(policy_content.split()),
                "source_url": source_url
            })
            
            # Stream the full cleaned policy content to the client
            yield await generate_stream_event("policy_content", {
                "content": policy_content,
                "word_count": len(policy_content.split()),
                "char_count": len(policy_content),
                "message": "Full cleaned privacy policy content"
            })
            
            # Stream the summarization and capture final summary
            final_summary = None
            async for event in summarize_with_streaming(policy_content):
                # Parse the event to extract final summary
                if event and "data: " in event:
                    try:
                        # Extract JSON from "data: {...}\n\n"
                        json_str = event.replace("data: ", "").strip()
                        event_data = json.loads(json_str)
                        # Check for completion event
                        if event_data.get("type") == "complete" and "data" in event_data:
                            if "final_summary" in event_data["data"]:
                                final_summary = event_data["data"]["final_summary"]
                                print(f"[FIREBASE] Captured final summary: {len(final_summary.split())} words")
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        # Silently skip malformed events
                        pass
                
                yield event
            
            # Save to Firebase if we just scraped and summarized (not from cache)
            if not from_cache and request.package_name and final_summary and policy_content:
                print(f"[FIREBASE] Saving to cache: {request.package_name}")
                if save_to_cache(
                    request.package_name,
                    policy_content,
                    final_summary,
                    source_url or ""
                ):
                    print(f"[FIREBASE] Successfully saved {request.package_name}")
                    yield await generate_stream_event("cache_save_complete", {
                        "package_name": request.package_name,
                        "message": "Policy saved to Firebase cache for future requests"
                    })
                else:
                    print(f"[FIREBASE] Failed to save {request.package_name}")
        
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            yield await generate_stream_event("error", {"error_message": str(e)})
    
    return StreamingResponse(scrape_summarize_stream(), media_type="text/event-stream")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream answers for real-time chat experience (for future enhancement)
    """
    return {"message": "Streaming endpoint coming soon"}

# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log when server starts"""
    print("\n✓ FastAPI server started on http://localhost:8000")
    print("✓ Both chatbots are ready!")
    print("\nAPI Documentation:")
    print("  - Interactive Swagger UI: http://localhost:8000/docs")
    print("  - ReDoc: http://localhost:8000/redoc")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    print("\n✓ Shutting down server...")
    torch.cuda.empty_cache()
    gc.collect()
    print("✓ GPU memory cleared")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import os
    
    # Support both local development and HuggingFace Spaces
    # HuggingFace Spaces expects port 7860
    port = int(os.getenv("PORT", 7860))
    host = "0.0.0.0"
    
    print("\n" + "="*70)
    print("STARTING SERVER")
    print("="*70)
    print(f"\nServer will run on http://localhost:{port}")
    print("Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host=host, port=port)
