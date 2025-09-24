#!/usr/bin/env python3
"""
POS Tagging Script for Dutch Newspaper Data
Adapted from existing context extraction code
"""

import pandas as pd
import spacy
import os
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from threading import Lock

# Thread-safe spaCy model loading
nlp_lock = Lock()
nlp = None

def get_nlp_model():
    """Thread-safe spaCy model getter"""
    global nlp
    if nlp is None:
        with nlp_lock:
            if nlp is None:
                nlp = spacy.load("nl_core_news_sm")
    return nlp

# Define keyword dictionary
KEYWORDS = {
    'blanke': ['blanke', 'blanken'],
    'bosneger': ['bosneger', 'bosnegers'],
    'creool': ['creool', 'creolen'],
    'gekleurd': ['gekleurd', 'gekleurden'],
    'halfbloed': ['halfbloed', 'halfbloeden'],
    'hottentot': ['hottentot', 'hottentotten'],
    'inboorling': ['inboorling', 'inboorlingen'],
    'indisch': ['indisch', 'indische'],
    'indo': ['indo', "indo's"],
    'indiaan': ['indiaan', 'indianen'],
    'inheems': ['inheems', 'inheemsen'],
    'inlander': ['inlander', 'inlanders'],
    'kaffer': ['kaffer', 'kaffers'],
    'khoi': ['khoi'],
    'kleurling': ['kleurling', 'kleurlingen'],
    'moor': ['moor', 'moren'],
    'marron': ['marron', 'marrons'],
    'mesties': ['mesties'],
    'mulat': ['mulat', 'mulatten'],
    'neger': ['neger', 'negers', 'negerin', 'negerinnen'],
    'njai': ['njai'],
    'primitief': ['primitief', 'primitieven'],
    'wildeman': ['wildeman', 'wildemannen'],
    'barbaar': ['barbaar', 'barbaren'],
    'koeli': ['koeli', 'koelie', 'koelies']
}

# Create a flat set of all keyword variants for quick lookup
ALL_KEYWORDS = set()
for variants in KEYWORDS.values():
    ALL_KEYWORDS.update(variant.lower() for variant in variants)

def process_text(text):
    """
    Process text with POS tagging and keyword highlighting
    
    Args:
        text (str): Input text to process
        
    Returns:
        tuple: (pos_tags, content_tagged)
    """
    if not text or pd.isna(text):
        return "", ""
    
    # Convert to string and clean
    text = str(text).strip()
    if not text:
        return "", ""
    
    # Process with spaCy (thread-safe)
    nlp_model = get_nlp_model()
    doc = nlp_model(text)
    
    pos_tags = []
    content_tagged = []
    
    for token in doc:
        # Get POS tag
        pos_tag = token.pos_
        
        # Create POS tag format: [word]_POS
        pos_tags.append(f"[{token.text}]_{pos_tag}")
        
        # Check if token is a keyword and is used as a noun
        token_lower = token.text.lower()
        if token_lower in ALL_KEYWORDS and pos_tag == "NOUN":
            # Highlight keyword when used as noun
            content_tagged.append(f"{token.text}_{pos_tag}")
        else:
            # Keep original text
            content_tagged.append(token.text)
    
    return " ".join(pos_tags), " ".join(content_tagged)

def process_chunk(chunk_data):
    """
    Process a chunk of rows for multithreading
    
    Args:
        chunk_data (tuple): (chunk_df, chunk_start_idx)
        
    Returns:
        tuple: (chunk_start_idx, processed_results)
    """
    chunk_df, chunk_start_idx = chunk_data
    results = []
    
    for idx, row in chunk_df.iterrows():
        try:
            pos_tags, content_tagged = process_text(row['content'])
            results.append((idx, pos_tags, content_tagged))
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            results.append((idx, "", ""))
    
    return chunk_start_idx, results

def process_tsv_file(input_file, output_file, test_mode=False, max_rows=None, num_threads=None):
    """
    Process a single TSV file with POS tagging
    
    Args:
        input_file (str): Path to input TSV file
        output_file (str): Path to output TSV file
        test_mode (bool): If True, process only a small sample
        max_rows (int): Maximum number of rows to process in test mode
        num_threads (int): Number of threads to use for processing
    """
    print(f"Processing {input_file}")
    
    # Read TSV file
    try:
        df = pd.read_csv(input_file, sep='\t', encoding='utf-8')
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return
    
    # Check if required columns exist
    required_columns = ['newspaper', 'year', 'content']
    if not all(col in df.columns for col in required_columns):
        print(f"Missing required columns in {input_file}. Expected: {required_columns}")
        return
    
    # Remove content_clean column if it exists (cleanup from previous runs)
    if 'content_clean' in df.columns:
        df = df.drop('content_clean', axis=1)
        print(f"Removed existing 'content_clean' column from {input_file}")
    
    # Remove existing POS columns if they exist (cleanup from previous runs)
    if 'pos_tags' in df.columns:
        df = df.drop('pos_tags', axis=1)
        print(f"Removed existing 'pos_tags' column from {input_file}")
    
    if 'content_tagged' in df.columns:
        df = df.drop('content_tagged', axis=1)
        print(f"Removed existing 'content_tagged' column from {input_file}")
    
    # Apply test mode restrictions
    if test_mode:
        original_len = len(df)
        if max_rows:
            df = df.head(max_rows)
        else:
            df = df.head(100)  # Default test size
        print(f"TEST MODE: Processing {len(df)} rows out of {original_len} total rows")
    else:
        print(f"Processing {len(df)} rows")
    
    # Initialize new columns
    df['pos_tags'] = ""
    df['content_tagged'] = ""
    
    # Determine number of threads
    if num_threads is None:
        num_threads = min(multiprocessing.cpu_count(), 4)  # Default to 4 threads max
    
    # Calculate chunk size for multithreading
    chunk_size = max(1, len(df) // num_threads)
    
    if len(df) < num_threads * 2:  # If dataset is too small for multithreading
        print("Dataset too small for multithreading, processing sequentially")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            try:
                pos_tags, content_tagged = process_text(row['content'])
                df.at[idx, 'pos_tags'] = pos_tags
                df.at[idx, 'content_tagged'] = content_tagged
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                df.at[idx, 'pos_tags'] = ""
                df.at[idx, 'content_tagged'] = ""
    else:
        # Multithreaded processing
        print(f"Using {num_threads} threads with chunk size {chunk_size}")
        
        # Split dataframe into chunks
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            chunks.append((chunk, i))
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all chunks
            future_to_chunk = {executor.submit(process_chunk, chunk_data): chunk_data[1] 
                              for chunk_data in chunks}
            
            # Collect results with progress bar
            with tqdm(total=len(df), desc="Processing rows (multithreaded)") as pbar:
                for future in as_completed(future_to_chunk):
                    chunk_start_idx, results = future.result()
                    
                    # Update dataframe with results
                    for idx, pos_tags, content_tagged in results:
                        df.at[idx, 'pos_tags'] = pos_tags
                        df.at[idx, 'content_tagged'] = content_tagged
                    
                    pbar.update(len(results))
    
    # Save processed file
    try:
        df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
        print(f"Saved processed file to {output_file}")
    except Exception as e:
        print(f"Error saving {output_file}: {e}")

def test_keyword_tagging():
    """
    Test the keyword tagging functionality with sample sentences
    """
    print("🧪 TESTING KEYWORD TAGGING FUNCTIONALITY")
    print("=" * 50)
    
    # Test sentences with keywords
    test_sentences = [
        "De blanke man liep door de straat.",
        "Hij was een primitieve neger uit Afrika.",
        "De creool sprak Nederlands.",
        "Een gekleurd persoon kwam binnen.",
        "De inlander werkte hard.",
        "This sentence has no keywords.",
        "De koeli werkte in de haven, een blanke man keek toe."
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nTest {i}: {sentence}")
        pos_tags, content_tagged = process_text(sentence)
        
        # Check if any keywords were found
        keywords_found = []
        for word in content_tagged.split():
            if '_' in word and not word.startswith('['):
                parts = word.rsplit('_', 1)
                if len(parts) == 2 and parts[0].lower() in ALL_KEYWORDS:
                    keywords_found.append(word)
        
        print(f"POS tags: {pos_tags}")
        print(f"Content tagged: {content_tagged}")
        if keywords_found:
            print(f"Keywords found: {', '.join(keywords_found)}")
        else:
            print("No keywords found")
    
    print("\n" + "=" * 50)

def main():
    """Main function to process all TSV files"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='POS Tagging for Dutch Newspaper Data')
    parser.add_argument('--test', action='store_true', help='Run in test mode (process small sample)')
    parser.add_argument('--test-rows', type=int, default=100, help='Number of rows to process in test mode')
    parser.add_argument('--threads', type=int, default=None, help='Number of threads to use')
    parser.add_argument('--input-dir', type=str, default='/data/groups/trifecta/jiaqiz/newspaper_tagging', 
                       help='Input directory containing TSV files')
    parser.add_argument('--output-dir', type=str, default='/data/groups/trifecta/jiaqiz/newspaper_tagging/tagged',
                       help='Output directory for tagged files')
    parser.add_argument('--files', nargs='+', help='Specific files to process (optional)')
    parser.add_argument('--test-keywords', action='store_true', help='Test keyword tagging functionality')
    parser.add_argument('--single-file', action='store_true', help='Process files one by one sequentially')
    
    args = parser.parse_args()
    
    # If testing keywords, run test and exit
    if args.test_keywords:
        test_keyword_tagging()
        return
    
    # Define paths
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define TSV files to process
    if args.files:
        tsv_files = args.files
    else:
        tsv_files = [
            "handelsblad_1860_1899.tsv", 
            "handelsblad_1900_1939.tsv",
            "handelsblad_1940_1959.tsv",
            "telegraaf_1860_1899.tsv",
            "telegraaf_1900_1939.tsv",
            "telegraaf_1940_1959.tsv"
        ]
    
    print("🔍 DUTCH NEWSPAPER POS TAGGING")
    print("=" * 50)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Processing {len(tsv_files)} TSV files")
    print(f"Keywords to highlight: {len(ALL_KEYWORDS)}")
    
    if args.test:
        print(f"🧪 TEST MODE: Processing only {args.test_rows} rows per file")
    
    if args.threads:
        print(f"🧵 Using {args.threads} threads")
    else:
        print(f"🧵 Using default number of threads (max {min(multiprocessing.cpu_count(), 4)})")
    
    if args.single_file:
        print("📄 SINGLE FILE MODE: Processing files one by one")
    
    print()
    
    # Process files
    if args.single_file:
        # Single file mode - process one at a time
        for i, tsv_file in enumerate(tsv_files, 1):
            input_path = os.path.join(input_dir, tsv_file)
            
            # Check if input file exists
            if not os.path.exists(input_path):
                print(f"Input file not found: {input_path}")
                continue
            
            # Add test prefix if in test mode
            if args.test:
                output_filename = f"test_tagged_{tsv_file}"
            else:
                output_filename = f"tagged_{tsv_file}"
            
            output_path = os.path.join(output_dir, output_filename)
            
            # Check if output file already exists
            if os.path.exists(output_path):
                print(f"⚠️  Output file already exists: {output_path}")
                response = input(f"Overwrite? (y/n): ").lower().strip()
                if response != 'y':
                    print(f"Skipping {tsv_file}")
                    continue
            
            print(f"\n📄 Processing file {i}/{len(tsv_files)}: {tsv_file}")
            print(f"Input: {input_path}")
            print(f"Output: {output_path}")
            
            # Process the file
            process_tsv_file(input_path, output_path, 
                            test_mode=args.test, 
                            max_rows=args.test_rows if args.test else None,
                            num_threads=args.threads)
            
            print(f"✅ File {tsv_file} processed successfully!")
            
            # Ask if user wants to continue (except for last file)
            if i < len(tsv_files):
                print(f"\n{len(tsv_files) - i} files remaining: {tsv_files[i:]}")
                response = input("Continue to next file? (y/n/q to quit): ").lower().strip()
                if response == 'q':
                    print("Processing stopped by user")
                    break
                elif response != 'y':
                    print("Processing paused. Run again to continue.")
                    break
            
            print("-" * 50)
    
    else:
        # Batch mode - process all files
        for tsv_file in tsv_files:
            input_path = os.path.join(input_dir, tsv_file)
            
            # Add test prefix if in test mode
            if args.test:
                output_filename = f"test_tagged_{tsv_file}"
            else:
                output_filename = f"tagged_{tsv_file}"
            
            output_path = os.path.join(output_dir, output_filename)
            
            # Check if input file exists
            if not os.path.exists(input_path):
                print(f"Input file not found: {input_path}")
                continue
            
            # Process the file
            process_tsv_file(input_path, output_path, 
                            test_mode=args.test, 
                            max_rows=args.test_rows if args.test else None,
                            num_threads=args.threads)
            
            print()
    
    print("✅ Processing complete!")
    print(f"Tagged files saved in: {output_dir}")
    
    if args.test:
        print("\n🧪 TEST MODE COMPLETE")
        print("If the test results look good, run without --test flag to process all data.")

# Additional utility functions for analysis
def analyze_keyword_usage(tagged_file):
    """
    Analyze keyword usage in a tagged file
    
    Args:
        tagged_file (str): Path to tagged TSV file
    """
    df = pd.read_csv(tagged_file, sep='\t', encoding='utf-8')
    
    keyword_counts = {}
    
    for idx, row in df.iterrows():
        content_tagged = str(row['content_tagged'])
        
        # Count keyword occurrences (those ending with _NOUN)
        for word in content_tagged.split():
            if word.endswith('_NOUN'):
                keyword = word.replace('_NOUN', '').lower()
                if keyword in ALL_KEYWORDS:
                    keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    return keyword_counts

def get_keyword_contexts(tagged_file, keyword, window_size=5):
    """
    Get contexts where a keyword appears as a noun
    
    Args:
        tagged_file (str): Path to tagged TSV file
        keyword (str): Keyword to search for
        window_size (int): Number of words before/after for context
    
    Returns:
        list: List of contexts where keyword appears as noun
    """
    df = pd.read_csv(tagged_file, sep='\t', encoding='utf-8')
    
    contexts = []
    
    for idx, row in df.iterrows():
        content_tagged = str(row['content_tagged'])
        words = content_tagged.split()
        
        for i, word in enumerate(words):
            if word.lower() == f"{keyword.lower()}_noun":
                # Extract context window
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                context = words[start:end]
                
                contexts.append({
                    'row': idx,
                    'year': row['year'],
                    'newspaper': row['newspaper'],
                    'context': ' '.join(context),
                    'keyword_position': i - start
                })
    
    return contexts

if __name__ == "__main__":
    main()