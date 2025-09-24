#!/usr/bin/env python3
"""
Multithreaded Delpher Filtered Newspaper Extractor (1880-1959)
Optimized for maximum CPU core utilization
"""

import requests
import xml.etree.ElementTree as ET
import time
import os
import re
import pandas as pd
from urllib.parse import quote
import unicodedata
from tqdm import tqdm
import logging
from datetime import datetime, timedelta
import shutil
import concurrent.futures
from threading import Lock, local
import multiprocessing
from functools import partial

# Setup logging with thread safety
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('delpher_extraction.log'),
        logging.StreamHandler()
    ]
)

# Thread-local storage for requests sessions
thread_local = local()

def get_session():
    """Get thread-local requests session for connection pooling"""
    if not hasattr(thread_local, 'session'):
        thread_local.session = requests.Session()
        thread_local.session.headers.update({
            'User-Agent': 'Delpher-Extractor/1.0'
        })
    return thread_local.session

# Global settings - OPTIMIZED FOR MULTITHREADING
start_year = 1880  
current_year = 1959  
records_per_batch = 100  # Smaller batches for better parallelization
max_workers = min(64, (os.cpu_count() or 1))      # Use up to 64 workers (1x cores) # Please change it according to your case

# Thread-safe file writing lock
file_write_lock = Lock()

# please change it by adding a directory that you like! 
# Base output directory structure
base_output_dir = ""

# ==================== FILTERING CONFIGURATION ====================

class FilteringConfig:
    """Configuration class for metadata filtering"""
    
    def __init__(self):
        # Enable/disable different filter types
        self.enable_newspaper_filter = True
        self.enable_date_filter = True
        self.enable_content_filter = True
        self.enable_geographic_filter = True
        
        # Newspaper filter settings
        self.allowed_newspapers = [
            'Algemeen Handelsblad',
            'De Telegraaf'
        ]
        
        # Date filter settings
        self.start_date = datetime(1880, 1, 1)
        self.end_date = datetime(1959, 12, 31)
        
        # Content filter settings
        self.min_title_length = 5
        self.max_title_length = 500
        self.required_content_types = ['artikel']
        self.blocked_content_types = [
            'advertentie', 
            'annonce',
            'familiebericht',
            'illustratie met onderschrift',
            'reclame',
            'advertissement'
        ]
        
        # Geographic filter settings
        self.allowed_geographic_regions = [
            'Landelijk',
            'Nederlands-Indië',
            'Indonesië', 
            'Nederlandse Antillen',
            'Suriname'
        ]
        
        # OCR text filter settings
        self.min_ocr_length = 100
        self.max_ocr_length = 50000
        
    def print_filter_summary(self):
        """Print summary of active filters"""
        print("🎯 MULTITHREADED FILTERS FOR 1880-1959:")
        print("=" * 60)
        print(f"🧵 Max Workers: {max_workers}")
        print(f"🖥️  CPU Cores: {os.cpu_count()}")
        print(f"📦 Batch Size: {records_per_batch}")
        
        if self.enable_newspaper_filter:
            print(f"📰 Newspaper Filter: {len(self.allowed_newspapers)} newspapers allowed")
        
        if self.enable_date_filter:
            print(f"📅 Date Filter: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        
        if self.enable_content_filter:
            print(f"📄 Content Filter: Types {self.required_content_types}")
        
        if self.enable_geographic_filter:
            print(f"🌍 Geographic Filter: {len(self.allowed_geographic_regions)} regions allowed")
        
        print(f"📝 OCR Length Filter: {self.min_ocr_length}-{self.max_ocr_length} characters")
        print()

# ==================== FILTERING FUNCTIONS ====================

def apply_metadata_filters(record_data, namespaces, config):
    """Apply all configured filters to a newspaper record"""
    try:
        # Extract metadata fields
        date_elem = record_data.find('.//dc:date', namespaces)
        type_elem = record_data.find('.//dc:type', namespaces)
        title_elem = record_data.find('.//dc:title', namespaces)
        papertitle_elem = record_data.find('.//ddd:papertitle', namespaces)
        spatial_elem = record_data.find('.//ddd:spatial', namespaces)
        
        # Extract text values
        date_str = date_elem.text if date_elem is not None else ""
        content_type = type_elem.text if type_elem is not None else ""
        title = title_elem.text if title_elem is not None else ""
        papertitle = papertitle_elem.text if papertitle_elem is not None else ""
        spatial = spatial_elem.text if spatial_elem is not None else ""
        
    except Exception as e:
        return False, "metadata_extraction_error"
    
    # Apply filters
    if config.enable_content_filter:
        if content_type not in config.required_content_types:
            return False, f"content_type_rejected: {content_type}"
        
        if content_type in config.blocked_content_types:
            return False, f"content_type_blocked: {content_type}"
        
        if len(title) < config.min_title_length or len(title) > config.max_title_length:
            return False, f"title_length_rejected: {len(title)}"
    
    if config.enable_date_filter and config.start_date and config.end_date:
        try:
            date_part = date_str.split(' ')[0] if ' ' in date_str else date_str
            record_date = datetime.strptime(date_part, '%Y/%m/%d')
            
            if not (config.start_date <= record_date <= config.end_date):
                return False, f"date_out_of_range: {date_part}"
        except:
            return False, f"date_parse_error: {date_str}"
    
    if config.enable_newspaper_filter:
        if not any(allowed_paper in papertitle for allowed_paper in config.allowed_newspapers):
            return False, f"newspaper_not_allowed: {papertitle}"
    
    if config.enable_geographic_filter:
        if not any(allowed_region in spatial for allowed_region in config.allowed_geographic_regions):
            return False, f"geographic_region_not_allowed: {spatial}"
    
    return True, "passed_all_filters"

def apply_ocr_filters(ocr_text, config):
    """Apply filters to OCR text content"""
    if not ocr_text:
        return False, "no_ocr_text"
    
    text_length = len(ocr_text.strip())
    
    if text_length < config.min_ocr_length:
        return False, f"ocr_too_short: {text_length}"
    
    if text_length > config.max_ocr_length:
        return False, f"ocr_too_long: {text_length}"
    
    return True, "ocr_passed"

# ==================== UTILITY FUNCTIONS ====================

def normalize_text(text):
    """Basic text normalization"""
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_date_components(date_str):
    """Extract year, month, day from date string"""
    if not date_str:
        return None, None, None
    
    patterns = [
        r'(\d{4})/(\d{1,2})/(\d{1,2})',
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, date_str)
        if match:
            year, month, day = match.groups()
            try:
                return str(int(year)), f"{int(month):02d}", f"{int(day):02d}"
            except:
                continue
    
    year_match = re.search(r'(\d{4})', date_str)
    if year_match:
        return year_match.group(1), "01", "01"
    
    return None, None, None

def get_decade_folder(year):
    """Get decade folder"""
    try:
        year_int = int(year)
        decade = (year_int // 10) * 10
        return f"{decade}x"
    except:
        return "unknownx"

def create_xml_content(ocr_text, title="", date="", identifier="", metadata_key="", papertitle="", spatial=""):
    """Create XML content with metadata"""
    root = ET.Element("article")
    
    metadata = ET.SubElement(root, "metadata")
    ET.SubElement(metadata, "title").text = title or ""
    ET.SubElement(metadata, "date").text = date or ""
    ET.SubElement(metadata, "identifier").text = identifier or ""
    ET.SubElement(metadata, "metadata_key").text = metadata_key or ""
    ET.SubElement(metadata, "newspaper").text = papertitle or ""
    ET.SubElement(metadata, "geographic").text = spatial or ""
    ET.SubElement(metadata, "collection").text = "DDD_artikel"
    ET.SubElement(metadata, "extraction_date").text = datetime.now().isoformat()
    ET.SubElement(metadata, "filtered").text = "true"
    
    content = ET.SubElement(root, "content")
    content.text = ocr_text or ""
    
    ET.indent(root, space="  ", level=0)
    return ET.tostring(root, encoding='unicode', xml_declaration=True)

def generate_file_path(date_str, metadata_key, collection="kranten"):
    """Generate file path based on metadata"""
    year, month, day = extract_date_components(date_str)
    
    if not year:
        year, month, day = "unknown", "01", "01"
    
    decade_folder = get_decade_folder(year)
    
    if metadata_key:
        parts = metadata_key.split(':')
        article_id = parts[-1] if parts else "0001"
        base_id = ':'.join(parts[:-1]) if len(parts) > 1 else metadata_key
    else:
        article_id = "0001"
        base_id = "unknown"
    
    folder_path = os.path.join(
        base_output_dir,
        f"kranten_pd_{decade_folder}",
        year,
        month,
        day,
        f"DDD_{base_id.replace(':', '_')}_mpeg21" if base_id != "unknown" else "DDD_unknown_mpeg21"
    )
    
    filename = f"DDD_{base_id.replace(':', '_')}_{article_id}_articletext.xml"
    
    return folder_path, filename

# ==================== THREAD-SAFE API FUNCTIONS ====================

def search_delpher_by_year_batch(year, start_record, collection="kranten", maximumRecords=100):
    """Search Delpher for a specific batch - thread-safe"""
    base_url = "http://jsru.kb.nl/sru/sru"
    date_query = f"date >= {year}-01-01 AND date < {year + 1}-01-01"
    collection_id = "DDD_artikel"
    
    params = {
        'operation': 'searchRetrieve',
        'x-collection': collection_id,
        'query': date_query,
        'maximumRecords': maximumRecords,
        'startRecord': start_record,
        'recordSchema': 'ddd'
    }
    
    session = get_session()
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = session.get(base_url, params=params, timeout=60)
            
            if response.status_code == 200:
                root = ET.fromstring(response.text)
                namespaces = {'srw': 'http://www.loc.gov/zing/srw/'}
                num_records = root.find('.//srw:numberOfRecords', namespaces)
                
                total_count = int(num_records.text) if num_records is not None else 0
                records = root.findall('.//srw:record', namespaces)
                
                return root, total_count, len(records)
            else:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                continue
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            continue
    
    return None, 0, 0

def extract_ocr_text_thread_safe(ocr_url):
    """Extract OCR text - thread-safe"""
    try:
        session = get_session()
        response = session.get(ocr_url, timeout=30)
        if response.status_code == 200:
            try:
                ocr_root = ET.fromstring(response.text)
                text_content = ""
                for text_elem in ocr_root.iter():
                    if text_elem.text and text_elem.text.strip():
                        text_content += text_elem.text.strip() + " "
                return text_content.strip()
            except:
                return response.text
        return None
    except Exception:
        return None

# ==================== MULTITHREADED PROCESSING FUNCTIONS ====================

def process_single_record(record_data, namespaces, config, year):
    """Process a single record - designed for threading"""
    try:
        # Apply metadata filters first
        metadata_passed, filter_reason = apply_metadata_filters(record_data, namespaces, config)
        
        if not metadata_passed:
            return None, filter_reason
        
        # Extract metadata
        title_elem = record_data.find('.//dc:title', namespaces)
        date_elem = record_data.find('.//dc:date', namespaces)
        metadata_key_elem = record_data.find('.//ddd:metadataKey', namespaces)
        papertitle_elem = record_data.find('.//ddd:papertitle', namespaces)
        spatial_elem = record_data.find('.//ddd:spatial', namespaces)
        
        title = normalize_text(title_elem.text) if title_elem is not None and title_elem.text else ""
        date_str = date_elem.text if date_elem is not None and date_elem.text else ""
        metadata_key = metadata_key_elem.text if metadata_key_elem is not None and metadata_key_elem.text else ""
        papertitle = papertitle_elem.text if papertitle_elem is not None and papertitle_elem.text else ""
        spatial = spatial_elem.text if spatial_elem is not None and spatial_elem.text else ""
        
        # Validate year
        record_year, month, day = extract_date_components(date_str)
        if not record_year or int(record_year) != year:
            return None, f"date_mismatch: {date_str}"
        
        # Extract OCR URL
        ocr_url = None
        for elem in record_data.findall('.//dc:identifier', namespaces):
            if elem.text and ':ocr' in elem.text:
                ocr_url = elem.text
                break
        
        if not ocr_url:
            return None, "no_ocr_url"
        
        # Extract OCR text
        ocr_text = extract_ocr_text_thread_safe(ocr_url)
        
        # Apply OCR filters
        ocr_passed, ocr_reason = apply_ocr_filters(ocr_text, config)
        
        if not ocr_passed:
            return None, ocr_reason
        
        # Normalize OCR text
        ocr_text = normalize_text(ocr_text)
        
        # Prepare data for file creation
        record_data = {
            'title': title,
            'date_str': date_str,
            'metadata_key': metadata_key,
            'papertitle': papertitle,
            'spatial': spatial,
            'ocr_text': ocr_text
        }
        
        return record_data, "success"
        
    except Exception as e:
        return None, f"processing_error: {str(e)}"

def save_record_to_file(record_data):
    """Save a single record to file - thread-safe"""
    try:
        # Generate file path
        folder_path, filename = generate_file_path(record_data['date_str'], record_data['metadata_key'])
        
        # Thread-safe directory creation
        with file_write_lock:
            os.makedirs(folder_path, exist_ok=True)
        
        # Create XML content
        xml_content = create_xml_content(
            record_data['ocr_text'],
            record_data['title'],
            record_data['date_str'],
            record_data['metadata_key'],
            record_data['metadata_key'],
            record_data['papertitle'],
            record_data['spatial']
        )
        
        # Save file
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        return True, file_path
        
    except Exception as e:
        return False, str(e)

def process_batch_multithreaded(batch_args):
    """Process a batch of records using multithreading"""
    year, start_record, config = batch_args
    
    try:
        # Get batch data from API
        root, total_count, record_count = search_delpher_by_year_batch(year, start_record, "kranten", records_per_batch)
        
        if root is None:
            return 0, {'api_error': record_count}, total_count
        
        namespaces = {
            'srw': 'http://www.loc.gov/zing/srw/',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'ddd': 'http://www.kb.nl/ddd',
            'tel': 'http://krait.kb.nl/coop/tel/handbook/telterms.html'
        }
        
        records = root.findall('.//srw:record', namespaces)
        
        if not records:
            return 0, {'no_records': 1}, total_count
        
        # Process records in parallel
        batch_filter_stats = {}
        successful_records = []
        
        # Use ThreadPoolExecutor for OCR extraction (I/O bound)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(records))) as executor:
            # Create processing tasks
            process_tasks = []
            for record in records:
                record_data = record.find('.//srw:recordData', namespaces)
                if record_data is not None:
                    future = executor.submit(process_single_record, record_data, namespaces, config, year)
                    process_tasks.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(process_tasks):
                try:
                    result, reason = future.result()
                    if result is not None:
                        successful_records.append(result)
                        batch_filter_stats['success'] = batch_filter_stats.get('success', 0) + 1
                    else:
                        batch_filter_stats[reason] = batch_filter_stats.get(reason, 0) + 1
                except Exception as e:
                    batch_filter_stats['thread_error'] = batch_filter_stats.get('thread_error', 0) + 1
        
        # Save successful records to files
        files_saved = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            save_tasks = [executor.submit(save_record_to_file, record) for record in successful_records]
            
            for future in concurrent.futures.as_completed(save_tasks):
                try:
                    success, result = future.result()
                    if success:
                        files_saved += 1
                    else:
                        batch_filter_stats['save_error'] = batch_filter_stats.get('save_error', 0) + 1
                except Exception:
                    batch_filter_stats['save_error'] = batch_filter_stats.get('save_error', 0) + 1
        
        batch_filter_stats['saved'] = files_saved
        
        return files_saved, batch_filter_stats, total_count
        
    except Exception as e:
        logging.error(f"Batch processing error for year {year}, start_record {start_record}: {e}")
        return 0, {'batch_error': 1}, 0

def process_year_multithreaded(year, config):
    """Process a full year using multithreading"""
    logging.info(f"Starting multithreaded processing for year: {year}")
    
    # First, get total count
    root, total_count, _ = search_delpher_by_year_batch(year, 1, "kranten", 1)
    if total_count == 0:
        logging.info(f"No records found for year {year}")
        return 0, {}
    
    logging.info(f"Year {year}: Found {total_count:,} total records")
    
    # Create batch arguments
    batch_args = []
    start_record = 1
    while start_record <= total_count:
        batch_args.append((year, start_record, config))
        start_record += records_per_batch
    
    logging.info(f"Year {year}: Processing {len(batch_args)} batches with {max_workers} workers")
    
    # Process batches in parallel
    total_files_saved = 0
    overall_filter_stats = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch processing tasks
        batch_futures = [executor.submit(process_batch_multithreaded, args) for args in batch_args]
        
        # Process results with progress bar
        with tqdm(total=len(batch_futures), desc=f"Year {year}", unit="batch") as pbar:
            for future in concurrent.futures.as_completed(batch_futures):
                try:
                    files_saved, filter_stats, _ = future.result()
                    total_files_saved += files_saved
                    
                    # Aggregate filter statistics
                    for key, count in filter_stats.items():
                        overall_filter_stats[key] = overall_filter_stats.get(key, 0) + count
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'saved': total_files_saved,
                        'batch_files': files_saved
                    })
                    
                except Exception as e:
                    logging.error(f"Batch future error: {e}")
                    pbar.update(1)
    
    logging.info(f"Year {year} completed: {total_files_saved:,} files saved")
    return total_files_saved, overall_filter_stats

# ==================== MAIN FUNCTION ====================

def main():
    print("=" * 80)
    print("MULTITHREADED DELPHER FILTERED NEWSPAPER EXTRACTOR (1880-1959)")
    print("=" * 80)
    print(f"🧵 Max Workers: {max_workers}")
    print(f"🖥️  CPU Cores Available: {os.cpu_count()}")
    print(f"📦 Records per batch: {records_per_batch}")
    print(f"📅 Extracting from {start_year} to {current_year}")
    print(f"📁 Output directory: {base_output_dir}")
    
    # Initialize filtering configuration
    config = FilteringConfig()
    config.print_filter_summary()
    
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Generate years to process
    years = list(range(start_year, current_year + 1))
    
    print(f"\n📋 Years to process: {len(years)}")
    print(f"   {years}")
    
    # Confirm before starting
    response = input(f"\nProceed with MULTITHREADED processing? (y/n): ")
    if response.lower() != 'y':
        print("Processing cancelled.")
        return
    
    start_time = time.time()
    total_files = 0
    successful_years = 0
    overall_filter_stats = {}
    
    print(f"\n🚀 Starting multithreaded extraction...")
    
    # Process years sequentially (but batches within years in parallel)
    for i, year in enumerate(years, 1):
        year_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Processing Year {year} ({i}/{len(years)}) - MULTITHREADED")
        print(f"{'='*60}")
        
        try:
            files_saved, year_filter_stats = process_year_multithreaded(year, config)
            total_files += files_saved
            successful_years += 1
            
            # Aggregate statistics
            for key, count in year_filter_stats.items():
                overall_filter_stats[key] = overall_filter_stats.get(key, 0) + count
            
            year_duration = time.time() - year_start_time
            
            print(f"✅ Year {year} COMPLETE: {files_saved:,} files in {year_duration/60:.1f} minutes")
            
        except Exception as e:
            logging.error(f"❌ FAILED to process year {year}: {e}")
            print(f"❌ Year {year} FAILED: {e}")
            continue
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Final summary
    print(f"\n" + "=" * 80)
    print("🎉 MULTITHREADED EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"⏱️  Total time: {total_duration/60:.1f} minutes ({total_duration/3600:.1f} hours)")
    print(f"📄 Total files saved: {total_files:,}")
    print(f"📈 Processing rate: {total_files/(total_duration/60):.1f} files/minute")
    print(f"✅ Successful years: {successful_years}/{len(years)}")
    
    # Print filter statistics
    print(f"\n📊 FILTER STATISTICS:")
    print("=" * 50)
    total_processed = sum(overall_filter_stats.values())
    if total_processed > 0:
        for reason, count in sorted(overall_filter_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_processed) * 100
            print(f"  {reason}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n🎯 Multithreading efficiency achieved!")
    print(f"💾 Files saved to: {base_output_dir}")

if __name__ == "__main__":
    main()