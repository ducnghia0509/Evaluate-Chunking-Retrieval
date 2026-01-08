"""
Chunking Logger - Comprehensive logging and statistics for chunking operations
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import statistics


class ChunkingLogger:
    """Logger for tracking chunking operations and statistics"""
    
    def __init__(self, strategy_name: str, output_dir: str = "../../logs/chunking"):
        """
        Initialize the chunking logger
        
        Args:
            strategy_name: Name of the chunking strategy
            output_dir: Directory to save log files
        """
        self.strategy_name = strategy_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'strategy': strategy_name,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'duration_seconds': None,
            'files_processed': 0,
            'total_chunks': 0,
            'chunks_by_file': {},
            'token_stats': {
                'total_tokens': 0,
                'min_tokens': float('inf'),
                'max_tokens': 0,
                'avg_tokens': 0,
                'median_tokens': 0,
                'token_distribution': []
            },
            'errors': [],
            'warnings': []
        }
        
        # Setup logging
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_dir / f"{strategy_name}_{self.timestamp}.log"
        
        self.logger = logging.getLogger(f'chunking.{strategy_name}')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info(f"Initialized chunking logger for strategy: {strategy_name}")
        self.logger.info(f"Log file: {log_file}")
    
    def log_file_start(self, file_path: str):
        """Log the start of processing a file"""
        self.logger.info(f"Processing file: {file_path}")
    
    def log_file_complete(self, file_path: str, num_chunks: int, file_stats: Dict[str, Any] = None):
        """
        Log completion of file processing
        
        Args:
            file_path: Path to the processed file
            num_chunks: Number of chunks created
            file_stats: Additional statistics for this file
        """
        self.stats['files_processed'] += 1
        self.stats['total_chunks'] += num_chunks
        self.stats['chunks_by_file'][file_path] = {
            'num_chunks': num_chunks,
            'stats': file_stats or {}
        }
        
        self.logger.info(f"Completed: {file_path} -> {num_chunks} chunks")
        
        if file_stats:
            self.logger.debug(f"File stats: {json.dumps(file_stats, indent=2)}")
    
    def log_chunk_stats(self, chunks: List[Dict[str, Any]]):
        """
        Log and track statistics for a list of chunks
        
        Args:
            chunks: List of chunk dictionaries
        """
        if not chunks:
            return
        
        token_counts = []
        for chunk in chunks:
            token_count = chunk.get('metadata', {}).get('token_count', 0)
            if token_count > 0:
                token_counts.append(token_count)
        
        if token_counts:
            # Update global token stats
            self.stats['token_stats']['total_tokens'] += sum(token_counts)
            self.stats['token_stats']['min_tokens'] = min(
                self.stats['token_stats']['min_tokens'], 
                min(token_counts)
            )
            self.stats['token_stats']['max_tokens'] = max(
                self.stats['token_stats']['max_tokens'], 
                max(token_counts)
            )
            self.stats['token_stats']['token_distribution'].extend(token_counts)
            
            # Log chunk stats
            self.logger.debug(f"Chunk token stats - Min: {min(token_counts)}, "
                            f"Max: {max(token_counts)}, "
                            f"Avg: {sum(token_counts)/len(token_counts):.2f}")
    
    def log_error(self, file_path: str, error: Exception):
        """Log an error during processing"""
        error_msg = f"Error processing {file_path}: {str(error)}"
        self.logger.error(error_msg)
        self.stats['errors'].append({
            'file': file_path,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })
    
    def log_warning(self, message: str):
        """Log a warning"""
        self.logger.warning(message)
        self.stats['warnings'].append({
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    
    def finalize(self):
        """Finalize logging and save statistics"""
        self.stats['end_time'] = datetime.now().isoformat()
        
        # Calculate duration
        start = datetime.fromisoformat(self.stats['start_time'])
        end = datetime.fromisoformat(self.stats['end_time'])
        duration = (end - start).total_seconds()
        self.stats['duration_seconds'] = duration
        
        # Calculate final token statistics
        token_dist = self.stats['token_stats']['token_distribution']
        if token_dist:
            self.stats['token_stats']['avg_tokens'] = statistics.mean(token_dist)
            self.stats['token_stats']['median_tokens'] = statistics.median(token_dist)
            self.stats['token_stats']['std_dev_tokens'] = statistics.stdev(token_dist) if len(token_dist) > 1 else 0
            
            # Add percentiles
            sorted_tokens = sorted(token_dist)
            self.stats['token_stats']['percentiles'] = {
                'p25': sorted_tokens[len(sorted_tokens) // 4],
                'p50': sorted_tokens[len(sorted_tokens) // 2],
                'p75': sorted_tokens[3 * len(sorted_tokens) // 4],
                'p90': sorted_tokens[9 * len(sorted_tokens) // 10],
                'p95': sorted_tokens[95 * len(sorted_tokens) // 100],
                'p99': sorted_tokens[99 * len(sorted_tokens) // 100] if len(sorted_tokens) >= 100 else sorted_tokens[-1]
            }
        
        # Remove the raw distribution to save space
        del self.stats['token_stats']['token_distribution']
        
        # Save statistics to JSON
        stats_file = self.output_dir / f"{self.strategy_name}_{self.timestamp}_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        # Log summary
        self.logger.info("=" * 80)
        self.logger.info("CHUNKING SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Strategy: {self.strategy_name}")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        self.logger.info(f"Files processed: {self.stats['files_processed']}")
        self.logger.info(f"Total chunks: {self.stats['total_chunks']}")
        
        if self.stats['total_chunks'] > 0:
            avg_chunks_per_file = self.stats['total_chunks'] / self.stats['files_processed']
            self.logger.info(f"Avg chunks per file: {avg_chunks_per_file:.2f}")
        
        if token_dist:
            self.logger.info(f"Token statistics:")
            self.logger.info(f"  - Total tokens: {self.stats['token_stats']['total_tokens']:,}")
            self.logger.info(f"  - Min tokens: {self.stats['token_stats']['min_tokens']}")
            self.logger.info(f"  - Max tokens: {self.stats['token_stats']['max_tokens']}")
            self.logger.info(f"  - Avg tokens: {self.stats['token_stats']['avg_tokens']:.2f}")
            self.logger.info(f"  - Median tokens: {self.stats['token_stats']['median_tokens']:.2f}")
            self.logger.info(f"  - Std dev: {self.stats['token_stats']['std_dev_tokens']:.2f}")
            self.logger.info(f"  - P95: {self.stats['token_stats']['percentiles']['p95']}")
        
        if self.stats['errors']:
            self.logger.error(f"Errors encountered: {len(self.stats['errors'])}")
        
        if self.stats['warnings']:
            self.logger.warning(f"Warnings: {len(self.stats['warnings'])}")
        
        self.logger.info(f"Statistics saved to: {stats_file}")
        self.logger.info("=" * 80)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return self.stats.copy()
