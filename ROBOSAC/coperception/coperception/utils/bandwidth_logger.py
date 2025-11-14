"""
Bandwidth measurement utility for ROBOSAC communication.
Tracks actual bytes transmitted between agents during feature fusion.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional
import time


class BandwidthLogger:
    """Logs actual bandwidth usage during agent communication."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "bandwidth"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Track transmissions per frame
        self.current_frame_transmissions: List[Dict] = []
        self.total_bytes_per_agent: Dict[int, int] = {}
        self.transmission_count = 0
        
        # CSV file setup
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.log_dir / f"{experiment_name}_{timestamp}.csv"
        self._write_csv_header()
    
    def _write_csv_header(self):
        """Write CSV header for bandwidth log."""
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'frame_id', 'sender_agent', 'receiver_agent', 
                'bytes_transmitted', 'effective_bytes', 'feature_shape', 
                'tensor_dtype', 'compression_ratio', 'total_frame_bytes'
            ])
    
    def log_transmission(self, sender_agent: int, receiver_agent: int, 
                        bytes_transmitted: int, feature_shape: tuple,
                        tensor_dtype: str = "unknown", compression_ratio: float = 1.0):
        """Log a single transmission between agents with enhanced details."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Track per-agent totals
        if sender_agent not in self.total_bytes_per_agent:
            self.total_bytes_per_agent[sender_agent] = 0
        self.total_bytes_per_agent[sender_agent] += bytes_transmitted
        
        # Calculate effective bytes (considering compression)
        effective_bytes = int(bytes_transmitted * compression_ratio)
        
        # Store transmission record with enhanced details
        transmission = {
            'timestamp': timestamp,
            'frame_id': self.transmission_count,
            'sender_agent': sender_agent,
            'receiver_agent': receiver_agent,
            'bytes_transmitted': bytes_transmitted,
            'effective_bytes': effective_bytes,
            'feature_shape': str(feature_shape),
            'tensor_dtype': tensor_dtype,
            'compression_ratio': compression_ratio,
            'total_frame_bytes': sum(self.total_bytes_per_agent.values())
        }
        
        self.current_frame_transmissions.append(transmission)
        self.transmission_count += 1
    
    def finalize_frame(self):
        """Finalize current frame and write to CSV."""
        if not self.current_frame_transmissions:
            return
            
        # Write all transmissions for this frame
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for transmission in self.current_frame_transmissions:
                writer.writerow([
                    transmission['timestamp'],
                    transmission['frame_id'],
                    transmission['sender_agent'],
                    transmission['receiver_agent'],
                    transmission['bytes_transmitted'],
                    transmission['effective_bytes'],
                    transmission['feature_shape'],
                    transmission['tensor_dtype'],
                    transmission['compression_ratio'],
                    transmission['total_frame_bytes']
                ])
        
        # Reset for next frame
        self.current_frame_transmissions.clear()
    
    def get_summary(self) -> Dict:
        """Get bandwidth usage summary."""
        total_bytes = sum(self.total_bytes_per_agent.values())
        num_senders = len(self.total_bytes_per_agent)  # 송신자 수
        
        # 프레임 수 계산 (CSV 파일에서)
        frame_count = 0
        if self.csv_file.exists():
            with self.csv_file.open('r', encoding='utf-8') as f:
                frame_count = sum(1 for line in f) - 1  # 헤더 제외
        
        return {
            'total_bytes': total_bytes,
            'num_agents': num_senders,  # 송신자 수 반환
            'num_senders': num_senders,  # 명시적으로 송신자 수
            'bytes_per_agent': self.total_bytes_per_agent.copy(),
            'avg_bytes_per_agent': total_bytes / num_senders if num_senders > 0 else 0,
            'frame_count': frame_count,
            'transmission_count': self.transmission_count,  # 실제 전송 횟수
            'csv_file': str(self.csv_file)
        }
    
    def reset(self):
        """Reset logger for new experiment."""
        self.current_frame_transmissions.clear()
        self.total_bytes_per_agent.clear()
        self.transmission_count = 0


# Global bandwidth logger instance
_global_bandwidth_logger: Optional[BandwidthLogger] = None


def get_bandwidth_logger() -> Optional[BandwidthLogger]:
    """Get the global bandwidth logger instance."""
    return _global_bandwidth_logger


def set_bandwidth_logger(logger: BandwidthLogger):
    """Set the global bandwidth logger instance."""
    global _global_bandwidth_logger
    _global_bandwidth_logger = logger


def init_bandwidth_logger(log_dir: str = "logs", experiment_name: str = "bandwidth") -> BandwidthLogger:
    """Initialize and set the global bandwidth logger."""
    logger = BandwidthLogger(log_dir, experiment_name)
    set_bandwidth_logger(logger)
    return logger
