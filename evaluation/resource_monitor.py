#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Resource monitoring utilities for benchmarking ASR systems
Tracks GPU memory, CPU usage, and system resources during transcription
"""

import psutil
import time
import threading
from typing import List, Dict, Optional
import numpy as np

# Try to import GPU monitoring libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not available. GPU monitoring disabled.")

try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except (ImportError, Exception):
    NVML_AVAILABLE = False
    print("[WARNING] pynvml not available. NVIDIA GPU monitoring limited.")


class ResourceMonitor:
    """Monitor system resources during processing"""
    
    def __init__(self, interval: float = 0.5):
        """
        Initialize resource monitor
        
        Args:
            interval: Sampling interval in seconds (default 0.5s)
        """
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Storage for metrics
        self.gpu_memory_used = []  # MB
        self.gpu_utilization = []  # Percentage
        self.cpu_percent = []  # Percentage
        self.ram_used = []  # MB
        self.timestamps = []  # Relative timestamps
        
        # Process handle for CPU/RAM tracking
        self.process = psutil.Process()
        
        # Start time
        self.start_time = None
        
    def start(self):
        """Start monitoring resources"""
        self.monitoring = True
        self.start_time = time.time()
        
        # Clear previous data
        self.gpu_memory_used.clear()
        self.gpu_utilization.clear()
        self.cpu_percent.clear()
        self.ram_used.clear()
        self.timestamps.clear()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring resources"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                current_time = time.time() - self.start_time
                self.timestamps.append(current_time)
                
                # CPU and RAM usage
                cpu = self.process.cpu_percent(interval=None)
                ram = self.process.memory_info().rss / (1024 * 1024)  # Convert to MB
                self.cpu_percent.append(cpu)
                self.ram_used.append(ram)
                
                # GPU monitoring
                gpu_mem = 0
                gpu_util = 0
                
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    # Get GPU memory using PyTorch
                    gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                    
                    # Try to get utilization via NVML
                    if NVML_AVAILABLE:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_util = util.gpu
                        except Exception:
                            pass
                
                self.gpu_memory_used.append(gpu_mem)
                self.gpu_utilization.append(gpu_util)
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"[WARNING] Error in resource monitoring: {e}")
                time.sleep(self.interval)
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics of resource usage
        
        Returns:
            Dictionary with resource usage statistics
        """
        summary = {
            'gpu_memory': {
                'peak_mb': max(self.gpu_memory_used) if self.gpu_memory_used else 0,
                'mean_mb': np.mean(self.gpu_memory_used) if self.gpu_memory_used else 0,
                'min_mb': min(self.gpu_memory_used) if self.gpu_memory_used else 0,
            },
            'gpu_utilization': {
                'peak_pct': max(self.gpu_utilization) if self.gpu_utilization else 0,
                'mean_pct': np.mean(self.gpu_utilization) if self.gpu_utilization else 0,
            },
            'cpu': {
                'peak_pct': max(self.cpu_percent) if self.cpu_percent else 0,
                'mean_pct': np.mean(self.cpu_percent) if self.cpu_percent else 0,
            },
            'ram': {
                'peak_mb': max(self.ram_used) if self.ram_used else 0,
                'mean_mb': np.mean(self.ram_used) if self.ram_used else 0,
                'min_mb': min(self.ram_used) if self.ram_used else 0,
            },
            'duration_s': self.timestamps[-1] if self.timestamps else 0,
            'samples': len(self.timestamps)
        }
        
        return summary
    
    def get_time_series(self) -> Dict:
        """
        Get time series data for plotting
        
        Returns:
            Dictionary with time series arrays
        """
        return {
            'timestamps': self.timestamps.copy(),
            'gpu_memory_mb': self.gpu_memory_used.copy(),
            'gpu_util_pct': self.gpu_utilization.copy(),
            'cpu_pct': self.cpu_percent.copy(),
            'ram_mb': self.ram_used.copy()
        }


def calculate_resource_efficiency(monitor_data: Dict, audio_duration: float) -> Dict:
    """
    Calculate resource efficiency metrics
    
    Args:
        monitor_data: Resource monitoring summary from get_summary()
        audio_duration: Duration of audio in seconds
        
    Returns:
        Dictionary with efficiency metrics
    """
    metrics = {}
    
    # Memory efficiency: MB per second of audio
    if audio_duration > 0:
        metrics['gpu_memory_per_second'] = monitor_data['gpu_memory']['peak_mb'] / audio_duration
        metrics['ram_per_second'] = monitor_data['ram']['peak_mb'] / audio_duration
    else:
        metrics['gpu_memory_per_second'] = 0
        metrics['ram_per_second'] = 0
    
    # Compute intensity: GPU utilization percentage
    metrics['gpu_utilization'] = monitor_data['gpu_utilization']['mean_pct']
    metrics['cpu_utilization'] = monitor_data['cpu']['mean_pct']
    
    # Memory growth rate (useful for identifying memory leaks)
    # This would require time series analysis
    
    return metrics


def print_resource_summary(name: str, summary: Dict, audio_duration: float):
    """
    Print formatted resource usage summary
    
    Args:
        name: Name of the system being tested
        summary: Resource summary from get_summary()
        audio_duration: Duration of audio in seconds
    """
    print(f"\n--- {name} Resource Usage ---")
    print(f"GPU Memory:")
    print(f"  Peak: {summary['gpu_memory']['peak_mb']:.1f} MB")
    print(f"  Mean: {summary['gpu_memory']['mean_mb']:.1f} MB")
    print(f"  Per Second: {summary['gpu_memory']['peak_mb']/audio_duration:.2f} MB/s")
    
    print(f"GPU Utilization:")
    print(f"  Peak: {summary['gpu_utilization']['peak_pct']:.1f}%")
    print(f"  Mean: {summary['gpu_utilization']['mean_pct']:.1f}%")
    
    print(f"RAM Usage:")
    print(f"  Peak: {summary['ram']['peak_mb']:.1f} MB")
    print(f"  Mean: {summary['ram']['mean_mb']:.1f} MB")
    print(f"  Per Second: {summary['ram']['peak_mb']/audio_duration:.2f} MB/s")
    
    print(f"CPU Utilization:")
    print(f"  Peak: {summary['cpu']['peak_pct']:.1f}%")
    print(f"  Mean: {summary['cpu']['mean_pct']:.1f}%")
    
    print(f"Monitoring Duration: {summary['duration_s']:.1f}s ({summary['samples']} samples)")
