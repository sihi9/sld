import torch
import torch.nn as nn
import gc
import psutil
import os
from typing import Dict, List, Tuple
import time
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np

class MemoryProfiler:
    """
    Comprehensive memory profiler for PyTorch models, especially useful for
    analyzing spiking neural networks with temporal dynamics.
    """
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.baseline_memory = 0
        self.measurements = []
    
    def get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**2,
                'cached': torch.cuda.memory_reserved() / 1024**2,
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**2,
                'max_cached': torch.cuda.max_memory_reserved() / 1024**2
            }
        return {'allocated': 0, 'cached': 0, 'max_allocated': 0, 'max_cached': 0}
    
    def get_cpu_memory(self) -> float:
        """Get current CPU memory usage in MB."""
        return self.process.memory_info().rss / 1024**2
    
    def reset_peak_memory(self):
        """Reset peak memory tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.peak_memory = 0
        self.baseline_memory = self.get_cpu_memory()
    
    @contextmanager
    def profile_memory(self, label: str = ""):
        """Context manager to profile memory usage."""
        # Clean up before measurement
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Baseline measurements
        start_cpu = self.get_cpu_memory()
        start_gpu = self.get_gpu_memory()
        
        print(f"\n{'='*50}")
        print(f"Memory Profile: {label}")
        print(f"{'='*50}")
        print(f"Start CPU Memory: {start_cpu:.2f} MB")
        print(f"Start GPU Memory: {start_gpu['allocated']:.2f} MB (allocated)")
        
        try:
            yield self
        finally:
            # Final measurements
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_cpu = self.get_cpu_memory()
            end_gpu = self.get_gpu_memory()
            
            cpu_diff = end_cpu - start_cpu
            gpu_diff = end_gpu['allocated'] - start_gpu['allocated']
            
            print(f"\nEnd CPU Memory: {end_cpu:.2f} MB")
            print(f"End GPU Memory: {end_gpu['allocated']:.2f} MB (allocated)")
            print(f"CPU Memory Diff: {cpu_diff:+.2f} MB")
            print(f"GPU Memory Diff: {gpu_diff:+.2f} MB")
            print(f"Peak GPU Memory: {end_gpu['max_allocated']:.2f} MB")
            print(f"{'='*50}\n")
            
            # Store measurement
            self.measurements.append({
                'label': label,
                'cpu_start': start_cpu,
                'cpu_end': end_cpu,
                'cpu_diff': cpu_diff,
                'gpu_start': start_gpu['allocated'],
                'gpu_end': end_gpu['allocated'],
                'gpu_diff': gpu_diff,
                'gpu_peak': end_gpu['max_allocated']
            })


class SpikingUNetMemoryAnalyzer:
    """
    Specialized analyzer for SpikingUNet memory consumption.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.profiler = MemoryProfiler()
        
    def analyze_model_components(self) -> Dict:
        """Analyze memory consumption of different model components."""
        results = {}
        
        # Model parameters
        param_memory = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
        results['parameters_mb'] = param_memory
        
        # Model buffers (includes neuron states)
        buffer_memory = sum(b.numel() * b.element_size() for b in self.model.buffers()) / 1024**2
        results['buffers_mb'] = buffer_memory
        
        # Count spiking neurons
        neuron_count = 0
        for module in self.model.modules():
            if 'LIF' in module.__class__.__name__ or 'Neuron' in module.__class__.__name__:
                neuron_count += 1
        results['neuron_count'] = neuron_count
        
        return results
    
    def profile_inference_sequence(self, 
                                 input_shape: Tuple[int, ...],
                                 timesteps: int = 10,
                                 batch_size: int = 1,
                                 with_gradients: bool = False) -> Dict:
        """
        Profile memory usage during a full inference sequence.
        
        Args:
            input_shape: Shape of single input (C, H, W)
            timesteps: Number of timesteps to simulate
            batch_size: Batch size
            with_gradients: Whether to keep gradients (training mode)
        """
        torch.cuda.reset_peak_memory_stats()  # Reset peak counter
        # Create input tensor [T, B, C, H, W]
        full_input_shape = (timesteps, batch_size) + input_shape
        
        mode_label = "with gradients" if with_gradients else "inference only"
        
        with self.profiler.profile_memory(f"Full Sequence ({mode_label})"):
            # Reset model states
            for m in self.model.modules():
                if hasattr(m, 'reset'):
                    m.reset()
            
            # Create input
            x = torch.randn(full_input_shape, device=self.device)
            
            if with_gradients:
                x.requires_grad_(True)
                self.model.train()
                output = self.model(x)
                # Simulate backward pass
                loss = output.sum()
                loss.backward()
            else:
                self.model.eval()
                with torch.no_grad():
                    output = self.model(x)
            
        return self.profiler.measurements[-1]
    
    def profile_timestep_by_timestep(self, 
                                   input_shape: Tuple[int, ...],
                                   timesteps: int = 10,
                                   batch_size: int = 1) -> List[Dict]:
        """
        Profile memory usage timestep by timestep to see accumulation.
        """
        single_input_shape = (batch_size,) + input_shape
        timestep_results = []
        
        # Reset model states
        for m in self.model.modules():
            if hasattr(m, 'reset'):
                m.reset()
        
        self.model.eval()
        
        print(f"\nTimestep-by-timestep memory analysis:")
        print(f"Input shape per timestep: {single_input_shape}")
        
        base_gpu = self.profiler.get_gpu_memory()['allocated']
        
        for t in range(timesteps):
            with torch.no_grad():
                x_t = torch.randn(single_input_shape, device=self.device)
                
                # Manual single-step forward (if your model supports it)
                if hasattr(self.model, '_forward_single_step'):
                    output_t = self.model._forward_single_step(x_t)
                else:
                    # Fallback: use full model with single timestep
                    x_input = x_t.unsqueeze(0)  # Add time dimension
                    output_t = self.model(x_input)
                
                current_gpu = self.profiler.get_gpu_memory()['allocated']
                memory_increase = current_gpu - base_gpu
                
                timestep_results.append({
                    'timestep': t,
                    'gpu_memory_mb': current_gpu,
                    'memory_increase_mb': memory_increase
                })
                
                if t % 5 == 0 or t == timesteps - 1:
                    print(f"Timestep {t:2d}: {current_gpu:.2f} MB GPU "
                          f"(+{memory_increase:.2f} MB from baseline)")
        
        return timestep_results
    
    def compare_sequence_vs_stepwise(self, 
                                   input_shape: Tuple[int, ...],
                                   timesteps: int = 10,
                                   batch_size: int = 1) -> Dict:
        """
        Compare memory usage between full sequence processing and stepwise processing.
        """
        results = {}
        
        # Method 1: Full sequence
        print("\n" + "="*60)
        print("COMPARISON: Sequence vs Stepwise Processing")
        print("="*60)
        
        full_sequence_result = self.profile_inference_sequence(
            input_shape, timesteps, batch_size, with_gradients=False
        )
        results['full_sequence'] = full_sequence_result
        
        # Method 2: Stepwise (simulated)
        with self.profiler.profile_memory("Stepwise Processing"):
            # Reset model
            for m in self.model.modules():
                if hasattr(m, 'reset'):
                    m.reset()
            
            self.model.eval()
            outputs = []
            
            with torch.no_grad():
                for t in range(timesteps):
                    x_t = torch.randn((batch_size,) + input_shape, device=self.device)
                    
                    # Add time dimension for model
                    x_input = x_t.unsqueeze(0)
                    output_t = self.model(x_input)
                    outputs.append(output_t)
                    
                    # Optional: Clear intermediate results to save memory
                    # del x_input, output_t
        
        results['stepwise'] = self.profiler.measurements[-1]
        
        # Analysis
        seq_peak = results['full_sequence']['gpu_peak']
        step_peak = results['stepwise']['gpu_peak']
        
        print(f"\nMemory Comparison Summary:")
        print(f"Full Sequence Peak GPU: {seq_peak:.2f} MB")
        print(f"Stepwise Peak GPU: {step_peak:.2f} MB")
        print(f"Memory Savings (stepwise): {seq_peak - step_peak:.2f} MB "
              f"({((seq_peak - step_peak) / seq_peak * 100):.1f}%)")
        
        return results
    
    def generate_memory_report(self, 
                             input_shape: Tuple[int, ...] = (1, 128, 128),
                             timesteps_range: List[int] = [30],
                             batch_sizes: List[int] = [1]) -> Dict:
        """
        Generate comprehensive memory usage report.
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE MEMORY ANALYSIS FOR SPIKING UNET")
        print("="*80)
        
        report = {
            'model_info': self.analyze_model_components(),
            'scaling_tests': {},
            'gradient_comparison': {}
        }
        
        print(f"\nModel Information:")
        print(f"Parameters: {report['model_info']['parameters_mb']:.2f} MB")
        print(f"Buffers: {report['model_info']['buffers_mb']:.2f} MB")
        print(f"Neuron Count: {report['model_info']['neuron_count']}")
        
        # Test different configurations
        for batch_size in batch_sizes:
            for timesteps in timesteps_range:
                test_key = f"B{batch_size}_T{timesteps}"
                
                print(f"\nTesting: Batch Size {batch_size}, Timesteps {timesteps}")
                
                # Inference without gradients
                result_no_grad = self.profile_inference_sequence(
                    input_shape, timesteps, batch_size, with_gradients=False
                )
                
                # Inference with gradients
                result_with_grad = self.profile_inference_sequence(
                    input_shape, timesteps, batch_size, with_gradients=True
                )
                
                report['scaling_tests'][test_key] = {
                    'no_gradients': result_no_grad,
                    'with_gradients': result_with_grad,
                    'gradient_overhead_mb': (result_with_grad['gpu_peak'] - 
                                           result_no_grad['gpu_peak'])
                }
        
        return report
    
    def plot_memory_scaling(self, report: Dict):
        """Plot memory scaling results."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Extract data for plotting
            batch_sizes = []
            timesteps_list = []
            mem_no_grad = []
            mem_with_grad = []
            gradient_overhead = []
            
            for test_key, results in report['scaling_tests'].items():
                batch_size = int(test_key.split('_')[0][1:])
                timesteps = int(test_key.split('_')[1][1:])
                
                batch_sizes.append(batch_size)
                timesteps_list.append(timesteps)
                mem_no_grad.append(results['no_gradients']['gpu_peak'])
                mem_with_grad.append(results['with_gradients']['gpu_peak'])
                gradient_overhead.append(results['gradient_overhead_mb'])
            
            # Plot 1: Memory vs Timesteps (for each batch size)
            unique_batches = sorted(set(batch_sizes))
            for batch in unique_batches:
                mask = [b == batch for b in batch_sizes]
                ts = [t for i, t in enumerate(timesteps_list) if mask[i]]
                mem_ng = [m for i, m in enumerate(mem_no_grad) if mask[i]]
                mem_wg = [m for i, m in enumerate(mem_with_grad) if mask[i]]
                
                ax1.plot(ts, mem_ng, 'o-', label=f'B{batch} (no grad)')
                ax1.plot(ts, mem_wg, 's--', label=f'B{batch} (with grad)')
            
            ax1.set_xlabel('Timesteps')
            ax1.set_ylabel('Peak GPU Memory (MB)')
            ax1.set_title('Memory vs Timesteps')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Memory vs Batch Size
            unique_timesteps = sorted(set(timesteps_list))
            for ts in unique_timesteps:
                mask = [t == ts for t in timesteps_list]
                bs = [b for i, b in enumerate(batch_sizes) if mask[i]]
                mem_ng = [m for i, m in enumerate(mem_no_grad) if mask[i]]
                mem_wg = [m for i, m in enumerate(mem_with_grad) if mask[i]]
                
                ax2.plot(bs, mem_ng, 'o-', label=f'T{ts} (no grad)')
                ax2.plot(bs, mem_wg, 's--', label=f'T{ts} (with grad)')
            
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Peak GPU Memory (MB)')
            ax2.set_title('Memory vs Batch Size')
            ax2.legend()
            ax2.grid(True)
            
            # Plot 3: Gradient Overhead
            scatter = ax3.scatter(timesteps_list, gradient_overhead, 
                                c=batch_sizes, s=100, alpha=0.7, cmap='viridis')
            ax3.set_xlabel('Timesteps')
            ax3.set_ylabel('Gradient Overhead (MB)')
            ax3.set_title('Memory Overhead from Gradients')
            plt.colorbar(scatter, ax=ax3, label='Batch Size')
            ax3.grid(True)
            
            # Plot 4: Memory Efficiency (memory per sample)
            efficiency_no_grad = [m/b for m, b in zip(mem_no_grad, batch_sizes)]
            efficiency_with_grad = [m/b for m, b in zip(mem_with_grad, batch_sizes)]
            
            ax4.scatter(timesteps_list, efficiency_no_grad, 
                       c='blue', alpha=0.7, label='No gradients')
            ax4.scatter(timesteps_list, efficiency_with_grad, 
                       c='red', alpha=0.7, label='With gradients')
            ax4.set_xlabel('Timesteps')
            ax4.set_ylabel('Memory per Sample (MB)')
            ax4.set_title('Memory Efficiency')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Plotting failed: {e}")
            print("You may need to install matplotlib: pip install matplotlib")


# Example usage function
def profile_spiking_unet_memory(model, device='cuda'):
    """
    Main function to run comprehensive memory analysis.
    """
    analyzer = SpikingUNetMemoryAnalyzer(model, device)
    
    # Generate comprehensive report
    report = analyzer.generate_memory_report(
        input_shape=(1, 128, 128),  # Adjust based on your model
        timesteps_range=[5, 10, 20],
        batch_sizes=[1, 2, 4]
    )
    
    # Additional specific tests
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSES")
    print("="*60)
    
    # Timestep-by-timestep analysis
    timestep_results = analyzer.profile_timestep_by_timestep(
        input_shape=(1, 128, 128),
        timesteps=10,
        batch_size=1
    )
    
    # Sequence vs stepwise comparison
    comparison = analyzer.compare_sequence_vs_stepwise(
        input_shape=(1, 128, 128),
        timesteps=10,
        batch_size=1
    )
    
    # Plot results
    analyzer.plot_memory_scaling(report)
    
    return report, timestep_results, comparison


# Usage example:
"""
# Assuming you have your SpikingUNetRNN model instantiated
model = SpikingUNetRNN(
    in_channels=1,
    out_channels=1,
    input_size=(128, 128),
    features=(64, 128, 256),
    # ... other parameters
)

# Move to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Run memory analysis
report, timestep_results, comparison = profile_spiking_unet_memory(model, device)

# Print summary
print("\nFinal Summary:")
print(f"Model has {report['model_info']['parameters_mb']:.2f} MB in parameters")
print(f"Model has {report['model_info']['neuron_count']} spiking neurons")
print("See detailed results in the report dictionary!")
"""