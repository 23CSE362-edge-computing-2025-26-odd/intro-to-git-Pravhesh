#!/usr/bin/env python3
"""
Deployment Simulation Script

This script simulates different deployment scenarios for the ECG diagnosis system,
analyzing performance, resource usage, and costs across various fog/edge architectures.

It provides what-if analysis for deployment decisions without requiring iFogSim.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
import time
import csv

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))


@dataclass
class SimulationConfig:
    """Configuration for deployment simulation."""
    simulation_duration_minutes: int = 60
    ecg_arrival_rate_per_minute: int = 6  # Every 10 seconds
    network_reliability: float = 0.99
    enable_failover: bool = True
    cost_analysis: bool = True


@dataclass  
class DeviceSpecs:
    """Device specifications for simulation."""
    name: str
    cpu_cores: int
    memory_mb: int
    storage_gb: int
    power_watts_idle: float
    power_watts_busy: float
    cost_per_hour: float
    bandwidth_mbps: float
    latency_to_next_tier_ms: float


@dataclass
class TaskProfile:
    """ECG processing task profile."""
    name: str
    cpu_time_ms: float
    memory_mb: int
    input_size_kb: float
    output_size_kb: float
    accuracy_impact: float = 1.0  # Relative to baseline


class DeploymentSimulator:
    """Simulate ECG diagnosis system deployment scenarios."""
    
    def __init__(self, config: SimulationConfig = None):
        """Initialize simulator with configuration."""
        self.config = config or SimulationConfig()
        self.devices = self._create_device_profiles()
        self.tasks = self._create_task_profiles()
        self.results = {}
        
    def _create_device_profiles(self) -> Dict[str, DeviceSpecs]:
        """Create device specification profiles."""
        return {
            'sensor': DeviceSpecs(
                name='ECG Sensor',
                cpu_cores=1,
                memory_mb=64,
                storage_gb=1,
                power_watts_idle=0.5,
                power_watts_busy=2.0,
                cost_per_hour=0.0,  # Owned device
                bandwidth_mbps=1.0,
                latency_to_next_tier_ms=5.0
            ),
            'edge_gateway': DeviceSpecs(
                name='Edge Gateway',
                cpu_cores=2,
                memory_mb=2048,
                storage_gb=32,
                power_watts_idle=3.0,
                power_watts_busy=15.0,
                cost_per_hour=0.02,
                bandwidth_mbps=10.0,
                latency_to_next_tier_ms=20.0
            ),
            'fog_node': DeviceSpecs(
                name='Fog Node',
                cpu_cores=8,
                memory_mb=8192,
                storage_gb=500,
                power_watts_idle=50.0,
                power_watts_busy=200.0,
                cost_per_hour=0.15,
                bandwidth_mbps=100.0,
                latency_to_next_tier_ms=50.0
            ),
            'cloud_server': DeviceSpecs(
                name='Cloud Server',
                cpu_cores=32,
                memory_mb=32768,
                storage_gb=1000,
                power_watts_idle=100.0,
                power_watts_busy=500.0,
                cost_per_hour=0.50,
                bandwidth_mbps=1000.0,
                latency_to_next_tier_ms=0.0
            )
        }
    
    def _create_task_profiles(self) -> Dict[str, TaskProfile]:
        """Create ECG processing task profiles based on benchmark results."""
        # Load benchmark results if available
        try:
            with open('benchmark_results.json', 'r') as f:
                benchmark_data = json.load(f)
                results = benchmark_data.get('results', [])
                
                # Extract latencies
                pytorch_latency = 136.9  # Default
                onnx_latency = 137.4     # Default
                fuzzy_latency = 4.9      # Default
                
                for result in results:
                    config = result['configuration']
                    if 'PyTorch Float32' in config and not result.get('has_fuzzy', False):
                        pytorch_latency = result['mean_latency_ms']
                    elif 'ONNX Float32' in config and not result.get('has_fuzzy', False):
                        onnx_latency = result['mean_latency_ms']
                    elif 'Fuzzy' in config:
                        fuzzy_latency = result['mean_latency_ms'] - onnx_latency
                        
        except FileNotFoundError:
            pytorch_latency = 136.9
            onnx_latency = 137.4
            fuzzy_latency = 4.9
        
        return {
            'acquisition': TaskProfile(
                name='ECG Data Acquisition',
                cpu_time_ms=5.0,
                memory_mb=32,
                input_size_kb=0.0,  # Sensor input
                output_size_kb=21.0,  # 30s @ 360Hz
                accuracy_impact=1.0
            ),
            'preprocessing': TaskProfile(
                name='Signal Preprocessing',
                cpu_time_ms=20.0,
                memory_mb=256,
                input_size_kb=21.0,
                output_size_kb=14.0,  # Spectrograms
                accuracy_impact=1.0
            ),
            'feature_extraction_pytorch': TaskProfile(
                name='Feature Extraction (PyTorch)',
                cpu_time_ms=pytorch_latency,
                memory_mb=512,
                input_size_kb=14.0,
                output_size_kb=2.0,  # Features
                accuracy_impact=1.0
            ),
            'feature_extraction_onnx': TaskProfile(
                name='Feature Extraction (ONNX)',
                cpu_time_ms=onnx_latency,
                memory_mb=256,  # Smaller memory footprint
                input_size_kb=14.0,
                output_size_kb=2.0,
                accuracy_impact=0.999  # Slight accuracy difference
            ),
            'fuzzy_logic': TaskProfile(
                name='Fuzzy Decision Logic',
                cpu_time_ms=fuzzy_latency,
                memory_mb=64,
                input_size_kb=2.0,
                output_size_kb=0.5,  # JSON result
                accuracy_impact=1.05  # Improves diagnosis accuracy
            ),
            'result_aggregation': TaskProfile(
                name='Result Aggregation',
                cpu_time_ms=2.0,
                memory_mb=32,
                input_size_kb=0.5,
                output_size_kb=0.5,
                accuracy_impact=1.0
            )
        }
    
    def _create_deployment_scenarios(self) -> Dict[str, Dict[str, str]]:
        """Define deployment scenarios to simulate."""
        return {
            'all_edge': {
                'description': 'All processing on edge gateway',
                'acquisition': 'sensor',
                'preprocessing': 'edge_gateway',
                'feature_extraction': 'edge_gateway',
                'fuzzy_logic': 'edge_gateway',
                'result_aggregation': 'edge_gateway',
                'model_type': 'onnx'  # Use efficient ONNX model
            },
            'edge_fog_hybrid': {
                'description': 'Preprocessing at edge, inference at fog',
                'acquisition': 'sensor',
                'preprocessing': 'edge_gateway',
                'feature_extraction': 'fog_node',
                'fuzzy_logic': 'fog_node',
                'result_aggregation': 'edge_gateway',
                'model_type': 'pytorch'  # More powerful fog node can use PyTorch
            },
            'fog_only': {
                'description': 'All processing at fog node',
                'acquisition': 'sensor',
                'preprocessing': 'fog_node',
                'feature_extraction': 'fog_node',
                'fuzzy_logic': 'fog_node',
                'result_aggregation': 'fog_node',
                'model_type': 'pytorch'
            },
            'cloud_only': {
                'description': 'All processing in cloud',
                'acquisition': 'sensor',
                'preprocessing': 'cloud_server',
                'feature_extraction': 'cloud_server',
                'fuzzy_logic': 'cloud_server',
                'result_aggregation': 'cloud_server',
                'model_type': 'pytorch'
            },
            'intelligent_hybrid': {
                'description': 'Optimal task placement',
                'acquisition': 'sensor',
                'preprocessing': 'edge_gateway',  # Low latency preprocessing
                'feature_extraction': 'fog_node',   # Balanced compute/latency
                'fuzzy_logic': 'edge_gateway',      # Low latency decision
                'result_aggregation': 'edge_gateway',
                'model_type': 'onnx'
            }
        }
    
    def simulate_scenario(self, scenario_name: str, scenario_config: Dict[str, str]) -> Dict[str, Any]:
        """Simulate a single deployment scenario."""
        print(f"Simulating: {scenario_name}")
        
        # Task execution order
        task_order = ['acquisition', 'preprocessing', 'feature_extraction', 'fuzzy_logic', 'result_aggregation']
        
        # Select model type
        model_type = scenario_config.get('model_type', 'onnx')
        if model_type == 'pytorch':
            feature_task = 'feature_extraction_pytorch'
        else:
            feature_task = 'feature_extraction_onnx'
        
        # Calculate end-to-end metrics
        total_latency_ms = 0.0
        total_bandwidth_kb = 0.0
        total_cost_per_hour = 0.0
        total_power_watts = 0.0
        device_utilization = {}
        accuracy_factor = 1.0
        
        current_device = None
        data_size_kb = 0.0
        
        for task_name in task_order:
            # Map feature extraction to specific model type
            if task_name == 'feature_extraction':
                task_name = feature_task
            
            # Get task and device
            task = self.tasks[task_name]
            device_name = scenario_config[task_name.replace('_pytorch', '').replace('_onnx', '')]
            device = self.devices[device_name]
            
            # Data transfer latency (if changing devices)
            if current_device and current_device != device_name:
                transfer_latency = self.devices[current_device].latency_to_next_tier_ms
                transfer_time = (data_size_kb * 8) / (self.devices[current_device].bandwidth_mbps * 1000)  # ms
                total_latency_ms += transfer_latency + transfer_time
                total_bandwidth_kb += data_size_kb
            
            # Task execution
            # Adjust CPU time based on device capability (more cores = faster)
            cpu_scaling = min(device.cpu_cores / 2, 4.0)  # Max 4x speedup
            actual_cpu_time = task.cpu_time_ms / cpu_scaling
            
            total_latency_ms += actual_cpu_time
            accuracy_factor *= task.accuracy_impact
            
            # Resource utilization
            if device_name not in device_utilization:
                device_utilization[device_name] = {
                    'cpu_time_ms': 0,
                    'peak_memory_mb': 0,
                    'cost_per_hour': device.cost_per_hour,
                    'power_idle_watts': device.power_watts_idle,
                    'power_busy_watts': device.power_watts_busy
                }
            
            device_utilization[device_name]['cpu_time_ms'] += actual_cpu_time
            device_utilization[device_name]['peak_memory_mb'] = max(
                device_utilization[device_name]['peak_memory_mb'],
                task.memory_mb
            )
            
            # Update for next iteration
            current_device = device_name
            data_size_kb = task.output_size_kb
        
        # Calculate costs and power consumption
        total_tasks_per_hour = self.config.ecg_arrival_rate_per_minute * 60
        
        for device_name, util in device_utilization.items():
            device = self.devices[device_name]
            
            # CPU utilization percentage
            total_busy_time_ms = util['cpu_time_ms'] * total_tasks_per_hour
            cpu_utilization = min(total_busy_time_ms / (60 * 60 * 1000), 1.0)  # Cap at 100%
            
            # Power consumption (weighted by utilization)
            idle_power = device.power_watts_idle * (1 - cpu_utilization)
            busy_power = device.power_watts_busy * cpu_utilization
            total_power_watts += idle_power + busy_power
            
            # Cost (only for utilized devices)
            if util['cpu_time_ms'] > 0:
                total_cost_per_hour += device.cost_per_hour
            
            util['cpu_utilization'] = cpu_utilization
            util['power_watts'] = idle_power + busy_power
        
        # Calculate quality of service metrics
        availability = self.config.network_reliability ** len(device_utilization)
        
        # Check resource constraints
        resource_violations = []
        for device_name, util in device_utilization.items():
            device = self.devices[device_name]
            if util['peak_memory_mb'] > device.memory_mb:
                resource_violations.append(f"{device_name}: Memory exceeded ({util['peak_memory_mb']} > {device.memory_mb} MB)")
            if util['cpu_utilization'] > 0.9:
                resource_violations.append(f"{device_name}: High CPU utilization ({util['cpu_utilization']:.1%})")
        
        return {
            'scenario': scenario_name,
            'description': scenario_config['description'],
            'model_type': model_type,
            'metrics': {
                'end_to_end_latency_ms': total_latency_ms,
                'bandwidth_usage_kb_per_request': total_bandwidth_kb,
                'cost_per_hour_usd': total_cost_per_hour,
                'power_consumption_watts': total_power_watts,
                'accuracy_factor': accuracy_factor,
                'availability': availability,
                'throughput_requests_per_minute': min(60000 / total_latency_ms, self.config.ecg_arrival_rate_per_minute)
            },
            'device_utilization': device_utilization,
            'resource_violations': resource_violations,
            'feasible': len(resource_violations) == 0
        }
    
    def run_simulation(self) -> Dict[str, Any]:
        """Run complete deployment simulation."""
        print("=== ECG Deployment Simulation ===\n")
        
        scenarios = self._create_deployment_scenarios()
        results = {}
        
        for scenario_name, scenario_config in scenarios.items():
            result = self.simulate_scenario(scenario_name, scenario_config)
            results[scenario_name] = result
        
        # Analysis and comparison
        analysis = self._analyze_results(results)
        
        return {
            'simulation_config': self.config,
            'scenarios': results,
            'analysis': analysis,
            'timestamp': time.time()
        }
    
    def _analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze simulation results and provide recommendations."""
        
        # Find best scenarios for different criteria
        feasible_scenarios = {name: result for name, result in results.items() if result['feasible']}
        
        if not feasible_scenarios:
            return {'error': 'No feasible scenarios found'}
        
        # Best latency
        best_latency = min(feasible_scenarios.items(), key=lambda x: x[1]['metrics']['end_to_end_latency_ms'])
        
        # Best cost
        best_cost = min(feasible_scenarios.items(), key=lambda x: x[1]['metrics']['cost_per_hour_usd'])
        
        # Best power efficiency  
        best_power = min(feasible_scenarios.items(), key=lambda x: x[1]['metrics']['power_consumption_watts'])
        
        # Best accuracy
        best_accuracy = max(feasible_scenarios.items(), key=lambda x: x[1]['metrics']['accuracy_factor'])
        
        # Overall score (weighted combination)
        def calculate_score(result):
            metrics = result['metrics']
            # Normalize and weight metrics (lower is better except accuracy)
            latency_score = 1000 / max(metrics['end_to_end_latency_ms'], 1)  # Higher is better
            cost_score = 1 / max(metrics['cost_per_hour_usd'], 0.01)  # Higher is better
            power_score = 100 / max(metrics['power_consumption_watts'], 1)  # Higher is better
            accuracy_score = metrics['accuracy_factor']  # Higher is better
            
            # Weighted average
            return (latency_score * 0.4 + cost_score * 0.3 + power_score * 0.2 + accuracy_score * 0.1)
        
        best_overall = max(feasible_scenarios.items(), key=lambda x: calculate_score(x[1]))
        
        # Generate recommendations
        recommendations = []
        
        if best_latency[1]['metrics']['end_to_end_latency_ms'] < 100:
            recommendations.append(f"For ultra-low latency (< 100ms): Use '{best_latency[0]}' scenario")
        
        if best_cost[1]['metrics']['cost_per_hour_usd'] < 0.10:
            recommendations.append(f"For cost optimization: Use '{best_cost[0]}' scenario")
        
        if best_power[1]['metrics']['power_consumption_watts'] < 50:
            recommendations.append(f"For power efficiency: Use '{best_power[0]}' scenario")
        
        recommendations.append(f"Overall best balance: '{best_overall[0]}' scenario")
        
        return {
            'best_scenarios': {
                'latency': {'name': best_latency[0], 'value': best_latency[1]['metrics']['end_to_end_latency_ms']},
                'cost': {'name': best_cost[0], 'value': best_cost[1]['metrics']['cost_per_hour_usd']},
                'power': {'name': best_power[0], 'value': best_power[1]['metrics']['power_consumption_watts']},
                'accuracy': {'name': best_accuracy[0], 'value': best_accuracy[1]['metrics']['accuracy_factor']},
                'overall': {'name': best_overall[0], 'score': calculate_score(best_overall[1])}
            },
            'recommendations': recommendations,
            'feasible_scenarios': list(feasible_scenarios.keys()),
            'infeasible_scenarios': [name for name in results.keys() if not results[name]['feasible']]
        }
    
    def export_results(self, results: Dict[str, Any], output_dir: str = "simulation_results"):
        """Export simulation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Full results JSON
        with open(output_path / "simulation_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Summary CSV
        with open(output_path / "scenario_comparison.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Scenario', 'Description', 'Model Type', 'Latency (ms)', 
                'Cost ($/hour)', 'Power (W)', 'Accuracy Factor', 
                'Throughput (req/min)', 'Feasible'
            ])
            
            for scenario_name, scenario_result in results['scenarios'].items():
                metrics = scenario_result['metrics']
                writer.writerow([
                    scenario_name,
                    scenario_result['description'],
                    scenario_result['model_type'],
                    f"{metrics['end_to_end_latency_ms']:.1f}",
                    f"{metrics['cost_per_hour_usd']:.3f}",
                    f"{metrics['power_consumption_watts']:.1f}",
                    f"{metrics['accuracy_factor']:.3f}",
                    f"{metrics['throughput_requests_per_minute']:.1f}",
                    scenario_result['feasible']
                ])
        
        # Device utilization CSV
        with open(output_path / "device_utilization.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario', 'Device', 'CPU Utilization (%)', 'Memory (MB)', 'Power (W)', 'Cost ($/hour)'])
            
            for scenario_name, scenario_result in results['scenarios'].items():
                for device_name, utilization in scenario_result['device_utilization'].items():
                    writer.writerow([
                        scenario_name,
                        device_name,
                        f"{utilization['cpu_utilization']*100:.1f}",
                        utilization['peak_memory_mb'],
                        f"{utilization['power_watts']:.1f}",
                        f"{utilization['cost_per_hour']:.3f}"
                    ])
        
        print(f"✓ Simulation results exported to: {output_path}")


def create_visualization_script(results: Dict[str, Any], output_dir: str):
    """Create a Python script to visualize results."""
    viz_script = f'''#!/usr/bin/env python3
"""
Visualization script for ECG deployment simulation results.

Generates charts comparing different deployment scenarios.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results():
    with open(Path(__file__).parent / "simulation_results.json", 'r') as f:
        return json.load(f)

def create_comparison_charts():
    results = load_results()
    scenarios = results['scenarios']
    
    # Extract data for plotting
    names = []
    latencies = []
    costs = []
    power = []
    accuracy = []
    
    for name, result in scenarios.items():
        if result['feasible']:
            names.append(name.replace('_', '\\n'))
            metrics = result['metrics']
            latencies.append(metrics['end_to_end_latency_ms'])
            costs.append(metrics['cost_per_hour_usd'])
            power.append(metrics['power_consumption_watts'])
            accuracy.append(metrics['accuracy_factor'])
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Latency comparison
    bars1 = ax1.bar(names, latencies, color='skyblue')
    ax1.set_title('End-to-End Latency')
    ax1.set_ylabel('Latency (ms)')
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, latencies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{{val:.1f}}', ha='center', va='bottom')
    
    # Cost comparison  
    bars2 = ax2.bar(names, costs, color='lightgreen')
    ax2.set_title('Operating Cost')
    ax2.set_ylabel('Cost ($/hour)')
    ax2.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, costs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'${{val:.3f}}', ha='center', va='bottom')
    
    # Power consumption
    bars3 = ax3.bar(names, power, color='orange')
    ax3.set_title('Power Consumption')
    ax3.set_ylabel('Power (Watts)')
    ax3.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, power):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{{val:.1f}}W', ha='center', va='bottom')
    
    # Accuracy factor
    bars4 = ax4.bar(names, accuracy, color='lightcoral')
    ax4.set_title('Accuracy Factor')
    ax4.set_ylabel('Relative Accuracy')
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline')
    for bar, val in zip(bars4, accuracy):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{{val:.3f}}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('scenario_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create radar chart for overall comparison
    create_radar_chart(names, latencies, costs, power, accuracy)

def create_radar_chart(names, latencies, costs, power, accuracy):
    # Normalize metrics (0-1 scale, higher is better)
    norm_latencies = [1 - (l - min(latencies)) / (max(latencies) - min(latencies)) for l in latencies]
    norm_costs = [1 - (c - min(costs)) / (max(costs) - min(costs)) for c in costs]  
    norm_power = [1 - (p - min(power)) / (max(power) - min(power)) for p in power]
    norm_accuracy = [(a - min(accuracy)) / (max(accuracy) - min(accuracy)) for a in accuracy]
    
    categories = ['Low Latency', 'Low Cost', 'Low Power', 'High Accuracy']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, name in enumerate(names):
        values = [norm_latencies[i], norm_costs[i], norm_power[i], norm_accuracy[i]]
        values.append(values[0])  # Close the polygon
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles.append(angles[0])
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Deployment Scenario Comparison\\n(Normalized Metrics)', size=16, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.savefig('radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    try:
        create_comparison_charts()
        print("✓ Visualization charts created successfully!")
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error creating charts: {{e}}")
'''
    
    with open(Path(output_dir) / "create_charts.py", 'w') as f:
        f.write(viz_script)
    
    print(f"  ✓ Visualization script: {output_dir}/create_charts.py")


def main():
    """Main simulation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ECG deployment simulation")
    parser.add_argument('--duration', type=int, default=60, help='Simulation duration (minutes)')
    parser.add_argument('--arrival-rate', type=int, default=6, help='ECG arrivals per minute')
    parser.add_argument('--output', default='simulation_results', help='Output directory')
    parser.add_argument('--visualization', action='store_true', help='Create visualization script')
    
    args = parser.parse_args()
    
    # Create simulation configuration
    config = SimulationConfig(
        simulation_duration_minutes=args.duration,
        ecg_arrival_rate_per_minute=args.arrival_rate,
        network_reliability=0.99,
        enable_failover=True,
        cost_analysis=True
    )
    
    # Run simulation
    simulator = DeploymentSimulator(config)
    results = simulator.run_simulation()
    
    # Display summary
    print("\n=== Simulation Summary ===")
    for scenario_name, scenario_result in results['scenarios'].items():
        metrics = scenario_result['metrics']
        feasible = "✓" if scenario_result['feasible'] else "✗"
        print(f"{feasible} {scenario_name:20s}: {metrics['end_to_end_latency_ms']:6.1f}ms, "
              f"${metrics['cost_per_hour_usd']:6.3f}/hr, {metrics['power_consumption_watts']:6.1f}W")
    
    # Best recommendations
    analysis = results['analysis']
    if 'best_scenarios' in analysis:
        print(f"\n=== Recommendations ===")
        for rec in analysis['recommendations']:
            print(f"• {rec}")
    
    # Export results
    simulator.export_results(results, args.output)
    
    # Create visualization script if requested
    if args.visualization:
        create_visualization_script(results, args.output)
    
    print(f"\n✅ Deployment simulation completed!")
    print(f"📊 Results: {args.output}/")
    if args.visualization:
        print(f"📈 Run visualization: python {args.output}/create_charts.py")


if __name__ == "__main__":
    main()
