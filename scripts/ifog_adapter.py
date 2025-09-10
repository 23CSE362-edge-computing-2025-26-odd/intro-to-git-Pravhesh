#!/usr/bin/env python3
"""
iFogSim Adapter for ECG Diagnosis System

This script creates deployment profiles and configuration files that can be used
with iFogSim to simulate fog computing deployment scenarios for the ECG diagnosis system.

It exports our measured performance metrics into formats suitable for iFogSim simulation,
including network topology definitions, application models, and device specifications.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import csv
from dataclasses import dataclass
import time

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))


@dataclass
class DeviceProfile:
    """Device specification for iFogSim."""
    name: str
    mips: int  # Million instructions per second
    ram: int   # MB
    uplink_bandwidth: float  # Mbps
    downlink_bandwidth: float  # Mbps
    rate_per_mips: float  # Cost per MIPS
    busy_power: float  # Watts when busy
    idle_power: float  # Watts when idle


@dataclass
class ApplicationModule:
    """Application module for iFogSim."""
    name: str
    ram: int  # MB required
    mips: int  # MIPS required for processing
    instances: int = 1


@dataclass
class NetworkLink:
    """Network link specification."""
    source: str
    destination: str
    latency: float  # ms
    bandwidth: float  # Mbps


class iFogSimAdapter:
    """Adapter to generate iFogSim configuration from our ECG system metrics."""
    
    def __init__(self, benchmark_results_file: str = "benchmark_results.json"):
        """Initialize adapter with benchmark results."""
        self.benchmark_file = benchmark_results_file
        self.results = self._load_benchmark_results()
        
        # ECG signal characteristics
        self.signal_size_kb = 30 * 360 * 2 / 1024  # 30s @ 360Hz, 2 bytes per sample ≈ 21KB
        self.spectrogram_size_kb = 64 * 57 * 4 / 1024  # 64x57 float32 ≈ 14KB per window
        self.features_size_kb = 512 * 4 / 1024  # 512 float32 features ≈ 2KB
        self.diagnosis_size_kb = 0.5  # JSON result ≈ 0.5KB
    
    def _load_benchmark_results(self) -> Dict[str, Any]:
        """Load benchmark results from JSON file."""
        try:
            with open(self.benchmark_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {self.benchmark_file} not found. Using default values.")
            return {
                'results': [
                    {'configuration': 'PyTorch Float32', 'mean_latency_ms': 136.9},
                    {'configuration': 'ONNX Float32', 'mean_latency_ms': 137.4},
                ]
            }
    
    def get_processing_latencies(self) -> Dict[str, float]:
        """Extract processing latencies for different configurations."""
        latencies = {}
        for result in self.results.get('results', []):
            config = result['configuration']
            latency = result.get('mean_latency_ms', 0)
            latencies[config] = latency
        
        return {
            'preprocessing_ms': 20.0,  # Estimated from windowing + spectrogram
            'pytorch_inference_ms': latencies.get('PyTorch Float32', 136.9),
            'onnx_inference_ms': latencies.get('ONNX Float32', 137.4),
            'fuzzy_logic_ms': 4.9,  # Overhead from benchmark
            'total_pytorch_ms': latencies.get('PyTorch Float32 + Fuzzy', 141.8),
            'total_onnx_ms': latencies.get('ONNX Float32 + Fuzzy', 146.5),
        }
    
    def create_device_profiles(self) -> List[DeviceProfile]:
        """Create device profiles for different deployment tiers."""
        return [
            # Edge devices (IoT sensors/gateways)
            DeviceProfile(
                name="ECG_Sensor",
                mips=100,  # Low-power sensor
                ram=64,
                uplink_bandwidth=1.0,  # Limited cellular/WiFi
                downlink_bandwidth=1.0,
                rate_per_mips=0.0,  # No compute cost for sensor
                busy_power=2.0,
                idle_power=0.5
            ),
            DeviceProfile(
                name="Edge_Gateway", 
                mips=2000,  # Raspberry Pi class device
                ram=2048,
                uplink_bandwidth=10.0,
                downlink_bandwidth=10.0,
                rate_per_mips=0.01,
                busy_power=15.0,
                idle_power=3.0
            ),
            
            # Fog nodes (local servers)
            DeviceProfile(
                name="Fog_Node",
                mips=10000,  # Local server
                ram=8192,
                uplink_bandwidth=100.0,
                downlink_bandwidth=100.0,
                rate_per_mips=0.02,
                busy_power=200.0,
                idle_power=50.0
            ),
            
            # Cloud datacenter
            DeviceProfile(
                name="Cloud_Server",
                mips=50000,  # Powerful cloud instance
                ram=32768,
                uplink_bandwidth=1000.0,
                downlink_bandwidth=1000.0,
                rate_per_mips=0.05,
                busy_power=500.0,
                idle_power=100.0
            )
        ]
    
    def create_application_modules(self) -> List[ApplicationModule]:
        """Create application modules based on our pipeline stages."""
        latencies = self.get_processing_latencies()
        
        # Convert latency to MIPS requirement (rough estimate)
        # Assume 1ms processing = 1000 MIPS requirement
        mips_factor = 1000
        
        return [
            ApplicationModule(
                name="ECG_Acquisition",
                ram=32,  # Minimal for data collection
                mips=int(5 * mips_factor),  # 5ms for data collection
                instances=1
            ),
            ApplicationModule(
                name="Preprocessing", 
                ram=256,  # Buffer for windowing + spectrogram
                mips=int(latencies['preprocessing_ms'] * mips_factor),
                instances=1
            ),
            ApplicationModule(
                name="Feature_Extraction",
                ram=512,  # Model weights + intermediate tensors
                mips=int(latencies['onnx_inference_ms'] * mips_factor),
                instances=1
            ),
            ApplicationModule(
                name="Fuzzy_Diagnosis",
                ram=64,  # Small memory for fuzzy rules
                mips=int(latencies['fuzzy_logic_ms'] * mips_factor),
                instances=1
            ),
            ApplicationModule(
                name="Result_Aggregation",
                ram=32,  # Minimal for result formatting
                mips=int(2 * mips_factor),  # 2ms for JSON formatting
                instances=1
            )
        ]
    
    def create_network_topology(self) -> List[NetworkLink]:
        """Create network topology with realistic latencies."""
        return [
            # Sensor to Edge Gateway
            NetworkLink("ECG_Sensor", "Edge_Gateway", 5.0, 1.0),
            
            # Edge Gateway to Fog Node  
            NetworkLink("Edge_Gateway", "Fog_Node", 20.0, 10.0),
            
            # Fog Node to Cloud
            NetworkLink("Fog_Node", "Cloud_Server", 50.0, 100.0),
            
            # Direct connections (bypassing intermediate tiers)
            NetworkLink("ECG_Sensor", "Fog_Node", 25.0, 1.0),
            NetworkLink("ECG_Sensor", "Cloud_Server", 100.0, 1.0),
            NetworkLink("Edge_Gateway", "Cloud_Server", 70.0, 10.0),
        ]
    
    def create_deployment_scenarios(self) -> List[Dict[str, Any]]:
        """Create different deployment scenarios to compare."""
        return [
            {
                "name": "All_Edge",
                "description": "Process everything on edge gateway",
                "placement": {
                    "ECG_Acquisition": "ECG_Sensor",
                    "Preprocessing": "Edge_Gateway", 
                    "Feature_Extraction": "Edge_Gateway",
                    "Fuzzy_Diagnosis": "Edge_Gateway",
                    "Result_Aggregation": "Edge_Gateway"
                }
            },
            {
                "name": "Edge_Fog_Hybrid",
                "description": "Preprocessing at edge, inference at fog",
                "placement": {
                    "ECG_Acquisition": "ECG_Sensor",
                    "Preprocessing": "Edge_Gateway",
                    "Feature_Extraction": "Fog_Node", 
                    "Fuzzy_Diagnosis": "Fog_Node",
                    "Result_Aggregation": "Edge_Gateway"
                }
            },
            {
                "name": "Fog_Only",
                "description": "All processing at fog node",
                "placement": {
                    "ECG_Acquisition": "ECG_Sensor",
                    "Preprocessing": "Fog_Node",
                    "Feature_Extraction": "Fog_Node",
                    "Fuzzy_Diagnosis": "Fog_Node", 
                    "Result_Aggregation": "Fog_Node"
                }
            },
            {
                "name": "Cloud_Only",
                "description": "All processing in cloud",
                "placement": {
                    "ECG_Acquisition": "ECG_Sensor",
                    "Preprocessing": "Cloud_Server",
                    "Feature_Extraction": "Cloud_Server",
                    "Fuzzy_Diagnosis": "Cloud_Server",
                    "Result_Aggregation": "Cloud_Server"
                }
            }
        ]
    
    def export_ifog_config(self, output_dir: str = "ifog_simulation"):
        """Export complete iFogSim configuration files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Device profiles
        devices = self.create_device_profiles()
        with open(output_path / "device_profiles.json", 'w') as f:
            json.dump([{
                'name': d.name,
                'mips': d.mips,
                'ram': d.ram, 
                'uplink_bandwidth': d.uplink_bandwidth,
                'downlink_bandwidth': d.downlink_bandwidth,
                'rate_per_mips': d.rate_per_mips,
                'busy_power': d.busy_power,
                'idle_power': d.idle_power
            } for d in devices], f, indent=2)
        
        # Application modules
        modules = self.create_application_modules()
        with open(output_path / "application_modules.json", 'w') as f:
            json.dump([{
                'name': m.name,
                'ram': m.ram,
                'mips': m.mips,
                'instances': m.instances
            } for m in modules], f, indent=2)
        
        # Network topology
        topology = self.create_network_topology()
        with open(output_path / "network_topology.json", 'w') as f:
            json.dump([{
                'source': link.source,
                'destination': link.destination, 
                'latency': link.latency,
                'bandwidth': link.bandwidth
            } for link in topology], f, indent=2)
        
        # Deployment scenarios
        scenarios = self.create_deployment_scenarios()
        with open(output_path / "deployment_scenarios.json", 'w') as f:
            json.dump(scenarios, f, indent=2)
        
        # Workload profile
        latencies = self.get_processing_latencies()
        workload = {
            "arrival_rate_per_minute": 6,  # 10-second intervals
            "signal_characteristics": {
                "raw_signal_size_kb": self.signal_size_kb,
                "spectrogram_size_kb": self.spectrogram_size_kb,
                "features_size_kb": self.features_size_kb,
                "diagnosis_size_kb": self.diagnosis_size_kb
            },
            "processing_requirements": latencies,
            "simulation_duration_minutes": 60
        }
        
        with open(output_path / "workload_profile.json", 'w') as f:
            json.dump(workload, f, indent=2)
        
        # iFogSim Java configuration template
        self._create_ifog_java_template(output_path)
        
        print(f"✓ iFogSim configuration exported to: {output_path}")
        return output_path
    
    def _create_ifog_java_template(self, output_path: Path):
        """Create a Java template for iFogSim integration."""
        java_template = '''// iFogSim Java Configuration Template for ECG Diagnosis System
// This template shows how to integrate the exported JSON configurations

import org.fog.entities.*;
import org.fog.placement.ModuleMapping;
import org.fog.placement.ModulePlacementEdgewards;
import org.fog.utils.FogLinearPowerModel;

public class ECGDiagnosisSimulation {
    
    public static void main(String[] args) {
        // Load device profiles from device_profiles.json
        // Load application modules from application_modules.json  
        // Load network topology from network_topology.json
        // Load deployment scenarios from deployment_scenarios.json
        // Load workload profile from workload_profile.json
        
        // Create fog devices based on profiles
        FogDevice sensor = createFogDevice("ECG_Sensor", 100, 64, 10000, 1.0, 1.0, 2.0, 0.5);
        FogDevice gateway = createFogDevice("Edge_Gateway", 2000, 2048, 10000, 10.0, 10.0, 15.0, 3.0);
        FogDevice fogNode = createFogDevice("Fog_Node", 10000, 8192, 10000, 100.0, 100.0, 200.0, 50.0);
        FogDevice cloud = createFogDevice("Cloud_Server", 50000, 32768, 10000, 1000.0, 1000.0, 500.0, 100.0);
        
        // Create application modules
        Application application = createApplication("ECG_DIAGNOSIS", userId);
        application.addAppModule("ECG_Acquisition", 32, 5000);
        application.addAppModule("Preprocessing", 256, 20000);
        application.addAppModule("Feature_Extraction", 512, 137400);
        application.addAppModule("Fuzzy_Diagnosis", 64, 4900);
        application.addAppModule("Result_Aggregation", 32, 2000);
        
        // Define application edges (data flow)
        application.addAppEdge("ECG_SENSOR", "ECG_Acquisition", 21, 5, "ECG_DATA");
        application.addAppEdge("ECG_Acquisition", "Preprocessing", 21, 20, "RAW_SIGNAL");
        application.addAppEdge("Preprocessing", "Feature_Extraction", 14, 137, "SPECTROGRAM");
        application.addAppEdge("Feature_Extraction", "Fuzzy_Diagnosis", 2, 5, "FEATURES");
        application.addAppEdge("Fuzzy_Diagnosis", "Result_Aggregation", 0.5, 2, "DIAGNOSIS");
        
        // Create module mapping for different scenarios
        ModuleMapping moduleMapping = createModuleMapping(scenario);
        
        // Run simulation
        FogBroker broker = new FogBroker("ECG_Broker");
        broker.submitApplication(application, new ModulePlacementEdgewards(fogDevices, sensors, actuators, application, moduleMapping));
        
        TimeKeeper.getInstance().setSimulationStartTime(Calendar.getInstance().getTimeInMillis());
        CloudSim.startSimulation();
    }
    
    // Helper methods for device and application creation...
}'''
        
        with open(output_path / "ECGDiagnosisSimulation.java", 'w') as f:
            f.write(java_template)
    
    def export_csv_profiles(self, output_dir: str = "ifog_simulation"):
        """Export CSV files for easy import into other simulation tools."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Device profiles CSV
        devices = self.create_device_profiles()
        with open(output_path / "devices.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Device', 'MIPS', 'RAM_MB', 'Uplink_Mbps', 'Downlink_Mbps', 'Cost_Per_MIPS', 'Busy_Power_W', 'Idle_Power_W'])
            for d in devices:
                writer.writerow([d.name, d.mips, d.ram, d.uplink_bandwidth, d.downlink_bandwidth, d.rate_per_mips, d.busy_power, d.idle_power])
        
        # Latency matrix CSV
        topology = self.create_network_topology()
        with open(output_path / "latency_matrix.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Source', 'Destination', 'Latency_ms', 'Bandwidth_Mbps'])
            for link in topology:
                writer.writerow([link.source, link.destination, link.latency, link.bandwidth])
        
        # Processing requirements CSV
        latencies = self.get_processing_latencies()
        with open(output_path / "processing_requirements.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Stage', 'Latency_ms', 'MIPS_Requirement'])
            for stage, latency in latencies.items():
                mips_req = int(latency * 1000)  # Convert ms to MIPS
                writer.writerow([stage, latency, mips_req])
        
        print(f"✓ CSV profiles exported to: {output_path}")


def main():
    """Main function to run iFogSim adapter."""
    print("=== iFogSim Adapter for ECG Diagnosis System ===\n")
    
    adapter = iFogSimAdapter()
    
    # Export iFogSim configurations
    adapter.export_ifog_config()
    
    # Export CSV profiles for other tools
    adapter.export_csv_profiles()
    
    print("\niFogSim adapter completed successfully!")
    print("Files generated:")
    print("- ifog_simulation/device_profiles.json")
    print("- ifog_simulation/application_modules.json") 
    print("- ifog_simulation/network_topology.json")
    print("- ifog_simulation/deployment_scenarios.json")
    print("- ifog_simulation/workload_profile.json")
    print("- ifog_simulation/ECGDiagnosisSimulation.java")
    print("- ifog_simulation/devices.csv")
    print("- ifog_simulation/latency_matrix.csv")
    print("- ifog_simulation/processing_requirements.csv")


if __name__ == "__main__":
    main()
