"""
Docker Scaling Performance Test

This script simulates testing your API with different numbers of Docker containers
by varying the load and measuring performance characteristics.

Since Render.com manages containers automatically, we'll simulate different
scaling scenarios by varying the request patterns and loads.
"""

import requests
import time
import json
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from monitor_performance import PerformanceMonitor

class DockerScalingSimulator:
    def __init__(self, api_url):
        self.api_url = api_url
        self.monitor = PerformanceMonitor(api_url)
        self.scaling_results = []
    
    def simulate_container_scaling(self):
        """
        Simulate different container scaling scenarios.
        Each scenario represents what would happen with different numbers of containers.
        """
        
        # Define scaling scenarios that simulate different container counts
        scaling_scenarios = [
            {
                'name': '1_container_simulation',
                'description': 'Light load - simulates 1 container',
                'concurrent_users': 5,
                'total_requests': 50,
                'expected_containers': 1
            },
            {
                'name': '2_container_simulation', 
                'description': 'Medium load - simulates 2 containers',
                'concurrent_users': 15,
                'total_requests': 100,
                'expected_containers': 2
            },
            {
                'name': '3_container_simulation',
                'description': 'High load - simulates 3 containers', 
                'concurrent_users': 30,
                'total_requests': 150,
                'expected_containers': 3
            },
            {
                'name': '4_container_simulation',
                'description': 'Very high load - simulates 4 containers',
                'concurrent_users': 50,
                'total_requests': 200,
                'expected_containers': 4
            },
            {
                'name': '5_container_simulation',
                'description': 'Peak load - simulates 5+ containers',
                'concurrent_users': 75,
                'total_requests': 250,
                'expected_containers': 5
            }
        ]
        
        results = []
        
        for scenario in scaling_scenarios:
            print(f"\n{'='*80}")
            print(f"🐳 Running {scenario['name']}")
            print(f"📋 {scenario['description']}")
            print(f"👥 {scenario['concurrent_users']} concurrent users")
            print(f"📊 {scenario['total_requests']} total requests")
            print(f"{'='*80}")
            
            # Run the test
            stats, detailed_results = self.monitor.load_test(
                num_requests=scenario['total_requests'],
                concurrent_users=scenario['concurrent_users'],
                method='file'
            )
            
            # Add scenario info to stats
            stats.update({
                'scenario_name': scenario['name'],
                'description': scenario['description'], 
                'expected_containers': scenario['expected_containers']
            })
            
            results.append(stats)
            self.monitor.print_stats(stats)
            
            # Wait between tests to allow system to stabilize
            print(f"\n⏳ Waiting 15 seconds for system stabilization...")
            time.sleep(15)
        
        return results
    
    def simulate_burst_traffic(self):
        """Simulate sudden burst traffic that would trigger auto-scaling"""
        print(f"\n{'='*80}")
        print(f"💥 BURST TRAFFIC SIMULATION")
        print(f"Simulating sudden traffic spike that triggers auto-scaling")
        print(f"{'='*80}")
        
        # Gradual ramp-up to simulate auto-scaling trigger
        burst_phases = [
            {'users': 10, 'requests': 30, 'phase': 'Warm-up'},
            {'users': 25, 'requests': 50, 'phase': 'Ramp-up'}, 
            {'users': 50, 'requests': 100, 'phase': 'Peak traffic'},
            {'users': 75, 'requests': 150, 'phase': 'Sustained high load'},
            {'users': 25, 'requests': 50, 'phase': 'Cool-down'}
        ]
        
        burst_results = []
        
        for phase in burst_phases:
            print(f"\n🔥 {phase['phase']}: {phase['users']} users, {phase['requests']} requests")
            
            start_time = time.time()
            stats, _ = self.monitor.load_test(
                num_requests=phase['requests'],
                concurrent_users=phase['users'],
                method='file'
            )
            
            stats['burst_phase'] = phase['phase']
            stats['phase_order'] = len(burst_results) + 1
            burst_results.append(stats)
            
            print(f"✅ {phase['phase']} complete - {stats['success_rate']:.1f}% success rate")
            
            # Short pause between phases
            time.sleep(5)
        
        return burst_results
    
    def create_scaling_visualization(self, scaling_results, burst_results):
        """Create visualizations showing scaling performance"""
        
        # Create scaling comparison chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Docker Container Scaling Performance Analysis', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        containers = [r['expected_containers'] for r in scaling_results]
        response_times = [r.get('avg_response_time_ms', 0) for r in scaling_results]
        success_rates = [r['success_rate'] for r in scaling_results]
        throughput = [r['requests_per_second'] for r in scaling_results]
        
        # Plot 1: Response Time vs Container Count
        ax1.plot(containers, response_times, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Simulated Container Count')
        ax1.set_ylabel('Average Response Time (ms)')
        ax1.set_title('Response Time vs Container Count')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success Rate vs Container Count  
        ax2.plot(containers, success_rates, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Simulated Container Count')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate vs Container Count')
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Throughput vs Container Count
        ax3.plot(containers, throughput, 'ro-', linewidth=2, markersize=8) 
        ax3.set_xlabel('Simulated Container Count')
        ax3.set_ylabel('Requests per Second')
        ax3.set_title('Throughput vs Container Count')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Burst Traffic Response Times
        if burst_results:
            phases = [r['burst_phase'] for r in burst_results]
            burst_response_times = [r.get('avg_response_time_ms', 0) for r in burst_results]
            
            ax4.bar(range(len(phases)), burst_response_times, color=['lightblue', 'yellow', 'red', 'orange', 'lightgreen'])
            ax4.set_xlabel('Traffic Phase')
            ax4.set_ylabel('Average Response Time (ms)')
            ax4.set_title('Burst Traffic Response Times')
            ax4.set_xticks(range(len(phases)))
            ax4.set_xticklabels([p.split()[0] for p in phases], rotation=45)
            ax4.grid(True, alpha=0.3)