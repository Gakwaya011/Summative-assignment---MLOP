import requests
import time
import statistics
import threading
import json
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io
import numpy as np

class PerformanceMonitor:
    def __init__(self, base_url):
        self.base_url = base_url
        self.results = []
        self.test_image = self.create_test_image()
    
    def create_test_image(self):
        """Create a test image for performance testing"""
        # Create a simple test image
        img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return {
            'data': img_bytes.getvalue(),
            'base64': base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        }
    
    def single_request(self, request_id, method='file'):
        """Make a single request and measure performance"""
        start_time = time.time()
        
        try:
            if method == 'file':
                files = {'file': ('test.jpg', self.test_image['data'], 'image/jpeg')}
                response = requests.post(f"{self.base_url}/predict", files=files, timeout=30)
            else:  # JSON method
                payload = {'image': self.test_image['base64']}
                response = requests.post(f"{self.base_url}/predict", json=payload, timeout=30)
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            result = {
                'request_id': request_id,
                'method': method,
                'timestamp': datetime.now().isoformat(),
                'response_time_ms': response_time,
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'response_size': len(response.content) if response.content else 0
            }
            
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    result['predicted_class'] = json_response.get('predicted_class', 'unknown')
                    result['confidence'] = json_response.get('confidence', 0.0)
                except:
                    result['json_error'] = True
            else:
                result['error'] = response.text[:200]  # First 200 chars of error
            
            return result
            
        except requests.exceptions.Timeout:
            return {
                'request_id': request_id,
                'method': method,
                'timestamp': datetime.now().isoformat(),
                'response_time_ms': 30000,  # Timeout
                'status_code': 0,
                'success': False,
                'error': 'Timeout'
            }
        except Exception as e:
            return {
                'request_id': request_id,
                'method': method,
                'timestamp': datetime.now().isoformat(),
                'response_time_ms': 0,
                'status_code': 0,
                'success': False,
                'error': str(e)[:200]
            }
    
    def load_test(self, num_requests=100, concurrent_users=10, method='file'):
        """Run a load test with specified parameters"""
        print(f"Starting load test: {num_requests} requests, {concurrent_users} concurrent users, method: {method}")
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            # Submit all requests
            futures = [
                executor.submit(self.single_request, i, method) 
                for i in range(num_requests)
            ]
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Print progress every 10 requests
                    if len(results) % 10 == 0:
                        success_rate = sum(1 for r in results if r['success']) / len(results) * 100
                        avg_response_time = statistics.mean([r['response_time_ms'] for r in results if r['response_time_ms'] > 0])
                        print(f"Progress: {len(results)}/{num_requests} - Success: {success_rate:.1f}% - Avg Response: {avg_response_time:.1f}ms")
                
                except Exception as e:
                    print(f"Error collecting result: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        if successful_requests:
            response_times = [r['response_time_ms'] for r in successful_requests]
            stats = {
                'total_requests': num_requests,
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate': len(successful_requests) / num_requests * 100,
                'total_time_seconds': total_time,
                'requests_per_second': num_requests / total_time,
                'avg_response_time_ms': statistics.mean(response_times),
                'median_response_time_ms': statistics.median(response_times),
                'min_response_time_ms': min(response_times),
                'max_response_time_ms': max(response_times),
                'p95_response_time_ms': np.percentile(response_times, 95),
                'p99_response_time_ms': np.percentile(response_times, 99),
                'concurrent_users': concurrent_users,
                'method': method
            }
        else:
            stats = {
                'total_requests': num_requests,
                'successful_requests': 0,
                'failed_requests': len(failed_requests),
                'success_rate': 0,
                'total_time_seconds': total_time,
                'requests_per_second': 0,
                'concurrent_users': concurrent_users,
                'method': method,
                'error': 'No successful requests'
            }
        
        return stats, results
    
    def run_comprehensive_test(self):
        """Run a comprehensive performance test with different configurations"""
        test_configurations = [
            {'users': 1, 'requests': 20, 'method': 'file'},
            {'users': 5, 'requests': 50, 'method': 'file'},
            {'users': 10, 'requests': 100, 'method': 'file'},
            {'users': 20, 'requests': 100, 'method': 'file'},
            {'users': 5, 'requests': 50, 'method': 'json'},
            {'users': 10, 'requests': 100, 'method': 'json'},
        ]
        
        all_results = []
        
        for config in test_configurations:
            print(f"\n{'='*60}")
            print(f"Running test: {config['users']} users, {config['requests']} requests, {config['method']} method")
            print(f"{'='*60}")
            
            stats, detailed_results = self.load_test(
                num_requests=config['requests'],
                concurrent_users=config['users'],
                method=config['method']
            )
            
            stats['test_name'] = f"{config['users']}users_{config['requests']}req_{config['method']}"
            all_results.append(stats)
            
            self.print_stats(stats)
            
            # Wait between tests
            print("Waiting 10 seconds before next test...")
            time.sleep(10)
        
        return all_results
    
    def print_stats(self, stats):
        """Print formatted statistics"""
        print(f"\n📊 Performance Results:")
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Successful: {stats['successful_requests']} ({stats['success_rate']:.1f}%)")
        print(f"   Failed: {stats['failed_requests']}")
        print(f"   Total Time: {stats['total_time_seconds']:.2f} seconds")
        print(f"   Requests/Second: {stats['requests_per_second']:.2f}")
        
        if 'avg_response_time_ms' in stats:
            print(f"   Average Response Time: {stats['avg_response_time_ms']:.1f}ms")
            print(f"   Median Response Time: {stats['median_response_time_ms']:.1f}ms")
            print(f"   95th Percentile: {stats['p95_response_time_ms']:.1f}ms")
            print(f"   99th Percentile: {stats['p99_response_time_ms']:.1f}ms")
            print(f"   Min Response Time: {stats['min_response_time_ms']:.1f}ms")
            print(f"   Max Response Time: {stats['max_response_time_ms']:.1f}ms")
    
    def create_performance_report(self, all_results, filename='performance_report.html'):
        """Create an HTML performance report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Performance Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .test-result {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; border-radius: 3px; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 API Performance Test Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>API URL: {self.base_url}</p>
            </div>
        """
        
        for result in all_results:
            success_class = 'success' if result['success_rate'] > 95 else 'warning' if result['success_rate'] > 80 else 'error'
            
            html_content += f"""
            <div class="test-result">
                <h2>{result['test_name']}</h2>
                <div class="metrics">
                    <div class="metric">
                        <strong>Success Rate:</strong> 
                        <span class="{success_class}">{result['success_rate']:.1f}%</span>
                    </div>
                    <div class="metric">
                        <strong>Requests/Second:</strong> {result['requests_per_second']:.2f}
                    </div>
                    <div class="metric">
                        <strong>Average Response:</strong> {result.get('avg_response_time_ms', 'N/A')} ms
                    </div>
                    <div class="metric">
                        <strong>95th Percentile:</strong> {result.get('p95_response_time_ms', 'N/A')} ms
                    </div>
                    <div class="metric">
                        <strong>Concurrent Users:</strong> {result['concurrent_users']}
                    </div>
                    <div class="metric">
                        <strong>Total Requests:</strong> {result['total_requests']}
                    </div>
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"📄 Performance report saved to: {filename}")

def main():
    # Replace with your actual API URL
    API_URL = "https://vehicle-classifier-api.onrender.com"
    
    # Test if API is accessible
    try:
        response = requests.get(f"{API_URL}/", timeout=10)
        if response.status_code != 200:
            print(f"❌ API not accessible at {API_URL}")
            return
        print(f"✅ API is accessible at {API_URL}")
    except Exception as e:
        print(f"❌ Cannot reach API: {e}")
        return
    
    # Create performance monitor
    monitor = PerformanceMonitor(API_URL)
    
    # Run comprehensive tests
    print("🚀 Starting comprehensive performance tests...")
    all_results = monitor.run_comprehensive_test()
    
    # Create report
    monitor.create_performance_report(all_results)
    
    # Save results as JSON
    with open('performance_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n✅ Performance testing complete!")
    print("📄 Results saved to: performance_report.html and performance_results.json")

if __name__ == "__main__":
    main()