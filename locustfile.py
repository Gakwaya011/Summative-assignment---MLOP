import time
import base64
import random
import os
from locust import HttpUser, task, between
from PIL import Image
import io
import numpy as np

class VehicleClassifierUser(HttpUser):
    """
    Locust user class to simulate traffic to the Vehicle Classifier API
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts"""
        self.test_images = self.create_test_images()
        print(f"User started with {len(self.test_images)} test images")
    
    def create_test_images(self):
        """Create synthetic test images for load testing"""
        test_images = []
        
        # Create different types of synthetic images
        image_types = [
            ("car", (255, 0, 0)),      # Red image for car
            ("truck", (0, 255, 0)),    # Green image for truck  
            ("bus", (0, 0, 255)),      # Blue image for bus
            ("motorcycle", (255, 255, 0)),  # Yellow image for motorcycle
            ("bicycle", (255, 0, 255))  # Magenta image for bicycle
        ]
        
        for vehicle_type, color in image_types:
            # Create a synthetic image
            img_array = np.full((128, 128, 3), color, dtype=np.uint8)
            
            # Add some random noise to make it more realistic
            noise = np.random.randint(0, 50, (128, 128, 3))
            img_array = np.clip(img_array.astype(int) + noise - 25, 0, 255).astype(np.uint8)
            
            # Convert to PIL Image
            img = Image.fromarray(img_array)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=85)
            img_bytes.seek(0)
            
            # Encode as base64
            img_b64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            
            test_images.append({
                'type': vehicle_type,
                'data': img_bytes.getvalue(),
                'base64': img_b64
            })
        
        return test_images
    
    @task(3)
    def test_health_endpoint(self):
        """Test the health check endpoint (higher frequency)"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task(5)
    def test_prediction_file_upload(self):
        """Test prediction endpoint with file upload"""
        # Select a random test image
        test_image = random.choice(self.test_images)
        
        # Prepare file upload
        files = {
            'file': ('test_image.jpg', test_image['data'], 'image/jpeg')
        }
        
        start_time = time.time()
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'predicted_class' in result and 'confidence' in result:
                        response.success()
                        print(f"✅ Prediction successful: {result['predicted_class']} ({response_time:.1f}ms)")
                    else:
                        response.failure("Missing prediction data in response")
                except Exception as e:
                    response.failure(f"Invalid JSON response: {str(e)}")
            else:
                response.failure(f"Prediction failed with status {response.status_code}: {response.text}")
    
    @task(2)
    def test_prediction_json(self):
        """Test prediction endpoint with JSON/base64 data"""
        # Select a random test image
        test_image = random.choice(self.test_images)
        
        payload = {
            'image': test_image['base64']
        }
        
        start_time = time.time()
        
        with self.client.post("/predict", 
                            json=payload, 
                            headers={'Content-Type': 'application/json'},
                            catch_response=True) as response:
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'predicted_class' in result and 'confidence' in result:
                        response.success()
                        print(f"✅ JSON prediction successful: {result['predicted_class']} ({response_time:.1f}ms)")
                    else:
                        response.failure("Missing prediction data in response")
                except Exception as e:
                    response.failure(f"Invalid JSON response: {str(e)}")
            else:
                response.failure(f"JSON prediction failed with status {response.status_code}")
    
    @task(1)  
    def test_prediction_with_invalid_data(self):
        """Test prediction endpoint with invalid data to check error handling"""
        invalid_payloads = [
            {},  # Empty payload
            {'image': 'invalid_base64'},  # Invalid base64
            {'image': ''},  # Empty image
        ]
        
        payload = random.choice(invalid_payloads)
        
        with self.client.post("/predict", 
                            json=payload,
                            catch_response=True) as response:
            # We expect this to fail gracefully with 400
            if response.status_code in [400, 500]:
                response.success()  # This is expected behavior
                print(f"✅ Error handling working: {response.status_code}")
            else:
                response.failure(f"Unexpected response to invalid data: {response.status_code}")

class HeavyLoadUser(VehicleClassifierUser):
    """Heavy load user with shorter wait times for stress testing"""
    wait_time = between(0.1, 0.5)  # Much shorter wait time
    
    @task(10)
    def rapid_fire_predictions(self):
        """Make rapid predictions for stress testing"""
        test_image = random.choice(self.test_images)
        
        files = {
            'file': ('stress_test.jpg', test_image['data'], 'image/jpeg')
        }
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Stress test failed: {response.status_code}")

class ReturnUser(HttpUser):
    """Simulates returning users who make multiple requests in sequence"""
    wait_time = between(0.5, 2)
    
    def on_start(self):
        self.test_images = VehicleClassifierUser.create_test_images(self)
    
    @task
    def user_session(self):
        """Simulate a user session with multiple requests"""
        # Health check
        self.client.get("/")
        
        # Make 3-5 predictions in sequence
        num_predictions = random.randint(3, 5)
        for _ in range(num_predictions):
            test_image = random.choice(self.test_images)
            files = {'file': ('session_test.jpg', test_image['data'], 'image/jpeg')}
            self.client.post("/predict", files=files)
            time.sleep(random.uniform(0.5, 1.5))  # Short pause between requests


# Custom locust configuration for different scenarios
class WebsiteUser(VehicleClassifierUser):
    """Normal website user behavior"""
    weight = 3  # 3x more likely to be chosen
    
class MobileUser(VehicleClassifierUser):
    """Mobile user with different patterns"""
    weight = 2
    wait_time = between(2, 5)  # Mobile users typically wait longer
    
class APIUser(VehicleClassifierUser):
    """API user making automated requests"""
    weight = 1
    wait_time = between(0.5, 1)  # API users make requests more frequently