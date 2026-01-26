#!/usr/bin/env python3
"""
API Authentication Handler for Materials Project API

This module provides robust handling of API authentication issues,
including validation, fallback mechanisms, and clear error messages.
"""

import os
import requests
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIAuthenticationHandler:
    """Handles Materials Project API authentication with robust error handling."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the API authentication handler.
        
        Args:
            api_key: Materials Project API key (optional)
        """
        self.api_key = api_key or os.getenv('MP_API_KEY')
        self.base_url = "https://materialsproject.org/rest/v2"
        self.headers = {}
        
        if self.api_key:
            self.headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
    
    def validate_api_key(self) -> Dict[str, Any]:
        """
        Validate the API key by making a simple test request.
        
        Returns:
            Dict containing validation result and details
        """
        if not self.api_key:
            return {
                'valid': False,
                'error': 'No API key provided',
                'message': 'Please provide a valid Materials Project API key',
                'suggestions': [
                    'Get a free API key at: https://materialsproject.org/api',
                    'Set the MP_API_KEY environment variable',
                    'Or provide the API key in the application settings'
                ]
            }
        
        try:
            # Make a simple test request to validate the API key
            test_url = f"{self.base_url}/materials/summary"
            params = {
                '_limit': 1,
                '_fields': 'material_id'
            }
            
            response = requests.get(test_url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 200:
                return {
                    'valid': True,
                    'message': 'API key is valid',
                    'status_code': response.status_code
                }
            elif response.status_code == 401:
                return {
                    'valid': False,
                    'error': 'Invalid API key',
                    'message': 'The provided API key is not valid or has expired',
                    'status_code': response.status_code,
                    'suggestions': [
                        'Verify your API key is correct',
                        'Check if your API key has expired',
                        'Regenerate your API key at: https://materialsproject.org/api',
                        'Ensure you are using the correct API endpoint'
                    ]
                }
            elif response.status_code == 429:
                return {
                    'valid': False,
                    'error': 'Rate limit exceeded',
                    'message': 'API rate limit exceeded. Please try again later.',
                    'status_code': response.status_code,
                    'suggestions': [
                        'Wait a few minutes and try again',
                        'Check your API usage limits',
                        'Consider using synthetic data as fallback'
                    ]
                }
            elif response.status_code == 500:
                return {
                    'valid': False,
                    'error': 'Server error',
                    'message': 'Materials Project server error. Please try again later.',
                    'status_code': response.status_code,
                    'suggestions': [
                        'Try again in a few minutes',
                        'Check Materials Project status at: https://status.materialsproject.org',
                        'Use synthetic data as fallback'
                    ]
                }
            else:
                return {
                    'valid': False,
                    'error': f'Unexpected status code: {response.status_code}',
                    'message': f'API returned unexpected status: {response.status_code}',
                    'status_code': response.status_code,
                    'suggestions': [
                        'Check your internet connection',
                        'Verify the API endpoint is correct',
                        'Use synthetic data as fallback'
                    ]
                }
                
        except requests.exceptions.Timeout:
            return {
                'valid': False,
                'error': 'Request timeout',
                'message': 'API request timed out. Please check your internet connection.',
                'suggestions': [
                    'Check your internet connection',
                    'Try again with a longer timeout',
                    'Use synthetic data as fallback'
                ]
            }
        except requests.exceptions.ConnectionError:
            return {
                'valid': False,
                'error': 'Connection error',
                'message': 'Unable to connect to Materials Project API.',
                'suggestions': [
                    'Check your internet connection',
                    'Verify the API endpoint is accessible',
                    'Use synthetic data as fallback'
                ]
            }
        except requests.exceptions.RequestException as e:
            return {
                'valid': False,
                'error': 'Request error',
                'message': f'Error making API request: {str(e)}',
                'suggestions': [
                    'Check your internet connection',
                    'Verify the API endpoint is correct',
                    'Use synthetic data as fallback'
                ]
            }
        except Exception as e:
            return {
                'valid': False,
                'error': 'Unknown error',
                'message': f'Unexpected error: {str(e)}',
                'suggestions': [
                    'Check your internet connection',
                    'Verify the API endpoint is correct',
                    'Use synthetic data as fallback'
                ]
            }
    
    def get_api_status(self) -> Dict[str, Any]:
        """
        Get the current API status and authentication information.
        
        Returns:
            Dict containing API status information
        """
        validation_result = self.validate_api_key()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'api_key_provided': bool(self.api_key),
            'api_key_valid': validation_result['valid'],
            'api_key_status': validation_result.get('message', 'Unknown'),
            'base_url': self.base_url,
            'validation_details': validation_result
        }
    
    def log_api_status(self):
        """Log the current API status with appropriate level."""
        status = self.get_api_status()
        
        if status['api_key_valid']:
            logger.info(f"✅ API Authentication: {status['api_key_status']}")
        else:
            logger.warning(f"⚠️ API Authentication: {status['api_key_status']}")
            if 'suggestions' in status['validation_details']:
                for suggestion in status['validation_details']['suggestions']:
                    logger.info(f"   → {suggestion}")
    
    def get_fallback_message(self) -> str:
        """
        Get a user-friendly message about using fallback data.
        
        Returns:
            String message about fallback options
        """
        status = self.get_api_status()
        
        if not status['api_key_valid']:
            message = f"""
            **API Authentication Issue Detected**
            
            Status: {status['api_key_status']}
            
            **Recommended Actions:**
            """
            
            if 'suggestions' in status['validation_details']:
                for i, suggestion in enumerate(status['validation_details']['suggestions'], 1):
                    message += f"\n{i}. {suggestion}"
            
            message += f"""
            
            **Using Synthetic Data:**
            The application will continue using synthetic data for materials discovery.
            While not as accurate as real Materials Project data, synthetic data still
            provides valuable insights for materials exploration.
            
            **Data Quality Comparison:**
            - Real Materials Project data: High accuracy, experimentally validated
            - Synthetic data: Good for exploration, may have higher uncertainty
            
            **To Enable Real Data:**
            1. Get a free API key: https://materialsproject.org/api
            2. Set MP_API_KEY environment variable
            3. Restart the application
            """
            
            return message
        else:
            return "✅ API authentication successful - using real Materials Project data"

def handle_api_authentication(api_key: Optional[str] = None) -> APIAuthenticationHandler:
    """
    Create and validate an API authentication handler.
    
    Args:
        api_key: Optional API key
        
    Returns:
        Configured APIAuthenticationHandler instance
    """
    handler = APIAuthenticationHandler(api_key)
    handler.log_api_status()
    
    return handler

def validate_and_use_api_key(api_key: Optional[str] = None) -> bool:
    """
    Validate API key and return whether to use real API or fallback.
    
    Args:
        api_key: Optional API key
        
    Returns:
        Boolean indicating whether to use real API (True) or fallback (False)
    """
    handler = handle_api_authentication(api_key)
    status = handler.get_api_status()
    
    if status['api_key_valid']:
        logger.info("✅ Using real Materials Project API")
        return True
    else:
        logger.warning("⚠️ Using synthetic data fallback")
        logger.info(handler.get_fallback_message())
        return False

# Example usage and testing
if __name__ == "__main__":
    print("=== API Authentication Handler Test ===")
    
    # Test with no API key
    print("\n1. Testing with no API key:")
    handler1 = handle_api_authentication(None)
    print(handler1.get_api_status())
    
    # Test with environment variable
    print("\n2. Testing with environment variable:")
    os.environ['MP_API_KEY'] = 'test_key'
    handler2 = handle_api_authentication()
    print(handler2.get_api_status())
    
    # Test validation function
    print("\n3. Testing validation function:")
    use_real_api = validate_and_use_api_key('test_key')
    print(f"Should use real API: {use_real_api}")