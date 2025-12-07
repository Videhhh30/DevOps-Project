"""
Enhanced URL Feature Extraction
Extracts intelligent features from URLs for better phishing detection
"""

import re
import math
from urllib.parse import urlparse
from typing import Dict, List
import numpy as np


class URLFeatureExtractor:
    """Extract intelligent features from URLs"""
    
    # Suspicious TLDs commonly used in phishing
    SUSPICIOUS_TLDS = {
        'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'click',
        'link', 'racing', 'download', 'stream', 'science', 'loan'
    }
    
    # Trusted TLDs
    TRUSTED_TLDS = {
        'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
        'ac.in', 'edu.in', 'gov.in', 'co.in', 'in'
    }
    
    # Common phishing keywords
    PHISHING_KEYWORDS = [
        'verify', 'account', 'update', 'confirm', 'login', 'signin',
        'secure', 'banking', 'suspend', 'restricted', 'unusual',
        'click', 'urgent', 'immediately', 'expire', 'alert'
    ]
    
    def __init__(self):
        """Initialize feature extractor"""
        pass
    
    def extract_features(self, url: str) -> np.ndarray:
        """
        Extract comprehensive features from URL
        
        Args:
            url: URL string
            
        Returns:
            Feature vector (numpy array)
        """
        features = []
        
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc
            path = parsed.path
            query = parsed.query
            
            # 1. URL Length Features
            features.append(len(url))                    # Total URL length
            features.append(len(domain))                 # Domain length
            features.append(len(path))                   # Path length
            features.append(len(query))                  # Query length
            
            # 2. Character Count Features
            features.append(url.count('.'))              # Number of dots
            features.append(url.count('-'))              # Number of hyphens
            features.append(url.count('_'))              # Number of underscores
            features.append(url.count('/'))              # Number of slashes
            features.append(url.count('?'))              # Number of question marks
            features.append(url.count('='))              # Number of equals
            features.append(url.count('@'))              # Number of @ (phishing indicator)
            features.append(url.count('&'))              # Number of ampersands
            
            # 3. Digit Features
            digit_count = sum(c.isdigit() for c in url)
            features.append(digit_count)                 # Total digits
            features.append(digit_count / max(len(url), 1))  # Digit ratio
            
            # 4. Domain Features
            domain_parts = domain.split('.')
            features.append(len(domain_parts))           # Number of subdomains
            features.append(len(domain_parts[-1]) if domain_parts else 0)  # TLD length
            
            # 5. TLD Reputation
            tld = domain_parts[-1] if domain_parts else ''
            features.append(1 if tld in self.SUSPICIOUS_TLDS else 0)  # Suspicious TLD
            features.append(1 if tld in self.TRUSTED_TLDS else 0)     # Trusted TLD
            
            # 6. Security Features
            features.append(1 if parsed.scheme == 'https' else 0)  # Uses HTTPS
            features.append(1 if parsed.port and parsed.port not in [80, 443] else 0)  # Non-standard port
            
            # 7. IP Address Detection
            ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
            features.append(1 if re.search(ip_pattern, domain) else 0)  # Contains IP
            
            # 8. Phishing Keyword Detection
            url_lower = url.lower()
            keyword_count = sum(1 for keyword in self.PHISHING_KEYWORDS if keyword in url_lower)
            features.append(keyword_count)               # Number of phishing keywords
            features.append(1 if keyword_count > 0 else 0)  # Has phishing keywords
            
            # 9. Entropy (randomness of domain)
            domain_entropy = self._calculate_entropy(domain)
            features.append(domain_entropy)              # Domain entropy
            
            # 10. Special Pattern Detection
            features.append(1 if '--' in url else 0)     # Double hyphen
            features.append(1 if '//' in path else 0)    # Double slash in path
            features.append(1 if 'www' in domain and domain.count('www') > 1 else 0)  # Multiple www
            
            # 11. Path Depth
            path_depth = len([p for p in path.split('/') if p])
            features.append(path_depth)                  # Path depth
            
            # 12. Domain-Path Ratio
            domain_path_ratio = len(domain) / max(len(path), 1)
            features.append(domain_path_ratio)           # Domain to path ratio
            
            # 13. Suspicious Patterns
            features.append(1 if re.search(r'[0-9]+[a-z]+[0-9]+', domain) else 0)  # Mixed digits/letters
            features.append(1 if len(domain) > 50 else 0)  # Very long domain
            
        except Exception as e:
            # If parsing fails, return zero features
            features = [0] * 35
        
        return np.array(features, dtype=float)
    
    def _calculate_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of text
        Higher entropy = more random (suspicious)
        
        Args:
            text: Input string
            
        Returns:
            Entropy value
        """
        if not text:
            return 0.0
        
        # Calculate character frequency
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        text_len = len(text)
        for count in freq.values():
            probability = count / text_len
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features
        
        Returns:
            List of feature names
        """
        return [
            'url_length', 'domain_length', 'path_length', 'query_length',
            'dot_count', 'hyphen_count', 'underscore_count', 'slash_count',
            'question_count', 'equals_count', 'at_count', 'ampersand_count',
            'digit_count', 'digit_ratio',
            'subdomain_count', 'tld_length',
            'is_suspicious_tld', 'is_trusted_tld',
            'is_https', 'has_nonstandard_port',
            'has_ip_address',
            'phishing_keyword_count', 'has_phishing_keywords',
            'domain_entropy',
            'has_double_hyphen', 'has_double_slash', 'has_multiple_www',
            'path_depth', 'domain_path_ratio',
            'has_mixed_alphanum', 'has_very_long_domain'
        ]


if __name__ == "__main__":
    # Test the feature extractor
    print("Testing URL Feature Extractor\n")
    print("=" * 70)
    
    extractor = URLFeatureExtractor()
    
    test_urls = [
        "https://msrit.com/results",
        "https://www.google.com",
        "http://paypal-secure-login.tk",
        "https://paypa1.com/verify-account",
        "http://192.168.1.1/admin",
        "https://github.com/user/repo"
    ]
    
    feature_names = extractor.get_feature_names()
    
    for url in test_urls:
        print(f"\nURL: {url}")
        print("-" * 70)
        features = extractor.extract_features(url)
        
        # Show key features
        print(f"  URL Length: {features[0]:.0f}")
        print(f"  Domain Length: {features[1]:.0f}")
        print(f"  Uses HTTPS: {'Yes' if features[18] else 'No'}")
        print(f"  Suspicious TLD: {'Yes' if features[16] else 'No'}")
        print(f"  Trusted TLD: {'Yes' if features[17] else 'No'}")
        print(f"  Has IP: {'Yes' if features[20] else 'No'}")
        print(f"  Phishing Keywords: {features[21]:.0f}")
        print(f"  Domain Entropy: {features[23]:.2f}")
        print(f"  Total Features: {len(features)}")
    
    print("\n" + "=" * 70)
    print("Feature extraction test completed!")
