
import Levenshtein
from urllib.parse import urlparse

class TyposquattingDetector:
    """
    Heuristic-based detector for typo-squatting domains.
    Checks if a domain is suspiciously similar to a known high-value target.
    """
    
    def __init__(self):
        # List of high-value targets often impersonated
        self.protected_domains = [
            "whatsapp.com",
            "facebook.com",
            "instagram.com",
            "google.com",
            "gmail.com",
            "youtube.com",
            "twitter.com",
            "x.com",
            "linkedin.com",
            "paypal.com",
            "amazon.com",
            "netflix.com",
            "microsoft.com",
            "apple.com",
            "yahoo.com",
            "dropbox.com",
            "adobe.com",
            "chase.com",
            "wellsfargo.com",
            "bankofamerica.com"
        ]
        
    def check(self, url):
        """
        Check if the URL is a typo-squat of a protected domain.
        
        Args:
            url: The URL to check
            
        Returns:
            tuple: (is_typosquat, target_domain, similarity_score)
        """
        # Normalize URL to extract domain
        # Handle malformed protocols (e.g., https:/example.com)
        if url.startswith('http:/') and not url.startswith('http://'):
            url = url.replace('http:/', 'http://', 1)
        elif url.startswith('https:/') and not url.startswith('https://'):
            url = url.replace('https:/', 'https://', 1)
            
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
            
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Fallback: if netloc is empty but path exists (sometimes happens with bad parsing)
            if not domain and parsed.path:
                # Try to get first part of path
                potential_domain = parsed.path.split('/')[0]
                if '.' in potential_domain:
                    domain = potential_domain
            
            if domain.startswith('www.'):
                domain = domain[4:]
        except:
            return False, None, 0.0
            
        # Check against protected domains
        for target in self.protected_domains:
            # Exact match is safe (it's the real site)
            if domain == target or domain.endswith('.' + target):
                continue
                
            # Calculate Levenshtein distance
            dist = Levenshtein.distance(domain, target)
            
            # If distance is small (1 or 2 edits) and domain is not the target
            # It's likely a typosquat (e.g., whatsaap.com vs whatsapp.com is dist=1)
            # We also check length ratio to avoid false positives on short domains
            if 0 < dist <= 2:
                # Additional check: make sure it's not just a subdomain or different TLD
                # For now, simple distance check is a good start
                return True, target, dist
                
        return False, None, 0.0
