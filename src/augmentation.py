import random
import string

class TyposquattingAugmenter:
    """
    Generates typosquatting variations of URLs to augment training data.
    Helps the model learn to distinguish between legitimate URLs and close mimics.
    """
    
    def __init__(self):
        self.substitutions = {
            'a': ['4', '@', 'q'],
            'b': ['8'],
            'e': ['3'],
            'g': ['9', 'q'],
            'i': ['1', 'l', '!'],
            'l': ['1', 'i'],
            'o': ['0'],
            's': ['5', 'z'],
            't': ['7', '+'],
            'z': ['2']
        }
        
    def generate_typosquats(self, url, num_variations=2):
        """
        Generate typosquatting variations for a given URL.
        
        Args:
            url: The original legitimate URL
            num_variations: Number of variations to generate
            
        Returns:
            List of typosquatted URLs
        """
        variations = set()
        
        # Extract domain to focus mutations there
        try:
            if '://' in url:
                protocol, rest = url.split('://', 1)
                if '/' in rest:
                    domain, path = rest.split('/', 1)
                    path = '/' + path
                else:
                    domain = rest
                    path = ''
            else:
                protocol = 'http'
                domain = url
                path = ''
        except:
            return []
            
        attempts = 0
        while len(variations) < num_variations and attempts < num_variations * 5:
            attempts += 1
            method = random.choice(['double', 'skip', 'swap', 'substitute'])
            
            new_domain = list(domain)
            if len(new_domain) < 4:
                continue
                
            idx = random.randint(1, len(new_domain) - 2)  # Avoid start/end for stability
            
            if method == 'double':
                # Double a character: whatsapp -> whatsaap
                new_domain.insert(idx, new_domain[idx])
                
            elif method == 'skip':
                # Skip a character: whatsapp -> whatsap
                new_domain.pop(idx)
                
            elif method == 'swap':
                # Swap adjacent characters: whatsapp -> whastapp
                new_domain[idx], new_domain[idx+1] = new_domain[idx+1], new_domain[idx]
                
            elif method == 'substitute':
                # Substitute with lookalike: google -> g00gle
                char = new_domain[idx].lower()
                if char in self.substitutions:
                    new_domain[idx] = random.choice(self.substitutions[char])
            
            variation = f"{protocol}://{''.join(new_domain)}{path}"
            if variation != url:
                variations.add(variation)
                
        return list(variations)
