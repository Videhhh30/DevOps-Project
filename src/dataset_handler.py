"""
Dataset Handler Module
Handles dataset loading, downloading, and generation
"""

import pandas as pd
import requests
import os


def download_dataset(url: str, save_path: str) -> bool:
    """
    Download dataset from URL
    
    Args:
        url: URL to download from
        save_path: Path to save the dataset
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading dataset from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save to file
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Dataset downloaded successfully to {save_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def generate_synthetic_dataset(save_path: str, num_samples: int = 500) -> bool:
    """
    Generate a realistic synthetic dataset with challenging cases
    
    Args:
        save_path: Path to save the dataset
        num_samples: Number of samples to generate
        
    Returns:
        True if successful
    """
    print(f"Generating realistic synthetic dataset with {num_samples} samples...")
    
    import random
    random.seed(42)
    
    # Legitimate URLs - varied and realistic
    legitimate_urls = [
        "https://www.google.com/search?q=python",
        "https://github.com/user/repository",
        "https://en.wikipedia.org/wiki/Machine_Learning",
        "https://www.amazon.com/product/B08N5WRWNW",
        "https://docs.microsoft.com/en-us/azure",
        "https://developer.apple.com/documentation",
        "https://www.facebook.com/profile.php?id=123",
        "https://twitter.com/user/status/123456",
        "https://www.linkedin.com/in/username",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.reddit.com/r/programming/comments/abc",
        "https://stackoverflow.com/questions/12345/how-to",
        "https://medium.com/@author/article-title-123",
        "https://www.netflix.com/browse/genre/83",
        "https://open.spotify.com/playlist/37i9dQZF1DX",
        "https://www.paypal.com/myaccount/home",
        "https://mail.google.com/mail/u/0/",
        "https://www.dropbox.com/home",
        "https://drive.google.com/drive/my-drive",
        "https://www.instagram.com/explore/",
        # Some legitimate but suspicious-looking URLs (edge cases)
        "http://192.168.0.1",  # Local router - legitimate
        "https://bit.ly/3xYz",  # URL shortener - could be legitimate
        "https://t.co/AbC123",  # Twitter shortener - legitimate
        "https://goo.gl/maps/xyz",  # Google Maps shortener
        "https://www.example-company.com",  # Hyphenated domain
        "https://my-account.service.com",  # Multiple hyphens
        "https://www.site123.com",  # Numbers in domain
        "http://old-website.org",  # HTTP but legitimate
    ]
    
    # Phishing URLs - realistic phishing patterns
    phishing_urls = [
        "http://paypal-secure-login.tk",
        "http://paypal.com-verify.tk",
        "http://secure-paypal.ml",
        "http://amaz0n-account.ga",
        "http://amazon-security.cf",
        "http://appleid-locked.xyz",
        "http://apple.com-verify.top",
        "http://microsoft-update.work",
        "http://windows-security.tk",
        "http://facebook-security.ml",
        "http://fb-verify-account.ga",
        "http://netflix-billing.cf",
        "http://netflix.payment-update.com",
        "http://google-security.xyz",
        "http://accounts-google.tk",
        "http://192.168.1.1/admin/login.php",
        "http://192.168.0.1/phpmyadmin",
        "http://verify@paypal.com.tk",
        "http://user@amazon-security.ml",
        "http://account-suspended.ga",
        "http://urgent-action-required.cf",
        "http://claim-your-prize.xyz",
        "http://winner-notification.top",
        "http://bank-verify.work",
        "http://secure-banking.tk",
        # Sophisticated phishing (harder to detect)
        "https://paypa1.com",  # Typosquatting with 1 instead of l
        "https://g00gle.com",  # Zeros instead of o's
        "https://micros0ft.com",
        "https://facebo0k.com",
        "https://arnaz0n.com",  # rn looks like m
    ]
    
    # Generate dataset with variations
    urls = []
    labels = []
    
    # Add legitimate URLs with variations
    for i in range(num_samples // 2):
        base_url = legitimate_urls[i % len(legitimate_urls)]
        
        # Add realistic variations
        if i % 5 == 0:
            url = base_url + f"/page/{i}"
        elif i % 5 == 1:
            url = base_url + f"?id={i}&ref=search"
        elif i % 5 == 2:
            url = base_url + f"#section{i}"
        elif i % 5 == 3:
            url = base_url + f"/category/item-{i}"
        else:
            url = base_url
        
        urls.append(url)
        labels.append(0)  # 0 = legitimate
    
    # Add phishing URLs with variations
    for i in range(num_samples // 2):
        base_url = phishing_urls[i % len(phishing_urls)]
        
        # Add variations to make them more realistic
        if i % 5 == 0:
            url = base_url + f"?token={random.randint(1000, 9999)}"
        elif i % 5 == 1:
            url = base_url + f"/verify/{random.randint(100, 999)}"
        elif i % 5 == 2:
            url = base_url + f"/login.php?redirect=home"
        elif i % 5 == 3:
            url = base_url + f"?session={random.randint(10000, 99999)}"
        else:
            url = base_url
        
        urls.append(url)
        labels.append(1)  # 1 = phishing
    
    # Add challenging ambiguous cases (20% of dataset) to make it more realistic
    ambiguous_count = num_samples // 5
    
    # Legitimate URLs that look suspicious (hard negatives)
    hard_negatives = [
        "http://192.168.1.1",  # Local IP but legitimate
        "http://10.0.0.1/admin",  # Internal admin panel
        "https://bit.ly/2xYz3aB",  # Shortener but legitimate
        "https://t.co/AbC123XyZ",  # Twitter shortener
        "http://localhost:8080/test",  # Development server
        "https://my-secure-site.com",  # Hyphens but legitimate
        "https://www.site-with-many-hyphens.org",
        "https://www.123website.com",  # Numbers but legitimate
        "https://subdomain.subdomain.example.com",  # Multiple subdomains
        "http://old-legacy-system.company.internal",
    ]
    
    # Phishing URLs that look legitimate (hard positives)
    hard_positives = [
        "https://paypa1.com/signin",  # Typosquatting
        "https://g00gle.com/accounts",  # Character substitution
        "https://www.micros0ft.com/login",
        "https://www.facebo0k.com/security",
        "https://www.arnaz0n.com/account",  # rn looks like m
        "https://www.app1e.com/id",
        "https://www.netfIix.com/billing",  # I instead of l
        "https://secure-paypal-login.com",  # Looks official but isn't
        "https://accounts-google-verify.com",
        "https://microsoft-security-alert.com",
    ]
    
    # Add hard negatives (legitimate but suspicious-looking)
    for i in range(ambiguous_count // 2):
        url = hard_negatives[i % len(hard_negatives)]
        if i % 3 == 0:
            url += f"?session={random.randint(1000, 9999)}"
        urls.append(url)
        labels.append(0)  # Actually legitimate
    
    # Add hard positives (phishing but legitimate-looking)
    for i in range(ambiguous_count // 2):
        url = hard_positives[i % len(hard_positives)]
        if i % 3 == 0:
            url += f"/verify?token={random.randint(10000, 99999)}"
        urls.append(url)
        labels.append(1)  # Actually phishing
    
    # Create DataFrame
    df = pd.DataFrame({'url': urls, 'label': labels})
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    df.to_csv(save_path, index=False)
    print(f"Realistic synthetic dataset saved to {save_path}")
    print(f"  Legitimate URLs: {(df['label'] == 0).sum()}")
    print(f"  Phishing URLs: {(df['label'] == 1).sum()}")
    print(f"  Total samples: {len(df)}")
    print(f"  Includes edge cases and ambiguous URLs for realistic accuracy")
    
    return True


def load_or_create_dataset(dataset_path: str = 'data/dataset.csv',
                           download_url: str = None) -> str:
    """
    Load dataset from file, download if needed, or generate synthetic data
    
    Args:
        dataset_path: Path to dataset file
        download_url: Optional URL to download dataset from
        
    Returns:
        Path to the dataset file
    """
    # Check if dataset exists
    if os.path.exists(dataset_path):
        print(f"Dataset found at {dataset_path}")
        return dataset_path
    
    print(f"Dataset not found at {dataset_path}")
    
    # Try to download if URL provided
    if download_url:
        print(f"\nAttempting to download dataset...")
        if download_dataset(download_url, dataset_path):
            return dataset_path
        print("Download failed. Falling back to synthetic data generation.")
    
    # Generate synthetic dataset
    print(f"\nGenerating synthetic dataset...")
    generate_synthetic_dataset(dataset_path, num_samples=500)
    
    return dataset_path


if __name__ == "__main__":
    # Test dataset handler
    print("Testing Dataset Handler\n")
    print("="*60)
    
    # Test synthetic dataset generation
    test_path = 'data/test_synthetic.csv'
    generate_synthetic_dataset(test_path, num_samples=100)
    
    # Load and display info
    df = pd.read_csv(test_path)
    print(f"\nDataset info:")
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print("\n" + "="*60)
    print("Dataset handler test completed!")
