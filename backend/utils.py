import re

def clean_text(text):
    """Lowercase and remove special characters from a string."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def format_duration(duration_text):
    """Extract numeric minutes from duration string."""
    match = re.search(r'\d+', duration_text)
    return int(match.group()) if match else 0

def build_search_text(item):
    """Create a text string from item fields for embedding."""
    return f"{item['name']} {item['type']} {item['duration']} Remote: {item['remote']} Adaptive: {item['adaptive']}"
