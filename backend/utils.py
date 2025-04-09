import re
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def clean_text(text):
    """Lowercase and remove special characters."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def extract_duration_minutes(duration_text):
    """Convert duration string to integer minutes."""
    match = re.search(r'(\d+)', str(duration_text))
    if match:
        return int(match.group(1))
    return 30  

def build_search_text(item):
    """Create searchable text from item metadata."""
    return f"{item.get('name', '')} {item.get('type', '')} {item.get('duration', '')} Remote:{item.get('remote', '')} Adaptive:{item.get('adaptive', '')}"


def generate_description(item):
    """Generate assessment description using Gemini API."""
    if not api_key:
        print("No API key found, using default description")
        return "Comprehensive assessment for role evaluation"
        
    prompt_text = f"""Generate a concise assessment description for {item.get('name', 'this assessment')} with these properties:
    - Type: {item.get('type', 'N/A')}
    - Duration: {item.get('duration', 'N/A')}
    - Remote support: {item.get('remote', 'N/A')}
    - Adaptive support: {item.get('adaptive', 'N/A')}
    Keep it under 120 characters. Start with the assessment purpose."""
    
    try:
        print(f"Calling Gemini API for {item.get('name', 'unknown assessment')}")
        model = genai.GenerativeModel('gemini-1.5-flash')  
        response = model.generate_content(prompt_text)
        return response.text.strip().replace('**', '')
    except Exception as e:
        print(f"Error with Gemini API: {str(e)}")
        # Fall back to a template-based description
        name = item.get('name', '').split(' - ')[0]
        duration = item.get('duration', '30 minutes')
        type_str = item.get('type', 'general')
        
        if 'supervisor' in name.lower() or 'manager' in name.lower():
            return f"Leadership assessment measuring management capabilities and decision-making skills ({duration})"
        elif 'technical' in type_str.lower() or 'coding' in type_str.lower():
            return f"Technical skills evaluation focusing on job-specific competencies ({duration})"
        elif 'customer' in name.lower() or 'service' in name.lower():
            return f"Customer interaction assessment measuring service orientation and communication skills ({duration})"
        else:
            return f"Comprehensive {name} evaluation to determine candidate job fit ({duration})"

def format_response(item):
    """Format assessment item according to API specification."""
    # Generate description if missing
    description = item.get('description')
    if not description:
        description = generate_description(item)
    
    # Extract test types as an array
    # test_types = []
    # if item.get('type'):
    #     test_types = [t.strip() for t in item.get('type').split('/')]

    test_types = item.get('test_type', ["General Assessment"])
    
    # Format the item according to API specification
    return {
        "name": item.get('name', 'Assessment'),
        "url": item.get('url', ''),
        "adaptive_support": "Yes" if item.get('adaptive', '').lower() == 'yes' else "No",
        "description": description,
        "duration": extract_duration_minutes(item.get('duration', '30 minutes')),
        "remote_support": "Yes" if item.get('remote', '').lower() == 'yes' else "No",
        "test_type": test_types if test_types else ["General Assessment"]
    }