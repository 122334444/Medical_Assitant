import os
import base64
import markdown
from markupsafe import Markup
from groq import Groq

# === CONFIGURATION ===
#GROQ_API_KEY = 
IMAGE_PATH = "a.jpg" 
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dicom"}

# === Medical Analysis Prompt ===
MEDICAL_QUERY = """
You are a Semantic Processing Agent (SPA) for medical diagnosis.

Your task is to process multimodal patient data (medical images, EHR, demographics, symptoms, and clinical notes) and convert it into a semantic Knowledge Graph (KG) representation for LLM-based diagnosis.

### Workflow
1. Extract pathological findings from medical images.
2. Identify key entities from clinical narratives using NLP and NER.
3. Map findings into KG triples:
   (<subject, predicate, object>)
4. Use medical ontologies such as SYMP, DO, FMA, and RadLex for entity classification.
5. Combine imaging findings and clinical text into a unified KG.
6. Generate an LLM-compatible diagnostic prompt using prompt engineering.

### Output
- Image findings
- Clinical entities
- Generated KG triples
- Final KG summary
- LLM-ready diagnostic prompt
- Possible disease prediction with short explanation

Format the response clearly using markdown and bullet points.
"""

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def analyze_image(image_path, groq_api_key):
    if not allowed_file(image_path):
        raise ValueError("Invalid file type. Allowed types: png, jpg, jpeg, dicom.")

    base64_image = encode_image(image_path)
    client = Groq(api_key=groq_api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": MEDICAL_QUERY},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )

    markdown_result = chat_completion.choices[0].message.content
    html_result = Markup(markdown.markdown(markdown_result, extensions=['fenced_code', 'tables']))
    
    return markdown_result, html_result

if __name__ == "__main__":
    try:
        markdown_text, html_text = analyze_image(IMAGE_PATH, GROQ_API_KEY)
        print("\n=== Analysis Result (Markdown) ===\n")
        print(markdown_text)
        
        print("\n=== Analysis Result (HTML) ===\n")
        print(html_text)
    except Exception as e:
        print(f"Error: {e}")