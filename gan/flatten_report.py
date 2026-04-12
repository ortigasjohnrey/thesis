import base64
import os
import re

# Configuration
HTML_INPUT = "gan/gan_performance_report.html"
HTML_OUTPUT = "gan/gan_report_standalone.html"

def get_base64_image(img_rel_path):
    """Converts a relative image path in the HTML to a Base64 Data URI."""
    # The HTML is in 'gan/', so paths starting with '../' are relative to project root
    if img_rel_path.startswith("../"):
        # ../reports/gan_validation/... -> reports/gan_validation/...
        target_path = img_rel_path[3:] 
    else:
        # gold_fidelity_audit.png -> gan/gold_fidelity_audit.png
        target_path = os.path.join("gan", img_rel_path)
    
    # Absolute path for reading
    abs_path = os.path.abspath(target_path)
    
    if not os.path.exists(abs_path):
        print(f"Warning: Image not found at {abs_path}")
        return img_rel_path
        
    with open(abs_path, "rb") as f:
        data = f.read()
        b64_str = base64.b64encode(data).decode('utf-8')
        
    ext = os.path.splitext(abs_path)[1].lower().replace(".", "")
    mime = "image/png" if ext == "png" else "image/jpeg"
    
    return f"data:image/{ext};base64,{b64_str}"

def flatten():
    if not os.path.exists(HTML_INPUT):
        print(f"Error: Input HTML not found at {HTML_INPUT}")
        return

    with open(HTML_INPUT, "r", encoding="utf-8") as f:
        content = f.read()

    # Find and replace all src attributes in <img> tags
    pattern = r'(<img\s+[^>]*?src=")([^"]+)(")'
    
    def replacement_func(match):
        prefix = match.group(1)
        img_path = match.group(2)
        suffix = match.group(3)
        
        # Don't re-encode if already data URI
        if img_path.startswith("data:"):
            return match.group(0)
            
        b64_data = get_base64_image(img_path)
        return f"{prefix}{b64_data}{suffix}"

    new_content = re.sub(pattern, replacement_func, content)

    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"Successfully flattened HTML into: {HTML_OUTPUT}")
    print(f"Final file size: {os.path.getsize(HTML_OUTPUT) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    flatten()
