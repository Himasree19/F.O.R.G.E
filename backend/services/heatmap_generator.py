from backend.services.image_visual_evidence import generate_image_visual_evidence

def generate_heatmap(image_path: str):
    return generate_image_visual_evidence(image_path)["overlay_url"]

def generate_gradcam(image_path: str):
    return generate_heatmap(image_path)
