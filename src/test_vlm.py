import os
import sys

# Ensure ai_scientist is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_scientist.vlm import create_client, get_response_from_vlm
from PIL import Image

def create_dummy_image(path):
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save(path)

if __name__ == "__main__":
    test_image_path = "test_vlm_image.png"
    if not os.path.exists(test_image_path):
        create_dummy_image(test_image_path)
    
    model_name = "local/qwen2.5-vl-32b"
    print(f"Testing VLM integration with {model_name}...")
    
    try:
        client, model = create_client(model_name)
        
        response, history = get_response_from_vlm(
            msg="What color is this image?",
            image_paths=[test_image_path],
            client=client,
            model=model,
            system_message="You are a helpful assistant.",
            print_debug=True
        )
        
        print("\nSUCCESS!")
        print("Model Response:", response)
    except Exception as e:
        print("\nFAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
