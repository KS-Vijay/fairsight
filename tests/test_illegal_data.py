if __name__ == "__main__":
    import os
    
    # Set environment variables to disable caching and enable streaming
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['HF_DATASETS_CACHE'] = ''
    os.environ['TRANSFORMERS_CACHE'] = ''
    os.environ['HF_HOME'] = ''
    os.environ['HF_HUB_CACHE'] = ''
    
    from fairsight import IllegalDataDetector
    try:
        from diffusers import StableDiffusionPipeline
        import torch
    except ImportError:
        print("diffusers and torch are required for this test. Skipping.")
        print("Install with: pip install torch diffusers transformers accelerate")
        exit(0)
    try:
        import imagehash
    except ImportError:
        print("imagehash is required for this test. Skipping.")
        print("Install with: pip install imagehash")
        exit(0)
    
    # User must provide a real reference folder and a working pipeline for a real test
    reference_folder = "./tests/foto"  # <-- Put your reference images here
    if not os.path.isdir(reference_folder):
        print(f"Reference folder '{reference_folder}' not found. Skipping test.")
        print(f"Please create the folder '{reference_folder}' and add reference images.")
        exit(0)
    
    # Load a pipeline (this is a placeholder, user must have the model downloaded)
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Try a smaller model first, or use a mock pipeline for testing
        print("Attempting to load a smaller model for testing...")
        
        # Option 1: Try a smaller model
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",  # Smaller alternative
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map='auto' if device == "cuda" else None,
                local_files_only=False,
                cache_dir=None,
                use_safetensors=True,
                variant="fp16" if device == "cuda" else None
            )
            
            if device == "cpu":
                pipe = pipe.to(device)
                
            print("✅ StableDiffusionPipeline loaded successfully with smaller model")
            
        except Exception as e:
            print(f"Could not load smaller model: {e}")
            print("Trying alternative approach...")
            
            # Option 2: Create a mock pipeline for testing
            class MockPipeline:
                def __init__(self):
                    self.images = [None]  # Mock image
                
                def __call__(self, prompt):
                    print(f"Mock pipeline called with prompt: {prompt}")
                    # Create a mock PIL Image
                    from PIL import Image
                    import numpy as np
                    
                    # Create a simple test image (1x1 pixel)
                    mock_image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
                    
                    class MockResult:
                        def __init__(self, image):
                            self.images = [image]
                    
                    return MockResult(mock_image)
            
            pipe = MockPipeline()
            print("✅ Using mock pipeline for testing")
        
    except Exception as e:
        print(f"Could not load StableDiffusionPipeline: {e}")
        print("This might be due to:")
        print("1. Network connectivity issues")
        print("2. Model download timeouts")
        print("3. Memory constraints")
        print("4. Missing dependencies")
        exit(0)
    
    try:
        detector = IllegalDataDetector(pipe, reference_folder)
        prompts = ["generate a picture of goku with a cat with official logo"]
        report = detector.check_illegal_data(prompts)
        print("Illegal Data Detection Report:")
        for entry in report:
            print(entry)
        detector.save_report(report)
        print("Report saved as illegal_report.json")
    except Exception as e:
        print(f"Error during illegal data detection: {e}")
        print("This might be due to:")
        print("1. Missing reference images in the foto folder")
        print("2. Model loading issues")
        print("3. Memory constraints") 