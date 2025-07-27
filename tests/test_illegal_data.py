import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == "__main__":
    import os
    from fairsight import IllegalDataDetector
    try:
        from diffusers import StableDiffusionPipeline
        import torch
    except ImportError:
        print("diffusers and torch are required for this test. Skipping.")
        exit(0)
    try:
        import imagehash
    except ImportError:
        print("imagehash is required for this test. Skipping.")
        exit(0)
    # User must provide a real reference folder and a working pipeline for a real test
    reference_folder = "./reference_images"  # <-- Put your reference images here
    if not os.path.isdir(reference_folder):
        print(f"Reference folder '{reference_folder}' not found. Skipping test.")
        exit(0)
    # Load a pipeline (this is a placeholder, user must have the model downloaded)
    try:
        #pipe = StableDiffusionPipeline.from_pretrained("andite/anything-v4.0").to("cuda")
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
    except Exception as e:
        print(f"Could not load StableDiffusionPipeline: {e}")
        exit(0)
    detector = IllegalDataDetector(pipe, reference_folder)
    prompts = ["Mickey Mouse in cyberpunk city"]
    report = detector.check_illegal_data(prompts)
    print("Illegal Data Detection Report:")
    for entry in report:
        print(entry)
    detector.save_report(report)
    print("Report saved as illegal_report.json") 