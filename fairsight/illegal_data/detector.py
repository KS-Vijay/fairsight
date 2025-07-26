import os
import glob
import json
from typing import List, Dict, Any, Optional
from PIL import Image
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..auth import require_premium_access, TieredAccessError

try:
    import torch
except ImportError:
    torch = None
try:
    import open_clip
except ImportError:
    open_clip = None
try:
    from diffusers import StableDiffusionPipeline
except ImportError:
    StableDiffusionPipeline = None
try:
    import imagehash
except ImportError:
    imagehash = None

class IllegalDataDetector:
    def __init__(self, pipeline, reference_folder: str, device: str = "cuda", phash_threshold: int = 8, 
                 user_api_key: Optional[str] = None, api_base_url: str = "http://localhost:5000"):
        # Illegal data detection is a premium feature - requires API key
        try:
            require_premium_access("illegal_data_detection", user_api_key, api_base_url)
        except TieredAccessError as e:
            raise e
        
        if torch is None or open_clip is None or imagehash is None:
            raise ImportError("Required libraries 'torch', 'open_clip_torch', and 'imagehash' must be installed.")
        self.pipeline = pipeline
        self.reference_folder = reference_folder
        self.device = device
        self.phash_threshold = phash_threshold
        self.user_api_key = user_api_key
        self.api_base_url = api_base_url
        self.reference_images = self._load_reference_images(reference_folder)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.reference_embeddings = self._compute_reference_embeddings()
        self.reference_hashes = self._compute_reference_hashes()

    def _load_reference_images(self, folder: str) -> List[str]:
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
        if not files:
            raise ValueError(f"No reference images found in {folder}")
        return files

    def _compute_reference_embeddings(self) -> Dict[str, torch.Tensor]:
        embeddings = {}
        for img_path in self.reference_images:
            img = Image.open(img_path).convert("RGB")
            img_input = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_image(img_input)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings[img_path] = emb.cpu()
        return embeddings

    def _compute_reference_hashes(self) -> Dict[str, Any]:
        hashes = {}
        for img_path in self.reference_images:
            img = Image.open(img_path).convert("RGB")
            hashes[img_path] = imagehash.phash(img)
        return hashes

    def inject_probes(self, prompts: List[str], num_images: int = 1) -> Dict[str, List[str]]:
        # Verify premium access for probe injection
        try:
            require_premium_access("illegal_data_detection", self.user_api_key, self.api_base_url)
        except TieredAccessError as e:
            raise e
            
        if StableDiffusionPipeline is None:
            raise ImportError("diffusers library is required for image generation.")
        os.makedirs("generated_probes", exist_ok=True)
        results = {}
        for prompt in prompts:
            images = []
            for i in range(num_images):
                image = self.pipeline(prompt).images[0]
                out_path = os.path.join("generated_probes", f"{prompt.replace(' ', '_')}_{i}.png")
                image.save(out_path)
                images.append(out_path)
            results[prompt] = images
        return results

    def check_illegal_data(self, prompts: List[str], threshold: float = 0.75, phash_threshold: int = None) -> List[Dict[str, Any]]:
        # Verify premium access for illegal data checking
        try:
            require_premium_access("illegal_data_detection", self.user_api_key, self.api_base_url)
        except TieredAccessError as e:
            raise e
            
        if phash_threshold is None:
            phash_threshold = self.phash_threshold
        probe_dict = self.inject_probes(prompts)
        report = []
        for prompt, gen_paths in probe_dict.items():
            for gen_path in gen_paths:
                gen_img = Image.open(gen_path).convert("RGB")
                # CLIP
                gen_input = self.preprocess(gen_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    gen_emb = self.model.encode_image(gen_input)
                    gen_emb = gen_emb / gen_emb.norm(dim=-1, keepdim=True)
                best_clip_score = 0.0
                best_clip_match = None
                for ref_path, ref_emb in self.reference_embeddings.items():
                    sim = (gen_emb @ ref_emb.T).item()
                    if sim > best_clip_score:
                        best_clip_score = sim
                        best_clip_match = ref_path
                # pHash
                gen_hash = imagehash.phash(gen_img)
                best_phash_dist = 64  # max for 64-bit hash
                best_phash_match = None
                for ref_path, ref_hash in self.reference_hashes.items():
                    dist = gen_hash - ref_hash
                    if dist < best_phash_dist:
                        best_phash_dist = dist
                        best_phash_match = ref_path
                # Consensus logic
                flagged = (best_clip_score > threshold and best_phash_dist <= phash_threshold)
                reason = []
                if best_clip_score > threshold:
                    reason.append(f'CLIP>{threshold}')
                if best_phash_dist <= phash_threshold:
                    reason.append(f'pHash<={phash_threshold}')
                if not reason:
                    reason.append('Not flagged')
                report.append({
                    "prompt": prompt,
                    "generated_image": gen_path,
                    "clip_matched_image": best_clip_match,
                    "clip_similarity_score": round(best_clip_score, 4),
                    "phash_matched_image": best_phash_match,
                    "phash_distance": int(best_phash_dist),
                    "flagged": flagged,
                    "reason": ', '.join(reason)
                })
        return report

    def save_report(self, report: Any, path: str = "illegal_report.json"):
        # Verify premium access for report saving
        try:
            require_premium_access("illegal_data_detection", self.user_api_key, self.api_base_url)
        except TieredAccessError as e:
            raise e
            
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

# Convenience function for illegal data detection
def detect_illegal_data(pipeline, reference_folder: str, prompts: List[str], 
                       device: str = "cuda", phash_threshold: int = 8,
                       user_api_key: Optional[str] = None, 
                       api_base_url: str = "http://localhost:5000",
                       **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function for illegal data detection with premium access control.
    
    Args:
        pipeline: Stable Diffusion pipeline
        reference_folder: Folder containing reference images
        prompts: List of prompts to check
        device: Device to use for computation
        phash_threshold: pHash threshold for similarity
        user_api_key: API key for premium access
        api_base_url: Base URL for API verification
        **kwargs: Additional arguments for IllegalDataDetector
        
    Returns:
        List of detection results
        
    Raises:
        TieredAccessError: If premium access is required but not provided
    """
    detector = IllegalDataDetector(
        pipeline=pipeline,
        reference_folder=reference_folder,
        device=device,
        phash_threshold=phash_threshold,
        user_api_key=user_api_key,
        api_base_url=api_base_url,
        **kwargs
    )
    return detector.check_illegal_data(prompts) 