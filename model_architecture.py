"""
# NeuroSeek-MoE Model Architecture (DeepSeek-MoE variant)

This module provides a self-contained scaffold for a multimodal Mixture-of-Experts
architecture that fuses text and image reasoning, captures routing signals, and
produces hybrid outputs (structured text + diagram) alongside routing
visualizations and metrics suitable for explainability and benchmarking.

Design goals implemented here:
- Modality-specific experts: E_text, E_image, and fusion E_joint
- Shared gating network with top-k routing per family
- Routing visualization (SVG bars) and JSON sidecar export
- Output pipeline stubs: text generation and diagram generation adapter

Note: This file is a framework-free prototype (no heavy DL deps). Replace the
expert forward() bodies and the diffusion adapter stub with real model calls in
your environment. The routing instrumentation and file outputs are ready-to-use.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import json
import math
import os
import random
import time


# =========================
# Core types and utilities
# =========================


class Disease(Enum):
    ALZHEIMERS = "AD"
    PARKINSONS = "PD"
    ALS = "ALS"
    HUNTINGTONS = "HD"
    MULTIPLE_SCLEROSIS = "MS"


def _ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _now_ms() -> int:
    return int(time.time() * 1000)


# =====================
# Expert base interfaces
# =====================


class ExpertBase:
    def __init__(self, name: str, disease_scope: Optional[Disease] = None) -> None:
        self.name = name
        self.disease_scope = disease_scope

    def forward(self, tokens_or_tensor) -> List[float]:
        raise NotImplementedError


class TextExpert(ExpertBase):
    def forward(self, tokens: List[int]) -> List[float]:
        # Stub: produce a fixed-size representation
        rnd = random.Random(hash(self.name) % (2**31 - 1))
        return [rnd.random() for _ in range(16)]


class ImageExpert(ExpertBase):
    def forward(self, image_tensor) -> List[float]:
        # Stub: produce a fixed-size representation
        rnd = random.Random(hash(self.name) % (2**31 - 1))
        return [rnd.random() for _ in range(16)]


class JointExpert(ExpertBase):
    def forward(self, text_repr: List[float], image_repr: List[float]) -> List[float]:
        # Stub: simple elementwise mix
        n = min(len(text_repr), len(image_repr))
        return [(text_repr[i] + image_repr[i]) / 2.0 for i in range(n)]


# =====================
# Pretrained/backbone adapters (optional)
# =====================


class BioLLMTextExpert(TextExpert):
    """Real biomedical LLM encoder using transformers or sentence-transformers.

    Uses BioBERT/SciBERT embeddings if available, otherwise BERT-base.
    """

    def __init__(self, name: str = "E_text_bio", model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract") -> None:
        super().__init__(name=name)
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._encoder = None
        
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            import torch  # type: ignore
            
            print(f"🔄 Loading {model_name} for text encoding...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            print(f"✅ Loaded {model_name}")
        except ImportError:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                print(f"🔄 Loading sentence-transformers for text encoding...")
                self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"✅ Loaded sentence-transformers")
            except ImportError:
                print("⚠️  No transformer libraries found, using simple embeddings")

    def forward(self, tokens: List[int]) -> List[float]:
        if self._model is not None and self._tokenizer is not None:
            try:
                import torch  # type: ignore
                # Convert token IDs back to text (simplified)
                text = " ".join(str(t) for t in tokens[:512])  # Truncate
                inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
                    return embeddings.cpu().numpy().tolist()
            except Exception as e:
                print(f"⚠️  BioBERT forward failed: {e}, using fallback")
        
        if self._encoder is not None:
            try:
                text = " ".join(str(t) for t in tokens[:512])
                embedding = self._encoder.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            except Exception as e:
                print(f"⚠️  SentenceTransformer forward failed: {e}, using fallback")
        
        # Fallback to base implementation
        return super().forward(tokens)


class VisionCLIPExpert(ImageExpert):
    """Real CLIP image encoder with PyTorch and CLIP.

    Falls back gracefully if CLIP is unavailable.
    """

    def __init__(self, name: str = "E_image_clip", model_name: str = "ViT-B/32") -> None:
        super().__init__(name=name)
        self.model_name = model_name
        self._model = None
        self._preprocess = None
        
        try:
            import clip  # type: ignore
            import torch  # type: ignore
            
            print(f"🔄 Loading CLIP {model_name} for image encoding...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model, self._preprocess = clip.load(model_name, device=device)
            print(f"✅ Loaded CLIP {model_name}")
        except ImportError:
            try:
                from transformers import CLIPProcessor, CLIPModel  # type: ignore
                import torch  # type: ignore
                
                print(f"🔄 Loading CLIP via transformers for image encoding...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self._model = self._model.to(device)
                self._preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                print(f"✅ Loaded CLIP via transformers")
            except ImportError:
                print("⚠️  No CLIP libraries found, using simple embeddings")

    def forward(self, image_tensor) -> List[float]:
        if self._model is not None and self._preprocess is not None:
            try:
                import torch  # type: ignore
                from PIL import Image  # type: ignore
                
                # Handle different input types
                if isinstance(image_tensor, torch.Tensor):
                    # Convert tensor to image
                    if image_tensor.dim() == 2:
                        image_tensor = image_tensor.unsqueeze(0)
                    img_array = image_tensor.cpu().numpy()
                    # Simple normalization and convert to PIL
                    img = Image.fromarray((img_array * 255).astype('uint8').squeeze() if len(img_array.shape) == 2 else img_array.squeeze())
                elif isinstance(image_tensor, list):
                    # If we get a list (our tokenized format), convert to PIL
                    if isinstance(image_tensor[0], list) and len(image_tensor[0]) == 1:
                        # Handle placeholder format [[0.0]]
                        # Create a small dummy image
                        img = Image.new('RGB', (224, 224), color='gray')
                    else:
                        # Try to create image from list
                        import numpy as np
                        arr = np.array(image_tensor).astype('float32')
                        if arr.max() <= 1.0:
                            arr = (arr * 255).astype('uint8')
                        img = Image.fromarray(arr.squeeze())
                else:
                    # Assume it's already a PIL Image or file path
                    img = image_tensor
                
                # Preprocess and encode
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                if hasattr(self._preprocess, '__call__') and 'torch' in str(type(self._preprocess)):
                    # CLIP's preprocess function
                    image_input = self._preprocess(img).unsqueeze(0).to(device)
                else:
                    # transformers CLIPProcessor
                    image_input = self._preprocess(images=img, return_tensors="pt")
                    image_input = {k: v.to(device) for k, v in image_input.items()}
                
                with torch.no_grad():
                    if isinstance(image_input, dict):
                        # transformers format
                        outputs = self._model.get_image_features(**image_input)
                    else:
                        # CLIP format
                        outputs = self._model.encode_image(image_input)
                    embedding = outputs.cpu().numpy().tolist()
                    return embedding[0]
            except Exception as e:
                print(f"⚠️  CLIP forward failed: {e}, using fallback")
        
        # Fallback to base implementation
        return super().forward(image_tensor)


class CrossModalTransformer(JointExpert):
    """Cross-modal transformer adapter for fusion (stubbed)."""

    def __init__(self, name: str = "E_joint_xattn", depth: int = 2) -> None:
        super().__init__(name=name)
        self.depth = depth

    def forward(self, text_repr: List[float], image_repr: List[float]) -> List[float]:
        # Simple gated mix emulating a few fusion layers
        n = min(len(text_repr), len(image_repr))
        out = []
        for i in range(n):
            alpha = 0.5 + 0.1 * math.sin(i)
            out.append(alpha * text_repr[i] + (1 - alpha) * image_repr[i])
        return out


# =====================
# Gating and routing
# =====================


def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [v / s for v in exps]


@dataclass
class RoutingResult:
    selected_text_idxs: List[int]
    selected_image_idxs: List[int]
    selected_joint_idxs: List[int]
    weights_text: List[float]
    weights_image: List[float]
    weights_joint: List[float]


class GatingNetwork:
    def __init__(self, num_text: int, num_image: int, num_joint: int) -> None:
        self.num_text = num_text
        self.num_image = num_image
        self.num_joint = num_joint

    def route(self, pooled_text: List[float], pooled_image: List[float], k_text: int, k_image: int, k_joint: int) -> RoutingResult:
        # Stub gating: compute simple scores via sums, add small randomness
        def scores(n: int, seed: int) -> List[float]:
            rnd = random.Random(seed)
            base = [rnd.random() for _ in range(n)]
            return base

        logits_t = scores(self.num_text, 11)
        logits_i = scores(self.num_image, 13)
        logits_j = scores(self.num_joint, 17)

        w_t = _softmax(logits_t)
        w_i = _softmax(logits_i)
        w_j = _softmax(logits_j)

        topk_t = sorted(range(self.num_text), key=lambda idx: w_t[idx], reverse=True)[: max(1, k_text)]
        topk_i = sorted(range(self.num_image), key=lambda idx: w_i[idx], reverse=True)[: max(1, k_image)]
        topk_j = sorted(range(self.num_joint), key=lambda idx: w_j[idx], reverse=True)[: max(1, k_joint)]

        return RoutingResult(
            selected_text_idxs=topk_t,
            selected_image_idxs=topk_i,
            selected_joint_idxs=topk_j,
            weights_text=w_t,
            weights_image=w_i,
            weights_joint=w_j,
        )


# =====================
# Routing visualization
# =====================


def _routing_svg(output_dir: str, family: str, weights: List[float], selected: List[int]) -> str:
    _ensure_dir(output_dir)
    width, height = 20 + 100 * len(weights), 140
    x = 10
    bar_w = 80
    gap = 20
    elements: List[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='#fff' stroke='#ddd'/>",
        f"<text x='10' y='20' font-family='Arial' font-size='12'>{family} routing</text>",
    ]
    for idx, w in enumerate(weights):
        h = int(100 * w)
        y = 120 - h
        fill = "#4a90e2" if idx in selected else "#bcd6f1"
        elements.append(f"<rect x='{x}' y='{y}' width='{bar_w}' height='{h}' fill='{fill}' rx='4'/>")
        elements.append(f"<text x='{x}' y='135' font-size='10' font-family='Arial'>e{idx}</text>")
        x += bar_w + gap
    elements.append("</svg>")
    fpath = os.path.join(output_dir, f"routing_{family}_{_now_ms()}.svg")
    _write_file(fpath, "\n".join(elements))
    return fpath


def export_routing_artifacts(output_dir: str, routing: RoutingResult) -> Dict[str, str]:
    svg_text = _routing_svg(output_dir, "text", routing.weights_text, routing.selected_text_idxs)
    svg_image = _routing_svg(output_dir, "image", routing.weights_image, routing.selected_image_idxs)
    svg_joint = _routing_svg(output_dir, "joint", routing.weights_joint, routing.selected_joint_idxs)
    sidecar = {
        "weights_text": routing.weights_text,
        "weights_image": routing.weights_image,
        "weights_joint": routing.weights_joint,
        "selected_text_idxs": routing.selected_text_idxs,
        "selected_image_idxs": routing.selected_image_idxs,
        "selected_joint_idxs": routing.selected_joint_idxs,
        "svgs": {"text": svg_text, "image": svg_image, "joint": svg_joint},
    }
    json_path = os.path.join(output_dir, f"routing_metrics_{_now_ms()}.json")
    _write_file(json_path, json.dumps(sidecar, ensure_ascii=False, indent=2))
    return {"text": svg_text, "image": svg_image, "joint": svg_joint, "metrics_json": json_path}


# =====================
# Diffusion adapter (stub)
# =====================


def generate_diagram(output_dir: str, prompt: str) -> str:
    _ensure_dir(output_dir)
    
    # Try diffusers Stable Diffusion with caching
    try:
        from diffusers import StableDiffusionPipeline  # type: ignore
        import torch  # type: ignore
        
        print(f"🖼️  Generating diagram with Stable Diffusion for: '{prompt[:50]}...'")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use smaller, faster model for demo
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,  # Disable for faster generation
                requires_safety_checker=False,
            )
        except Exception:
            # Fallback to even smaller model
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        
        pipe = pipe.to(device)
        
        # Generate with fewer steps for speed
        image = pipe(
            prompt,
            num_inference_steps=20,  # Reduced from default 50 for speed
            guidance_scale=7.5,
        ).images[0]
        
        path = os.path.join(output_dir, f"diagram_{_now_ms()}.png")
        image.save(path)
        print(f"✅ Diagram saved to {path}")
        return path
        
    except ImportError:
        print("⚠️  diffusers not installed, using SVG fallback")
    except Exception as e:
        print(f"⚠️  Stable Diffusion failed: {e}, using SVG fallback")
    
    # Fallback to SVG stub
    svg = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='640' height='400'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='#f9fafb' stroke='#e5e7eb'/>",
        f"<text x='20' y='40' font-family='Arial' font-size='16' fill='#111'>Diagram: {prompt}</text>",
        "<text x='20' y='80' font-family='Arial' font-size='12' fill='#666'>Enable diffusers for real image generation</text>",
        "</svg>",
    ]
    path = os.path.join(output_dir, f"diagram_{_now_ms()}.svg")
    _write_file(path, "\n".join(svg))
    return path


# =====================
# NeuroSeek-MoE model
# =====================


class NeuroSeekMoE:
    def __init__(
        self,
        text_experts: List[TextExpert],
        image_experts: List[ImageExpert],
        joint_experts: List[JointExpert],
        k_text: int = 1,
        k_image: int = 1,
        k_joint: int = 1,
        outputs_dir: str = "./outputs",
    ) -> None:
        self.text_experts = text_experts
        self.image_experts = image_experts
        self.joint_experts = joint_experts
        self.k_text = k_text
        self.k_image = k_image
        self.k_joint = k_joint
        self.outputs_dir = outputs_dir
        self.gate = GatingNetwork(len(text_experts), len(image_experts), len(joint_experts))

    def _pool(self, vec: List[float]) -> float:
        return sum(vec) / max(1, len(vec))

    def infer(
        self,
        text_tokens: List[int],
        image_tensor,
        disease: Optional[Disease] = None,
        generate_diagrams: bool = True,
    ) -> Dict[str, str]:
        # 1) Encode via all experts (in practice, would dispatch; we compute reprs to drive gating)
        text_reprs = [e.forward(text_tokens) for e in self.text_experts]
        image_reprs = [e.forward(image_tensor) for e in self.image_experts]
        pooled_text = [self._pool(r) for r in text_reprs]
        pooled_image = [self._pool(r) for r in image_reprs]

        # 2) Route (top-k per family)
        routing = self.gate.route(pooled_text, pooled_image, self.k_text, self.k_image, self.k_joint)

        # 3) Fuse with joint experts over selected indices
        fused_repr: Optional[List[float]] = None
        if routing.selected_joint_idxs:
            # Use highest-weighted selected text/image reprs as inputs to first joint expert
            t_idx = routing.selected_text_idxs[0]
            i_idx = routing.selected_image_idxs[0]
            j_idx = routing.selected_joint_idxs[0]
            fused_repr = self.joint_experts[j_idx].forward(text_reprs[t_idx], image_reprs[i_idx])

        # 4) Decode text (stub) and generate diagram (stub)
        prompt_pieces = ["neurodegenerative", disease.value if disease else "unknown", "fusion"]
        prompt = " ".join(prompt_pieces)
        text_answer = self._text_decode_stub(text_tokens, fused_repr)
        diagram_path = generate_diagram(self.outputs_dir, prompt) if generate_diagrams else ""

        # 5) Export routing artifacts
        viz_paths = export_routing_artifacts(self.outputs_dir, routing)

        return {
            "text": text_answer,
            "diagram_path": diagram_path,
            "routing_text_svg": viz_paths["text"],
            "routing_image_svg": viz_paths["image"],
            "routing_joint_svg": viz_paths["joint"],
            "routing_metrics_json": viz_paths["metrics_json"],
        }

    def _text_decode_stub(self, text_tokens: List[int], fused_repr: Optional[List[float]]) -> str:
        score = sum(fused_repr) / len(fused_repr) if fused_repr else 0.0
        return f"Structured explanation (stub). Fusion score={score:.3f}."

    def visualize_routing(self, routing: RoutingResult) -> Dict[str, str]:
        """Public helper to export routing artifacts for a captured RoutingResult."""
        return export_routing_artifacts(self.outputs_dir, routing)


class DeepSeekMoEWrapper(NeuroSeekMoE):
    """Wrapper that assembles modality-specific pretrained experts and fusion adapter.

    - E_text: BioLLMTextExpert (BioLlama/BioGPT placeholder)
    - E_image: VisionCLIPExpert (CLIP/BLIP placeholder)
    - E_fusion: CrossModalTransformer
    - Routing: top-k per family (inherited)
    - Outputs: text + optional image via Stable Diffusion adapter stub
    """

    @classmethod
    def build_default(cls, outputs_dir: str = "./outputs") -> "DeepSeekMoEWrapper":
        text_experts: List[TextExpert] = [
            BioLLMTextExpert(name="E_text_bio_0"),
            BioLLMTextExpert(name="E_text_bio_1"),
        ]
        image_experts: List[ImageExpert] = [
            VisionCLIPExpert(name="E_image_clip_0"),
            VisionCLIPExpert(name="E_image_clip_1"),
        ]
        joint_experts: List[JointExpert] = [
            CrossModalTransformer(name="E_joint_xattn_0"),
            CrossModalTransformer(name="E_joint_xattn_1"),
        ]
        return cls(text_experts, image_experts, joint_experts, k_text=1, k_image=1, k_joint=1, outputs_dir=outputs_dir)


# =====================
# Demo builder
# =====================


def build_demo_model(outputs_dir: str = "./outputs") -> NeuroSeekMoE:
    # 2 text, 2 image, 2 joint experts for demo (replace with real backends)
    text_experts = [TextExpert("E_text_0"), TextExpert("E_text_1")]
    image_experts = [ImageExpert("E_image_0"), ImageExpert("E_image_1")]
    joint_experts = [JointExpert("E_joint_0"), JointExpert("E_joint_1")]
    return NeuroSeekMoE(text_experts, image_experts, joint_experts, k_text=1, k_image=1, k_joint=1, outputs_dir=outputs_dir)


def demo_run() -> None:
    model = build_demo_model(outputs_dir="./outputs")
    # Toy tokens and image tensor placeholders
    text_tokens = [1, 2, 3, 4, 5]
    image_tensor = [[0.0]]
    out = model.infer(text_tokens, image_tensor, disease=Disease.ALZHEIMERS)
    print("Hybrid output (demo):")
    print(f"- Text: {out['text']}")
    print(f"- Diagram: {out['diagram_path']}")
    print(f"- Routing text SVG: {out['routing_text_svg']}")
    print(f"- Routing image SVG: {out['routing_image_svg']}")
    print(f"- Routing joint SVG: {out['routing_joint_svg']}")
    print(f"- Routing metrics JSON: {out['routing_metrics_json']}")


if __name__ == "__main__":
    demo_run()


