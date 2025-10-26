"""
model_loader.py (Modern Chat Model Version)
----------------------------------
Model and tokenizer loading module optimized for modern Causal LM
chat models like Qwen.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_chat_model(model_name: str = "Qwen/Qwen1.5-1.8B-Chat", device: str = None):
    """
    Load a Hugging Face text generation model and tokenizer.
    
    Args:
        model_name: Hugging Face model identifier
        device: Target device ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        Tuple of (model, tokenizer, device)
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # --- CRITICAL FIX for Chat models ---
        # Many chat models don't set a pad token by default.
        # We use the end-of-sentence (EOS) token as the padding token.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # --- END FIX ---
            
        # We must use AutoModelForCausalLM for chat models
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,  # Use FP16 for faster GPU inference
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,  # Use FP32 for CPU
            ).to(device)
            
        return model, tokenizer, device
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


if __name__ == "__main__":
    # Test model loading
    print("Testing Qwen 1.5 model loader...\n")
    model, tokenizer, device = load_chat_model("Qwen/Qwen1.5-1.8B-Chat")
    print(f"✓ Model loaded successfully on {device.upper()}")
    print(f"Model type: {type(model)}")
    print(f"PAD token: {tokenizer.pad_token}")