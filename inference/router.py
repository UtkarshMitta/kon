"""
Inference wrapper for the Mistral Query Router.
Loads the fine-tuned Mistral 7B + LoRA adapter and routes queries to model tiers.
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import Optional
import os
import requests
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SYSTEM_PROMPT, TIER_MAP, TIER_MODEL_MAP, BASE_MODEL


class MistralRouter:
    """
    Routes user queries to the optimal Mistral model tier.

    Usage:
        router = MistralRouter("your-hf-username/mistral-query-router")
        result = router.route("What is the square root of 4?")
        # → {"model_tier": 1, "confidence": 0.97, "model_name": "ministral-3b-latest"}
    """

    def __init__(
        self,
        model_path: str = None,
        base_model: str = BASE_MODEL,
        device: str = "auto",
        load_in_4bit: bool = True,
        mistral_api_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        hf_token: Optional[str] = None
    ):
        """
        Initialize the router.

        Args:
            model_path: Path to fine-tuned LoRA adapter (local or HF Hub repo)
            base_model: Base model name (for loading with adapter)
            device: Device to load model on ("auto", "cpu", "cuda")
            load_in_4bit: Whether to use 4-bit quantization for inference
            mistral_api_key: API key for Mistral Platform execution
            endpoint_url: Custom Hugging Face Inference Endpoint URL
            hf_token: Hugging Face API Token
        """
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self._device = device
        self._load_in_4bit = load_in_4bit
        
        # Priority: Constructor Args > Environment Variables
        self.endpoint_url = endpoint_url or os.getenv("HF_ENDPOINT_URL")
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        
        self.use_api = bool(self.endpoint_url and self.hf_token)
        
        if self.use_api:
            print(f"[API] Configured to use Dedicated Hugging Face Endpoint: {self.endpoint_url}")
            
        # Initialize Mistral client if key is available
        self.client = Mistral(api_key=self.mistral_api_key) if self.mistral_api_key else None

    def _load_model(self):
        """Lazy-load the model and tokenizer."""
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        print(f"[*] Loading base model: {self.base_model}")

        # Quantization config for inference
        if self._load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        model_kwargs = {
            "device_map": self._device,
            "trust_remote_code": True,
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model, **model_kwargs
        )

        # Load LoRA adapter
        print(f"[*] Loading LoRA adapter: {self.model_path}")
        self.model = PeftModel.from_pretrained(self.model, self.model_path)
        self.model.eval()

        print("[+] Model loaded!")

    def route(self, query: str) -> dict:
        """
        Route a query to the appropriate model tier.

        Args:
            query: The user's query string

        Returns:
            dict with keys:
                - model_tier: int (1-4)
                - confidence: float (0.0-1.0)
                - model_name: str (Mistral model API name)
                - tier_label: str (small/medium/large/xlarge)
        """
        # If API mode is enabled, route through the Dedicated Hugging Face Endpoint
        if getattr(self, "use_api", False):
            return self._route_via_api(query)
            
        self._load_model()

        # Build chat messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        # Tokenize
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        # Generate
        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Parse the JSON response
        result = self._parse_response(response)
        return result

    def _route_via_api(self, query: str) -> dict:
        """Route a query using the Dedicated Hugging Face Inference Endpoint (Custom Handler)."""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        
        # Custom handlers on HF Dedicated Endpoints use a direct POST to the root URL
        endpoint = self.endpoint_url.rstrip("/")
        
        # We MUST use the Mistral Chat Template format (<s>[INST] ... [/INST])
        # for the custom handler to correctly engage the model.
        templated_prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\nUser Query: {query}\n\nRemember: Output ONLY the JSON object. Example: {{\"model_tier\": 2, \"confidence\": 0.8}} [/INST]"
        
        payload = {
            "inputs": templated_prompt,
            "parameters": {
                "max_new_tokens": 128,
                "temperature": 0.01, # Keep it extremely cold for routing
                "do_sample": False
            }
        }
        
        try:
            print(f"[API] Sending query to Custom LoRA Dedicated Endpoint...")
            response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result_json = response.json()
            # Custom handlers return a list of dictionaries by default on HF
            if isinstance(result_json, list) and len(result_json) > 0:
                generated_text = result_json[0].get("generated_text", "")
                return self._parse_response(generated_text.strip())
            else:
                print(f"[!] Unexpected Custom API response format: {result_json}")
                return self._validate_and_enrich({"model_tier": 2, "confidence": 0.3})
                
        except Exception as e:
            print(f"[!] API Routing failed: {e}")
            # Fallback to a moderate tier if the API is down
            return self._validate_and_enrich({"model_tier": 2, "confidence": 0.0})
            print(f"[ERROR] Dedicated API Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response: {e.response.text}")
            print("   Falling back to default tier 2.")
            return self._validate_and_enrich({"model_tier": 2, "confidence": 0.3})

    def route_batch(self, queries: list[str]) -> list[dict]:
        """Route multiple queries at once."""
        return [self.route(q) for q in queries]

    def get_model_response(self, model_name: str, messages: list[dict]) -> tuple[str, dict]:
        """
        Fetch the actual response from the Mistral API and return content + usage.
        """
        if not self.client:
            return "Error: No MISTRAL_API_KEY", {"input_tokens": 0, "output_tokens": 0}
            
        try:
            print(f"[Mistral API] Calling model: {model_name}...")
            response = self.client.chat.complete(
                model=model_name,
                messages=messages,
                max_tokens=5000
            )
            content = response.choices[0].message.content
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
            return content, usage
        except Exception as e:
            print(f"[ERROR] Mistral API Error: {e}")
            return f"Error fetching response: {str(e)}", {"input_tokens": 0, "output_tokens": 0}

    def load_adapter(self, adapter_name: str, adapter_path: Optional[str] = None) -> bool:
        """
        Dynamically load a LoRA adapter into the vLLM server.
        Returns True if successful.
        """
        if not self.use_api:
            return False
            
        endpoint = self.endpoint_url.rstrip("/") + "/v1/load_lora_adapter"
        payload = {
            "lora_name": adapter_name,
            "lora_path": adapter_path or adapter_name
        }
        
        try:
            print(f"[vLLM] Attempting to dynamic load adapter: {adapter_name}...")
            res = requests.post(
                endpoint, 
                headers={"Authorization": f"Bearer {self.hf_token}"},
                json=payload
            )
            if res.status_code == 200:
                print(f"[vLLM] Successfully loaded adapter: {adapter_name}")
                return True
            else:
                print(f"[vLLM] Load failed ({res.status_code}): {res.text}")
                return False
        except Exception as e:
            print(f"[vLLM] Error loading adapter: {e}")
            return False

    def _parse_response(self, response: str) -> dict:
        """Parse the model's JSON response with fallback."""
        # Try direct JSON parse
        try:
            data = json.loads(response)
            return self._validate_and_enrich(data)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._validate_and_enrich(data)
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract tier number
        tier_match = re.search(r'(?:model_tier|tier|Tier|category|level)["\s\-_:]+(\d)', response)
        if tier_match:
            tier = int(tier_match.group(1))
            # Also try to find a confidence score
            conf_match = re.search(r'(?:confidence|Confidence|score|probability)["\s\-_:]+(0?\.\d+)', response)
            confidence = float(conf_match.group(1)) if conf_match else 0.5
            
            return self._validate_and_enrich({
                "model_tier": tier,
                "confidence": confidence,
            })

        # Ultimate fallback: tier 2 (medium) with low confidence
        print(f"[!] Could not parse response: {response}")
        return self._validate_and_enrich({
            "model_tier": 2,
            "confidence": 0.3,
        })

    def _validate_and_enrich(self, data: dict) -> dict:
        """Validate tier/confidence values and add model name."""
        tier = data.get("model_tier", 2)
        confidence = data.get("confidence", 0.5)

        # Clamp values
        tier = max(1, min(4, int(tier)))
        confidence = max(0.0, min(1.0, float(confidence)))

        # Map tier to label and model name
        reverse_map = {v: k for k, v in TIER_MAP.items()}
        tier_label = reverse_map.get(tier, "medium")
        model_name = TIER_MODEL_MAP.get(tier, "mistral-small-latest")

        return {
            "model_tier": tier,
            "confidence": confidence,
            "tier_label": tier_label,
            "model_name": model_name,
        }


def main():
    parser = argparse.ArgumentParser(description="Mistral Query Router Inference")
    parser.add_argument("--model-path", type=str, required=True, help="Fine-tuned model path or HF repo")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL, help="Base model name")
    parser.add_argument("--query", type=str, help="Single query to route")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--test", action="store_true", help="Run test queries")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    router = MistralRouter(
        model_path=args.model_path,
        base_model=args.base_model,
        load_in_4bit=not args.no_4bit,
    )

    if args.test:
        test_queries = [
            "What is 2+2?",
            "Summarize the plot of The Great Gatsby.",
            "Debug this Python code that has a race condition in thread synchronization.",
            "Design a distributed consensus algorithm for a geo-replicated database.",
            "Hello!",
            "Translate 'hello' to French.",
            "Compare REST vs GraphQL APIs in detail.",
            "Prove the Fundamental Theorem of Algebra.",
        ]
        print("🧪 Running test queries...\n")
        for q in test_queries:
            result = router.route(q)
            print(f"  Query: {q[:60]}...")
            print(f"  → Tier {result['model_tier']} ({result['tier_label']}) "
                  f"| Model: {result['model_name']}")
            print()
        return

    if args.query:
        result = router.route(args.query)
        print(json.dumps(result, indent=2))
        return

    if args.interactive:
        print("--- Mistral Query Router — Interactive Mode ---")
        print("   Type a query and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                query = input("Query: ").strip()
                if query.lower() in ("quit", "exit", "q"):
                    break
                if not query:
                    continue
                result = router.route(query)
                print(f"  → Tier {result['model_tier']} ({result['tier_label']}) "
                      f"| Model: {result['model_name']}\n")
            except KeyboardInterrupt:
                break
        print("Goodbye!")


if __name__ == "__main__":
    main()
