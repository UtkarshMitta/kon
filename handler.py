import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# Configuration
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER = "tans37/mistral-query-router"

class EndpointHandler():
    def __init__(self, path=""):
        # Load the base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        # Load base model in half precision for efficiency
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load the Peft adapter
        # 'path' is the directory where the handler is located (the adapter repo)
        self.model = PeftModel.from_pretrained(base_model, path)
        self.model.eval()
        
        print(f"[Handler] Loaded LoRA adapter from {path} onto {BASE_MODEL}")

    def __call__(self, data):
        """
        Args:
            data (:obj: `dict`):
                subset of the request body with the following keys:
                - `inputs`: the prompt to be processed
                - `parameters`: optional generation parameters
        """
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", {
            "max_new_tokens": 128,
            "temperature": 0.1,
            "top_p": 0.9,
            "do_sample": False
        })

        # Tokenize
        inputs = self.tokenizer(inputs, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                **parameters
            )

        # Decode
        # We only want the new tokens, so we slice the output
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_tokens[0][input_len:]
        prediction = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return [{"generated_text": prediction}]
