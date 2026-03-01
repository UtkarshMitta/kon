import sys
import io
import argparse
from config import TIER_MODEL_MAP

# Fix for Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from CSVWrapper import log_metrics_to_csv
import time
import os
from inference.router import MistralRouter
from inference.context_manager import ContextManager
from inference.wandb_logger import WandbLogger
from feedback.collect import FeedbackCollector
from inference.cost_calculator import CostCalculator

class AgentMetricsTracker:
    def __init__(self, metrics_file="agent_metrics.csv", wandb_logger=None):
        self.metrics_file = metrics_file
        self.turn_count = 0
        self.cumulative_cost = 0.0
        self.wb = wandb_logger

    def record_turn(self, model_used, tokens_used, input_complexity, cost, metrics_dict=None):
        """
        Record a single turn's metrics and save to CSV + W&B.
        """
        self.turn_count += 1
        self.cumulative_cost += cost
        
        turn_metrics = {
            "turn": self.turn_count,
            "model_used": model_used,
            "tokens_used": tokens_used,
            "input_complexity": input_complexity,
            "cumulative_cost": self.cumulative_cost
        }
        if metrics_dict:
            turn_metrics.update(metrics_dict)
        
        log_metrics_to_csv(self.metrics_file, turn_metrics)
        if self.wb:
            self.wb.log_metrics(turn_metrics)
            
        print(f"Recorded turn {self.turn_count}: {model_used} ({tokens_used} tokens, cost: ${cost:.6f})")

def main():
    parser = argparse.ArgumentParser(description="Mistral Adaptive Query Router Agent")
    parser.add_argument("--compare", action="store_true", help="Battle Mode: Call baseline model for cost comparison (HIDDEN)")
    args = parser.parse_args()

    print("Initializing Adaptive Agent and Mistral Router...")
    
    # 1. Initialize W&B and Feedback
    wb = WandbLogger()
    wb.start_run(config={
        "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
        "routing_strategy": "SFT-Router",
        "rl_feedback_enabled": True,
        "battle_mode": args.compare
    })
    
    tracker = AgentMetricsTracker(wandb_logger=wb)
    collector = FeedbackCollector()
    roi_tracker = CostCalculator()
    
    # 2. Setup Router and Context
    mistral_key = os.getenv("MISTRAL_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    hf_endpoint = os.getenv("HF_ENDPOINT_URL")
    
    router = MistralRouter(endpoint_url=hf_endpoint, hf_token=hf_token, mistral_api_key=mistral_key)
    context = ContextManager(api_key=mistral_key)
    
    print("\n--- Interactive Query Router Started ---")
    print("RL Feedback Loop & Precision ROI Active.")
    
    try:
        while True:
            q = input("\nQuery: ").strip()
            if q.lower() in ('quit', 'exit', 'q'):
                break
            if not q:
                continue
            
            # A. Route
            result = router.route(q)
            tier = result["model_tier"]
            model_name = result["model_name"]
            confidence = result["confidence"]
            complexity = result["tier_label"]
            
            # B. Context Management
            if context.should_pivot(tier):
                print(f"[Pivot] Tier shifted from {context.last_tier} to {tier}. Compressing context...")
                briefing, pivot_usage = context.generate_pivot_briefing()
                roi_tracker.record_turn(1, pivot_usage["input_tokens"], pivot_usage["output_tokens"], is_summarization=True)
            
            messages = context.get_messages_for_api(q)
            
            # C. Inference
            response_text, usage = router.get_model_response(model_name, messages)
            
            # --- Battle Mode: Call Baseline if needed ---
            baseline_cost_override = None
            if args.compare:
                baseline_model = TIER_MODEL_MAP[3]
                # We call the baseline model but don't show the response to the user
                _, baseline_usage = router.get_model_response(baseline_model, messages)
                baseline_cost_override = roi_tracker.calculate_cost(3, baseline_usage["input_tokens"], baseline_usage["output_tokens"])
            # --------------------------------------------

            # Record regular model turn
            turn_roi = roi_tracker.record_turn(tier, usage["input_tokens"], usage["output_tokens"], override_baseline_cost=baseline_cost_override)
            
            print(f"Routed to: Tier {tier} ({model_name})")
            print(f"\n[Mistral Response]:\n{response_text}\n")
            roi_tracker.print_turn_report(turn_roi)
            
            # D. Feedback Loop (RL Signal)
            feedback = input(f"Was Tier {tier} correct? (y/n or correct tier 1-4, press Enter for skip): ").strip().lower()
            
            corrected_tier = tier
            status = "correct"
            
            if feedback == 'n' or feedback in ['1','2','3','4']:
                if feedback in ['1','2','3','4']:
                    corrected_tier = int(feedback)
                else:
                    try:
                        corrected_tier = int(input("What was the correct tier? (1-4): ").strip())
                    except:
                        corrected_tier = tier # skip if error
                
                status = "corrected" if corrected_tier != tier else "confirmed"
                collector.add(q, tier, corrected_tier, confidence=confidence)
            elif feedback == 'y':
                status = "confirmed"
            else:
                status = "skipped"

            # E. Logging (W&B Table for RL)
            wb.log_feedback(q, tier, corrected_tier, confidence, status)
            
            # F. Metrics
            tokens_total = usage["input_tokens"] + usage["output_tokens"]
            tracker.record_turn(model_name, tokens_total, complexity, turn_roi["actual_cost"], {"confidence": confidence, "status": status})
            
            # G. Update Memory
            context.add_turn(q, response_text, tier)
            
    except KeyboardInterrupt:
        pass
    finally:
        wb.finish()
        roi_tracker.print_final_report()
        context.clear_context()
        print(f"\nSession ended. Metrics and Feedback synced to W&B.")

if __name__ == "__main__":
    main()
