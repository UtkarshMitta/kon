"""
Precise cost calculation and ROI tracking for Mistral Router.
Supports per-tier input/output pricing and baseline comparison.
"""

class CostCalculator:
    # Pricing per Million tokens (USD)
    PRICING = {
        1: {"input": 0.1, "output": 0.1},  # Tier 1
        2: {"input": 0.1, "output": 0.3},  # Tier 2
        3: {"input": 0.4, "output": 2.0},  # Tier 3 (Baseline)
        4: {"input": 0.5, "output": 1.5},  # Tier 4
    }

    def __init__(self, baseline_tier=3):
        self.baseline_tier = baseline_tier
        self.reset()

    def reset(self):
        """Reset session metrics."""
        self.total_actual_cost = 0.0
        self.total_baseline_cost = 0.0
        self.routing_costs = 0.0
        self.summarization_costs = 0.0
        self.model_costs = 0.0
        self.turns = 0

    def calculate_cost(self, tier, input_tokens, output_tokens):
        """Calculate cost for a given tier and token counts."""
        if tier not in self.PRICING:
            return 0.0
        
        rates = self.PRICING[tier]
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]
        return input_cost + output_cost

    def record_turn(self, actual_tier, input_tokens, output_tokens, is_summarization=False, override_baseline_cost=None):
        """
        Record a turn's costs.
        
        Args:
            actual_tier: The tier used for the model response or summarization.
            input_tokens: Total input tokens.
            output_tokens: Total output tokens.
            is_summarization: Whether this was a context management call.
            override_baseline_cost: If provided, use this actual cost instead of calculating.
        """
        # Calculate actual cost
        actual_cost = self.calculate_cost(actual_tier, input_tokens, output_tokens)
        
        # Calculate or use provided baseline cost
        if override_baseline_cost is not None:
            baseline_cost = override_baseline_cost
        else:
            baseline_cost = self.calculate_cost(self.baseline_tier, input_tokens, output_tokens)

        # Increment session totals
        self.total_actual_cost += actual_cost
        self.total_baseline_cost += baseline_cost
        
        if is_summarization:
            self.summarization_costs += actual_cost
        else:
            self.model_costs += actual_cost
            self.turns += 1

        return {
            "actual_cost": actual_cost,
            "baseline_cost": baseline_cost,
            "savings": baseline_cost - actual_cost
        }

    def get_session_summary(self):
        """Return a formatted summary of session ROI."""
        savings = self.total_baseline_cost - self.total_actual_cost
        savings_pct = (savings / self.total_baseline_cost * 100) if self.total_baseline_cost > 0 else 0
        
        return {
            "total_turns": self.turns,
            "actual_total_cost": self.total_actual_cost,
            "baseline_total_cost": self.total_baseline_cost,
            "summarization_overhead": self.summarization_costs,
            "total_savings": savings,
            "roi_percentage": savings_pct
        }

    def print_turn_report(self, turn_data):
        """Print a quick turn-level ROI update."""
        savings = turn_data['savings']
        icon = "✅" if savings > 0 else "📊"
        print(f"   {icon} Turn Cost: ${turn_data['actual_cost']:.6f} | Baseline: ${turn_data['baseline_cost']:.6f} | Savings: ${savings:.6f}")

    def print_final_report(self):
        """Print a detailed session-end ROI analysis."""
        summary = self.get_session_summary()
        print("\n" + "="*50)
        print("💰 PRECISION COST & ROI ANALYSIS")
        print("="*50)
        print(f"Total Session Turns:    {summary['total_turns']}")
        print(f"Actual Total Cost:      ${summary['actual_total_cost']:.6f}")
        print(f"  └─ Model Responses:   ${self.model_costs:.6f}")
        print(f"  └─ Summarization:     ${summary['summarization_overhead']:.6f}")
        print("-" * 30)
        print(f"Baseline Cost (Tier 3): ${summary['baseline_total_cost']:.6f}")
        print(f"Total Session Savings:  ${summary['total_savings']:.6f}")
        
        color = "\033[92m" if summary['roi_percentage'] > 0 else "\033[91m"
        reset = "\033[0m"
        print(f"Efficiency ROI:         {color}{summary['roi_percentage']:.1f}%{reset}")
        print("="*50 + "\n")
