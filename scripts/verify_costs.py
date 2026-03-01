import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.cost_calculator import CostCalculator

def test_cost_logic():
    print("Testing Precision Cost Calculator...")
    calc = CostCalculator(baseline_tier=3)

    # 1. Test Tier 1 Turn
    # Rate: 0.1/M In, 0.1/M Out
    # 500k in, 500k out = 1M total = $0.1
    turn1 = calc.record_turn(actual_tier=1, input_tokens=500_000, output_tokens=500_000)
    
    # Baseline (Tier 3): 0.4/M In, 2.0/M Out
    # 500k * 0.4 = 0.2
    # 500k * 2.0 = 1.0
    # Total = 1.2
    
    print(f"Tier 1 (Actual): ${turn1['actual_cost']:.2f} (Expected: $0.10)")
    print(f"Tier 1 (Baseline): ${turn1['baseline_cost']:.2f} (Expected: $1.20)")
    assert turn1['actual_cost'] == 0.10
    assert turn1['baseline_cost'] == 1.20

    # 2. Test Summarization Overhead (Tier 1)
    # 100k in, 100k out = $0.02
    calc.record_turn(actual_tier=1, input_tokens=100_000, output_tokens=100_000, is_summarization=True)
    
    # 3. Test Tier 2 Turn
    # Rate: 0.1/M In, 0.3/M Out
    # 1M in, 1M out = $0.1 + $0.3 = $0.4
    turn2 = calc.record_turn(actual_tier=2, input_tokens=1_000_000, output_tokens=1_000_000)
    print(f"Tier 2 (Actual): ${turn2['actual_cost']:.2f} (Expected: $0.40)")
    assert turn2['actual_cost'] == 0.40

    # 4. Final Summary
    summary = calc.get_session_summary()
    print("\nSummary Validation:")
    print(f"Actual Cost: ${summary['actual_total_cost']:.2f} (Expected: $0.52)") # 0.1 + 0.02 + 0.4
    print(f"Baseline Cost: ${summary['baseline_total_cost']:.2f} (Expected: $3.84)") # 1.2 + 0.24 (smrz) + 2.4 (t3 rates for t2 tokens)
    # Baseline for smrz: 100k*0.4=0.04, 100k*2.0=0.2. Total=0.24.
    # Baseline for t2 turn: 1M*0.4=0.4, 1M*2.0=2.0. Total=2.4.
    # Total Baseline = 1.2 + 0.24 + 2.4 = 3.84.
    
    assert round(summary['actual_total_cost'], 2) == 0.52
    assert round(summary['baseline_total_cost'], 2) == 3.84
    print(f"ROI percentage: {summary['roi_percentage']:.2f}%")
    
    print("\nAll cost verification tests passed!")

if __name__ == "__main__":
    test_cost_logic()
