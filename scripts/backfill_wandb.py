import json
import os
import sys
import csv
import pandas as pd
import wandb
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import WANDB_PROJECT

def backfill():
    print("Starting W&B Backfill...")
    
    feedback_file = "feedback/store/feedback.jsonl"
    metrics_file = "agent_metrics.csv"
    scratchpad_file = "scratchpad.json"
    
    if not os.path.exists(feedback_file):
        print("No feedback file found.")
        return

    # 1. Setup Data Table
    columns = ["timestamp", "query", "predicted_tier", "corrected_tier", "confidence", "status"]
    data = []

    # 2. Load Corrections (with Query Text)
    corrections = []
    with open(feedback_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corrections.append(json.loads(line))
    
    print(f"Found {len(corrections)} local corrections.")
    for entry in corrections:
        status = "corrected" if not entry.get("is_correct", True) else "confirmed"
        data.append([
            entry.get("timestamp"),
            entry.get("query"),
            entry.get("predicted_tier"),
            entry.get("corrected_tier"),
            entry.get("confidence"),
            status
        ])

    # 3. Try to recover most recent turns from scratchpad
    if os.path.exists(scratchpad_file):
        try:
            with open(scratchpad_file, "r", encoding="utf-8") as f:
                scratch = json.load(f)
                pending = scratch.get("pending_turns", [])
                count = 0
                for i in range(0, len(pending), 2):
                    user_q = pending[i]["content"]
                    # Avoid duplicates
                    if not any(row[1] == user_q for row in data):
                        data.append([
                            datetime.now().isoformat(),
                            user_q,
                            scratch.get("last_tier", "?"),
                            scratch.get("last_tier", "?"),
                            1.0,
                            "recovered_from_scratchpad"
                        ])
                        count += 1
            print(f"Recovered {count} recent turns from scratchpad.")
        except Exception as e:
            print(f"Could not read scratchpad: {e}")

    # 4. Setup W&B
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"recovered_history_{datetime.now().strftime('%H%M%S')}",
        config={"type": "recovery"}
    )
    
    # 5. Push Feedback Table
    table = wandb.Table(columns=columns, data=data)
    wandb.log({"human_feedback_training_signal": table})
    print(f"Synced {len(data)} rows to the feedback table.")
    
    # 6. Push Full Telemetry from CSV (even if headers are inconsistent)
    if os.path.exists(metrics_file):
        try:
            all_rows = []
            with open(metrics_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader)
                for row in reader:
                    # Pad to 7 columns
                    if len(row) < 7:
                        row.extend([""] * (7 - len(row)))
                    all_rows.append(row[:7])
            
            latest_11 = all_rows[-11:]
            cols = ["turn", "model_used", "tokens_used", "input_complexity", "cumulative_cost", "confidence", "status"]
            telemetry_table = wandb.Table(columns=cols, data=latest_11)
            wandb.log({"telemetry_history": telemetry_table})
            print(f"Pushed {len(latest_11)} telemetry rows from CSV.")
        except Exception as e:
            print(f"Could not read metrics CSV: {e}")

    wandb.finish()
    print("Backfill complete!")

if __name__ == "__main__":
    backfill()
