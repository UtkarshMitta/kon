import wandb
import os
from datetime import datetime

class WandbLogger:
    def __init__(self, project_name="mistral-router", entity=None):
        self.project_name = project_name
        self.entity = entity
        self.run = None
        self.rows = []
        self.columns = ["timestamp", "query", "predicted_tier", "corrected_tier", "confidence", "status"]
        
    def start_run(self, name=None, config=None):
        """Starts a new W&B run."""
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config
        )

    def log_metrics(self, metrics):
        """Logs scalar metrics."""
        if self.run:
            wandb.log(metrics)

    def log_feedback(self, query, predicted, corrected, confidence, status):
        """Adds a row to the local list and pushes a fresh table to W&B."""
        if self.run:
            self.rows.append([
                datetime.now().isoformat(),
                query,
                predicted,
                corrected,
                confidence,
                status
            ])
            
            # Create a FRESH table object to ensure W&B detects the change
            table = wandb.Table(columns=self.columns, data=self.rows)
            wandb.log({"human_feedback_training_signal": table})

    def finish(self):
        """Finishes the W&B run."""
        if self.run:
            self.run.finish()
