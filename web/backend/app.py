import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference.router import MistralRouter
from inference.context_manager import ContextManager
from inference.cost_calculator import CostCalculator
from inference.wandb_logger import WandbLogger
from config import TIER_MODEL_MAP

app = FastAPI(title="Mistral Router Dashboard API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (Reset per session if needed)
router = None
context = None
roi_tracker = CostCalculator()
wb = None

class QueryRequest(BaseModel):
    query: str
    compare: bool = False

class FeedbackRequest(BaseModel):
    query: str
    original_tier: int
    corrected_tier: int
    confidence: float

@app.on_event("startup")
async def startup_event():
    global router, context, wb
    
    mistral_key = os.getenv("MISTRAL_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    hf_endpoint = os.getenv("HF_ENDPOINT_URL")
    
    router = MistralRouter(endpoint_url=hf_endpoint, hf_token=hf_token, mistral_api_key=mistral_key)
    context = ContextManager(api_key=mistral_key)
    
    wb = WandbLogger()
    wb.start_run(config={
        "mode": "web_dashboard",
        "routing_strategy": "SFT-Router"
    })

@app.post("/api/query")
async def process_query(req: QueryRequest):
    global router, context, roi_tracker, wb
    
    try:
        # A. Route
        result = router.route(req.query)
        tier = result["model_tier"]
        model_name = result["model_name"]
        confidence = result["confidence"]
        
        # B. Context Pivot
        pivot_data = None
        if context.should_pivot(tier):
            briefing, pivot_usage = context.generate_pivot_briefing()
            roi_tracker.record_turn(1, pivot_usage["input_tokens"], pivot_usage["output_tokens"], is_summarization=True)
            pivot_data = {"briefing": briefing, "usage": pivot_usage}
        
        # C. Inference
        messages = context.get_messages_for_api(req.query)
        response_text, usage = router.get_model_response(model_name, messages)
        
        # D. Battle Mode (Optional)
        baseline_cost_override = None
        if req.compare:
            baseline_model = TIER_MODEL_MAP[3]
            _, baseline_usage = router.get_model_response(baseline_model, messages)
            baseline_cost_override = roi_tracker.calculate_cost(3, baseline_usage["input_tokens"], baseline_usage["output_tokens"])
            
        # E. Record Costs
        turn_roi = roi_tracker.record_turn(tier, usage["input_tokens"], usage["output_tokens"], override_baseline_cost=baseline_cost_override)
        
        # F. Update Memory
        context.add_turn(req.query, response_text, tier)
        
        # G. Log (Async/Background-ish)
        wb.log_feedback(req.query, tier, tier, confidence, "web_inference")
        
        return {
            "response": response_text,
            "routing": result,
            "usage": usage,
            "roi": turn_roi,
            "pivot": pivot_data,
            "context_briefing": context.briefing,
            "session_roi": roi_tracker.get_session_summary()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    global wb
    try:
        status = "confirmed" if req.original_tier == req.corrected_tier else "corrected"
        wb.log_feedback(req.query, req.original_tier, req.corrected_tier, req.confidence, status)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    return roi_tracker.get_session_summary()

@app.post("/api/reset")
async def reset_session():
    global roi_tracker
    context.clear_context()
    roi_tracker.reset()
    return {"status": "reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
