import json
import os
from mistralai import Mistral

class ContextManager:
    """
    Manages conversation context with an 'Elastic' approach:
    - Stable Tiers: Maintain raw history for accuracy.
    - Tier Shifts: Compress history into a 'Briefing' using Ministral 3B to save cost.
    """
    def __init__(self, api_key, scratchpad_path="scratchpad.json"):
        self.client = Mistral(api_key=api_key)
        self.scratchpad_path = scratchpad_path
        
        self.briefing = ""  # Cumulative summary
        self.pending_turns = []  # Raw turns since last pivot
        self.last_tier = None
        
        self._load_state()

    def _load_state(self):
        """Load state from disk."""
        if os.path.exists(self.scratchpad_path):
            try:
                with open(self.scratchpad_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.briefing = data.get("briefing", "")
                    self.pending_turns = data.get("pending_turns", [])
                    self.last_tier = data.get("last_tier")
            except:
                pass

    def save_state(self):
        """Save state to disk."""
        with open(self.scratchpad_path, 'w', encoding='utf-8') as f:
            json.dump({
                "briefing": self.briefing,
                "pending_turns": self.pending_turns,
                "last_tier": self.last_tier
            }, f, indent=4)

    def add_turn(self, query, response_text, tier):
        """Record a turn in the pending history."""
        self.pending_turns.append({"role": "user", "content": query})
        self.pending_turns.append({"role": "assistant", "content": response_text})
        self.last_tier = tier
        self.save_state()

    def should_pivot(self, new_tier):
        """Check if we need to compress context due to a tier change."""
        if self.last_tier is None:
            return False
        return new_tier != self.last_tier

    def generate_pivot_briefing(self):
        """
        Compresses everything we know into a new briefing and returns usage.
        """
        if not self.pending_turns and self.briefing:
            return self.briefing, {"input_tokens": 0, "output_tokens": 0}

        # Construct the summarization prompt
        history_str = "\n".join([f"{t['role']}: {t['content']}" for t in self.pending_turns])
        
        prompt = f"""Task: Summarize the ACTIVE conversation context for a new LLM.
- **Rule 1**: If the user has changed the topic (e.g. from code to math/history), output ONLY the word 'NONE'.
- **Rule 2**: If the topic is the same, output a 1-sentence summary of key facts.

OLD BRIEFING: {self.briefing if self.briefing else "None"}
NEW TURNS:
{history_str}
"""
        
        try:
            print("[Context] Generating Pivot Briefing via Ministral 3B...")
            response = self.client.chat.complete(
                model="ministral-3b-latest",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            new_briefing = response.choices[0].message.content.strip()
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
            
            # Check for topic shift
            if "NONE" in new_briefing.upper() and len(new_briefing) < 15:
                print("[Context] Topic shift detected. Clearing context briefing.")
                self.briefing = ""
            else:
                self.briefing = new_briefing

            self.pending_turns = [] # Wipe raw history after it's been summarized
            self.save_state()
            return self.briefing, usage
            
        except Exception as e:
            print(f"[!] Briefing error: {e}")
            return self.briefing, {"input_tokens": 0, "output_tokens": 0}

    def get_messages_for_api(self, current_query):
        """
        Returns the message list for the Mistral API.
        If we have a briefing, it starts with the briefing as a system injection.
        """
        messages = []
        
        if self.briefing:
            messages.append({
                "role": "system", 
                "content": (
                    "CONTEXT BRIEFING: Use the following information ONLY if it is directly relevant "
                    "to the user's current query. If the user has changed the subject, ignore this context.\n"
                    f"--- PRIOR CONTEXT ---\n{self.briefing}"
                )
            })
            
        # Add any pending turns (if any exist before a pivot)
        messages.extend(self.pending_turns)
        
        # Add the newest query
        messages.append({"role": "user", "content": current_query})
        
        return messages

    def clear_context(self):
        """Wipe state and file for a clean session."""
        print("[Context] Clearing session scratchpad...")
        self.briefing = ""
        self.pending_turns = []
        self.last_tier = None
        if os.path.exists(self.scratchpad_path):
            try:
                os.remove(self.scratchpad_path)
            except Exception as e:
                print(f"[!] Error clearing scratchpad: {e}")
