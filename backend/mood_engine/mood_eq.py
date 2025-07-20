import time
import json
import os
from typing import Dict, List, Optional, Tuple

class MoodEQ:
    EQ_BANDS = [
        "joy", "focus", "anger", "sad", "cold", "support",
        "sarcasm", "anxiety", "fatigue", "romantic", "suspicion", "inspiration"
    ]

    def __init__(self,
                 initial_state: Optional[Dict[str, float]] = None,
                 default_decay_rate: float = 0.05,
                 decay_interval_seconds: int = 120,
                 bias_map: Optional[Dict[str, float]] = None,
                 decay_profiles: Optional[Dict[str, float]] = None,
                 mood_drift_rules: Optional[List[Dict]] = None,
                 max_history_turns: int = 10,
                 state_file_path: Optional[str] = None):
        
        self.state_file_path = state_file_path
        self._mood_vector: Dict[str, float] = {band: 0.0 for band in self.EQ_BANDS}
        self._conversation_history: List[Tuple[str, str]] = []
        self._last_updated: float = time.time()

        self.default_decay_rate: float = default_decay_rate
        self.decay_interval_seconds: int = decay_interval_seconds
        self.bias_map: Dict[str, float] = bias_map if bias_map is not None else {}
        self.decay_profiles: Dict[str, float] = decay_profiles if decay_profiles is not None else {}
        self.mood_drift_rules: List[Dict] = mood_drift_rules if mood_drift_rules is not None else []
        self.max_history_turns = max_history_turns

        # --- CORRECTED INITIALIZATION LOGIC ---
        if not (self.state_file_path and self._load_state_from_file()):
            print(f"[MoodEQ] No state file found or error loading. Applying initial state.")
            if initial_state:
                for band, value in initial_state.items():
                    if band in self._mood_vector:
                        self._mood_vector[band] = value
                self._clamp_values()
        else:
            print(f"[MoodEQ] State loaded successfully from {self.state_file_path}")
        
        print(f"[MoodEQ] Initial history length: {len(self._conversation_history)}")
        print(f"[MoodEQ] Initialized. Current state: {self.get_current_vector()}")

    def _clamp_values(self):
        for band in self.EQ_BANDS:
            self._mood_vector[band] = max(0.0, min(1.0, self._mood_vector[band]))

    def update_mood(self, incoming_mood_delta: Dict[str, float]):
        for band, value in incoming_mood_delta.items():
            if band in self._mood_vector:
                adjusted_value = value * self.bias_map.get(band, 1.0)
                self._mood_vector[band] += adjusted_value
        self._clamp_values()
        self._last_updated = time.time()
        self._save_state_to_file()

    def apply_decay(self):
        current_time = time.time()
        if (current_time - self._last_updated) >= self.decay_interval_seconds:
            for band in self.EQ_BANDS:
                decay_amount = self.decay_profiles.get(band, self.default_decay_rate)
                self._mood_vector[band] -= decay_amount
            self._clamp_values()
            self._last_updated = current_time
            self._save_state_to_file()

    def drift_mood(self):
        drift_applied = False
        current_moods = self.get_current_vector()
        for rule in self.mood_drift_rules:
            if_mood, above_threshold = rule.get('if_mood'), rule.get('above_threshold')
            if if_mood and if_mood in current_moods and current_moods[if_mood] >= above_threshold:
                drift_applied = True
                for target_band, delta in rule.get('then_increase', {}).items():
                    if target_band in self._mood_vector: self._mood_vector[target_band] += delta
                for target_band, delta in rule.get('then_reduce', {}).items():
                    if target_band in self._mood_vector: self._mood_vector[target_band] -= delta
        if drift_applied:
            self._clamp_values()
            self._last_updated = time.time()
            self._save_state_to_file()

    def get_current_vector(self) -> Dict[str, float]:
        return self._mood_vector.copy()

    def get_dominant_moods(self, threshold: float = 0.2) -> List[str]:
        dominant = sorted(
            [(band, value) for band, value in self._mood_vector.items() if value >= threshold],
            key=lambda x: x[1], reverse=True
        )
        return [band for band, _ in dominant]

    def get_soul_state_object(self) -> Dict:
        return {
            "mood_vector": self.get_current_vector(),
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._last_updated)),
            "dominant_moods": self.get_dominant_moods()
        }

    def get_conversation_history(self) -> List[Tuple[str, str]]:
        return self._conversation_history.copy()

    def add_to_conversation_history(self, role: str, message: str):
        self._conversation_history.append((role, message))
        if len(self._conversation_history) > self.max_history_turns:
            self._conversation_history = self._conversation_history[-self.max_history_turns:]
        self._save_state_to_file()

    def _save_state_to_file(self):
        if not self.state_file_path: return
        session_data = {
            "mood_vector": self._mood_vector,
            "conversation_history": self._conversation_history,
            "last_updated_timestamp": self._last_updated
        }
        try:
            with open(self.state_file_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2)
        except IOError as e:
            print(f"[MoodEQ] Error saving state: {e}")

    def _load_state_from_file(self) -> bool:
        if not self.state_file_path: return False
        try:
            with open(self.state_file_path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
                self._mood_vector = loaded_data.get("mood_vector", self._mood_vector)
                self._conversation_history = [tuple(item) for item in loaded_data.get("conversation_history", [])]
                self._last_updated = loaded_data.get("last_updated_timestamp", time.time())
                self._clamp_values()
                return True
        except (FileNotFoundError, json.JSONDecodeError, IOError):
            return False