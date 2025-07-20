import ollama
import json
import random
import time
import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from mood_engine.mood_eq import MoodEQ
from mood_engine.offline_mood_classifier import OfflineMoodClassifier
from memory_engine.long_term_memory import LongTermMemory

MOOD_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'mood_engine', 'mood_config.json')
MOOD_MAPPINGS_PATH = os.path.join(os.path.dirname(__file__), 'mood_engine', 'mood_mappings.json')

class Personality:
    def __init__(self, config_path: str = 'config.json',
                 session_id: Optional[str] = None,
                 shared_embedder: SentenceTransformer = None,
                 shared_mood_classifier: OfflineMoodClassifier = None):
        
        self.session_id = session_id if session_id else "default_session"
        self.shared_embedder = shared_embedder

        with open(config_path, 'r', encoding="utf-8") as f:
            self.config = json.load(f)
        
        # --- Load All Configurations ---
        self.mood_eq_settings: Dict = {}
        self.behavior_modifiers: List[Dict] = []
        self.mood_drift_rules: List[Dict] = []
        self.mood_query_keywords: List[str] = []
        self.conversation_memory_settings: Dict = {}
        self.memory_settings: Dict = {}
        self.patience_settings: Dict = {}
        self.mood_style_directives: List[Dict] = []
        self.default_llm_max_tokens: int = 40
        self.initial_mood_state: Optional[Dict[str, float]] = None 
        self.breaking_point_mood_delta: Dict[str, float] = {}

        try:
            with open(MOOD_CONFIG_PATH, 'r', encoding="utf-8") as f:
                mood_config_data = json.load(f)
                self.mood_eq_settings = mood_config_data.get('mood_eq_settings', {})
                self.behavior_modifiers = mood_config_data.get('behavior_modifiers', [])
                self.mood_drift_rules = mood_config_data.get('mood_drift_rules', [])
                self.mood_query_keywords = mood_config_data.get('mood_query_keywords', [])
                self.conversation_memory_settings = mood_config_data.get('conversation_memory_settings', {"max_turns": 5})
                self.memory_settings = mood_config_data.get('memory_settings', {})
                self.patience_settings = mood_config_data.get('patience_tracker', {})
                self.breaking_point_mood_delta = self.patience_settings.get('breaking_point_mood_delta', {})
                self.mood_style_directives = mood_config_data.get('mood_style_directives', [])
                self.default_llm_max_tokens = mood_config_data.get('default_llm_max_tokens', 40)
                self.initial_mood_state = self.mood_eq_settings.get('initial_state')
            print(f"[Personality-{self.session_id}] Loaded core mood config from {MOOD_CONFIG_PATH}")
        except FileNotFoundError:
            print(f"[Personality-{self.session_id}] Warning: Mood config file not found.")
        except json.JSONDecodeError as e:
            print(f"[Personality-{self.session_id}] Error decoding Mood config: {e}.")

        self.keyword_triggers: List[Dict] = []
        self.semantic_mood_mapping: List[Dict] = []
        self.emoji_mood_mapping: Dict[str, List[Dict]] = {}
        try:
            with open(MOOD_MAPPINGS_PATH, 'r', encoding="utf-8") as f:
                mood_mappings_data = json.load(f)
                self.keyword_triggers = mood_mappings_data.get('keyword_triggers', [])
                self.semantic_mood_mapping = mood_mappings_data.get('semantic_mood_mapping', [])
                self.emoji_mood_mapping = mood_mappings_data.get('emoji_mood_mapping', {})
            print(f"[Personality-{self.session_id}] Loaded mood mappings from {MOOD_MAPPINGS_PATH}")
        except FileNotFoundError:
            print(f"[Personality-{self.session_id}] Warning: Mood mappings file not found.")
        except json.JSONDecodeError as e:
            print(f"[Personality-{self.session_id}] Error decoding Mood mappings: {e}.")

        session_data_dir = os.path.join(os.path.dirname(MOOD_CONFIG_PATH), 'session_data', self.session_id)
        os.makedirs(session_data_dir, exist_ok=True)
        silver_soul_path = os.path.join(session_data_dir, 'silver_soul.json')
        
        self.mood_eq = MoodEQ(
            initial_state=self.initial_mood_state,
            default_decay_rate=self.mood_eq_settings.get('default_decay_rate', 0.05),
            decay_interval_seconds=self.mood_eq_settings.get('decay_interval_seconds', 90),
            bias_map=self.mood_eq_settings.get('bias_map'),
            decay_profiles=self.mood_eq_settings.get('decay_profiles'),
            mood_drift_rules=self.mood_drift_rules,
            max_history_turns=self.conversation_memory_settings.get("max_turns", 5) * 2,
            state_file_path=silver_soul_path
        )
        print(f"[Personality-{self.session_id}] MoodEQ initialized. Session state file: {silver_soul_path}")

        ltm_db_path = os.path.join(os.path.dirname(__file__), 'memory_engine', 'db')
        self.long_term_memory = LongTermMemory(
            session_id=self.session_id,
            embedder=shared_embedder,
            db_path=ltm_db_path,
            memory_config=self.memory_settings
        )

        self.mood_classifier = shared_mood_classifier

        self.primary_llm_model: str = 'nous-hermes2:10.7b-solar-q4_K_M'
        self.primary_prompt_template: str = self.config['persona_prompt_template']
        self.stop_tokens: List[str] = ["<|im_end|>", "user:"]

        self.session_start_time: float = time.time()
        self.max_session_duration_seconds: int = 90 * 60
        self.fallback_active: bool = False
        self.fallback_cooldown_duration_seconds: int = 10 * 60
        self.last_stress_event_time: float = 0
        self._effective_llm_max_tokens: Optional[int] = None

        self.patience_threshold: int = self.patience_settings.get("threshold", 2)
        self.patience_counter: int = 0
        self.last_user_embedding: Optional[np.ndarray] = None
        self.repetition_similarity_threshold: float = 0.95

        print(f"[Personality-{self.session_id}] Using primary LLM model: {self.primary_llm_model}")
        print(f"[Personality-{self.session_id}] Conversation memory max turns: {self.mood_eq.max_history_turns // 2} (User + AI)")

    def _detect_and_apply_mood_changes(self, user_input: str):
        user_input_lower = user_input.lower()
        combined_delta: Dict[str, float] = {}
        for trigger in self.keyword_triggers:
            if any(keyword in user_input_lower for keyword in trigger['keywords']):
                for band, value in trigger['mood_delta'].items():
                    combined_delta[band] = combined_delta.get(band, 0.0) + value
        if self.mood_classifier:
            semantic_delta = self.mood_classifier.predict_eq_delta(user_input)
            if semantic_delta:
                for band, value in semantic_delta.items():
                    combined_delta[band] = combined_delta.get(band, 0.0) + value
        emoji_punc_delta = self._detect_emoji_punctuation_mood(user_input)
        if emoji_punc_delta:
            for band, value in emoji_punc_delta.items():
                combined_delta[band] = combined_delta.get(band, 0.0) + value
        if combined_delta:
            self.mood_eq.update_mood(combined_delta)
        else:
            self.mood_eq.apply_decay()

    def _detect_emoji_punctuation_mood(self, user_input: str) -> Dict[str, float]:
        detected_delta: Dict[str, float] = {}
        def add_to_delta(current_delta: Dict[str, float], incoming_delta: Dict[str, float]):
            for band, value in incoming_delta.items():
                current_delta[band] = current_delta.get(band, 0.0) + value
        for category in ["positive_emojis", "negative_emojis"]:
            if category in self.emoji_mood_mapping:
                for mapping in self.emoji_mood_mapping[category]:
                    for emoji_char in mapping['emojis']:
                        if emoji_char in user_input:
                            add_to_delta(detected_delta, mapping['mood_delta'])
        if "punctuation_patterns" in self.emoji_mood_mapping:
            for mapping in self.emoji_mood_mapping["punctuation_patterns"]:
                if re.search(mapping['pattern'], user_input):
                    add_to_delta(detected_delta, mapping['mood_delta'])
        return detected_delta

    def _apply_mood_behavior_modifiers(self, llm_raw_response: str) -> str:
        current_mood_vector = self.mood_eq.get_current_vector()
        modified_response = llm_raw_response
        applicable_modifiers = []
        for mod_rule in self.behavior_modifiers:
            mood_band = mod_rule['mood']
            threshold = mod_rule['threshold']
            if mood_band in current_mood_vector and current_mood_vector[mood_band] >= threshold:
                applicable_modifiers.append((current_mood_vector[mood_band], mod_rule))
        applicable_modifiers.sort(key=lambda x: x[0], reverse=True)
        if applicable_modifiers:
            final_prefix = ""
            final_suffix = ""
            for value, rule in applicable_modifiers:
                if rule.get('override_if_high', False):
                    final_prefix = random.choice(rule['prefix']) if rule.get('prefix') else ""
                    final_suffix = random.choice(rule['suffix']) if rule.get('suffix') else ""
                    break
            if not final_prefix and not final_suffix:
                _ , most_dominant_rule = applicable_modifiers[0]
                final_prefix = random.choice(most_dominant_rule['prefix']) if most_dominant_rule.get('prefix') else ""
                final_suffix = random.choice(most_dominant_rule['suffix']) if most_dominant_rule.get('suffix') else ""
            modified_response = f"{final_prefix} {modified_response} {final_suffix}".strip()
        return modified_response

    def _check_and_manage_system_state(self):
        current_time = time.time()
        if self.fallback_active:
            if (current_time - self.last_stress_event_time) < self.fallback_cooldown_duration_seconds: return
            else:
                self.fallback_active = False
                self.session_start_time = current_time
        if (current_time - self.session_start_time) > self.max_session_duration_seconds:
            self.fallback_active = True
            self.last_stress_event_time = current_time

    def _get_llm_prompt_with_mood(self, user_input: str) -> str:
        current_soul_state = self.mood_eq.get_soul_state_object()
        style_directive_str = ""
        self._effective_llm_max_tokens = self.default_llm_max_tokens 
        applicable_directives = []
        for directive_rule in self.mood_style_directives:
            mood_band = directive_rule['mood']
            threshold = directive_rule['threshold']
            if mood_band in current_soul_state['mood_vector'] and current_soul_state['mood_vector'][mood_band] >= threshold:
                applicable_directives.append((current_soul_state['mood_vector'][mood_band], directive_rule))
        applicable_directives.sort(key=lambda x: x[0], reverse=True)
        chosen_directive = None
        for value, rule in applicable_directives:
            if rule.get('override_directive', False):
                chosen_directive = rule
                break
        if not chosen_directive and applicable_directives:
            chosen_directive = applicable_directives[0][1]
        if chosen_directive:
            style_directive_str = chosen_directive['directive']
            self._effective_llm_max_tokens = chosen_directive.get('max_tokens', self.default_llm_max_tokens)
        else:
            style_directive_str = "Maintain Silver's concise, intelligent, and witty condescending tone. Prioritize brevity."
        
        current_history = self.mood_eq.get_conversation_history() 
        history_str = ""
        if current_history:
            history_str = "\n" + "\n".join([f"{role}: {msg}" for role, msg in current_history])

        relevant_memories_raw = self.long_term_memory.fetch_memories(user_input)
        relevant_memories = [mem['document'] for mem in relevant_memories_raw]
        long_term_memory_str = ""
        if relevant_memories:
            long_term_memory_str = "\n\n--- Relevant Long-Term Memories (for context) ---\n- " + "\n- ".join(relevant_memories)

        important_memories = self.long_term_memory.get_most_important_memories()
        if important_memories:
            long_term_memory_str += "\n\n--- CRITICAL MEMORIES (Proactively remind the user if relevant) ---\n- " + "\n- ".join(important_memories)

        insertion_marker = "'{user_input}'<|im_end|>"
        template_parts = self.primary_prompt_template.split(insertion_marker, 1)
        
        dynamic_prompt_injection = ""
        if style_directive_str: 
            dynamic_prompt_injection += f" {style_directive_str} "
        if long_term_memory_str: 
            dynamic_prompt_injection += long_term_memory_str
        if history_str:
            dynamic_prompt_injection += f"\n\n--- Start of Recent Conversation History ---\n{history_str}\n--- End of Recent Conversation History ---\n"
            dynamic_prompt_injection += "\n\n**Your primary task is to respond to the user's *latest* message, using the conversation history and long-term memories ONLY for context.**"
        
        final_prompt_template = template_parts[0] + dynamic_prompt_injection + insertion_marker + template_parts[1]
        final_prompt = final_prompt_template.format(user_input=user_input)
        return final_prompt

    def _generate_response(self, user_input: str) -> str:
        self._check_and_manage_system_state()
        if self.fallback_active:
            return "My core processors are currently cooling down."
        final_prompt = self._get_llm_prompt_with_mood(user_input)
        ollama_options = {"stop": self.stop_tokens}
        if self._effective_llm_max_tokens is not None:
            ollama_options["num_predict"] = self._effective_llm_max_tokens
        try:
            raw_llm_response = ollama.generate(model=self.primary_llm_model, prompt=final_prompt, options=ollama_options)['response'].strip()
            silver_response = self._apply_mood_behavior_modifiers(raw_llm_response)
            return silver_response
        except Exception as e:
            print(f"[Ollama Error-{self.session_id}]: {e}.")
            self.fallback_active = True
            self.last_stress_event_time = time.time()
            return "My connection to the ether is severed."

    def _query_mood(self) -> str:
        soul_state = self.mood_eq.get_soul_state_object()
        mood_vector = soul_state['mood_vector']
        dominant_moods = soul_state['dominant_moods']
        mood_descriptions = []
        for band, value in mood_vector.items():
            if value > 0.05:
                mood_descriptions.append(f"{int(value * 100)}% {band.capitalize()}")
        if mood_descriptions:
            mood_summary = ", ".join(mood_descriptions)
            if len(dominant_moods) > 0:
                dominant_str = ", ".join([m.capitalize() for m in dominant_moods])
                return f"Currently, my dominant axes include {dominant_str}. Specifically, I'm at {mood_summary}."
            else:
                return f"My emotional state is subtle. Perhaps {mood_summary}."
        else:
            return "My emotional state is calibrated."

    def _summarize_turn_for_ltm(self, user_input: str, silver_response: str) -> str:
        summary_prompt = f"""Condense the following user-AI exchange into a single, concise, factual sentence from a third-person perspective. Example: The user asked for the sum of 2+2, and the AI confirmed it was 4.
        
        Exchange:
        User: "{user_input}"
        AI: "{silver_response}"

        Summary:"""
        try:
            response = ollama.generate(model=self.primary_llm_model, prompt=summary_prompt, options={"stop": ["\n"], "num_predict": 50})
            summary = response['response'].strip()
            print(f"[LTM Summary] Generated: '{summary}'")
            return summary
        except Exception as e:
            print(f"[LTM Summary] Error generating summary: {e}")
            return f"The user said '{user_input}', and the AI replied '{silver_response}'."

    def _assess_importance(self, text: str) -> float:
        importance_prompt = f"""On a scale of 0.1 (trivial) to 1.0 (critical), how important is this statement for a personal assistant to remember? A 'critical' statement is a direct fact about the user (name, preference, goal) or a key instruction. A 'trivial' statement is a simple greeting, conversational filler, or a question that has been answered. Respond with ONLY the float value.

        Statement: "{text}"
        Importance:"""
        try:
            response = ollama.generate(model=self.primary_llm_model, prompt=importance_prompt, options={"num_predict": 5})
            importance_str = response['response'].strip().split()[0].replace(",", ".")
            return float(importance_str)
        except (ValueError, IndexError) as e:
            print(f"[LTM Importance] Could not parse importance float. Defaulting to 0.5. Error: {e}")
            return 0.5
        except Exception as e:
            print(f"[LTM Importance] Error generating importance: {e}")
            return 0.5

    def process_input(self, user_input: str) -> str:
        user_input_lower = user_input.lower()

        # --- SEMANTIC Patience / Breaking Point Logic ---
        current_embedding = self.shared_embedder.encode([user_input_lower])
        
        if self.last_user_embedding is not None:
            similarity = cosine_similarity(current_embedding, self.last_user_embedding)[0][0]
            if similarity > self.repetition_similarity_threshold:
                self.patience_counter += 1
                print(f"[PatienceTracker-{self.session_id}] Semantic repetition detected (Similarity: {similarity:.2f}). Counter at {self.patience_counter}/{self.patience_threshold}.")
            else:
                self.patience_counter = 0
        
        self.last_user_embedding = current_embedding

        if self.patience_counter >= self.patience_threshold:
            print(f"[PatienceTracker-{self.session_id}] Breaking point reached. Forcing direct response.")
            self.patience_counter = 0 
            self.last_user_embedding = None
            if self.breaking_point_mood_delta:
                self.mood_eq.update_mood(self.breaking_point_mood_delta)
                print(f"[Mood-{self.session_id}] Injected breaking point mood delta: {self.breaking_point_mood_delta}")
            
            # --- NEW: Smart Forced Prompt ---
            forced_prompt = f"""The user has been annoyingly repetitive with the input: "{user_input}". Your patience has snapped.
            Deliver a final, curt, dismissive response that incorporates the phrase: "{self.patience_settings.get('breaking_point_response', 'Fine. Here.')}".
            If the input was a question, provide the answer directly and concisely within the dismissal. Do NOT greet them back. Simply end the loop.
            Final Response:"""
            try:
                response = ollama.generate(model=self.primary_llm_model, prompt=forced_prompt, options={"num_predict": 40})
                return response['response'].strip()
            except Exception as e:
                print(f"[Ollama Error-{self.session_id}]: {e}.")
                return "Fine. 4. Now stop." # Hardcoded fallback

        # --- End of Semantic Patience Logic ---

        if any(keyword in user_input_lower for keyword in self.mood_query_keywords):
            return self._query_mood()
        
        remember_command = "remember this:"
        if user_input_lower.startswith(remember_command):
            memory_to_add = user_input[len(remember_command):].strip()
            if memory_to_add:
                self.long_term_memory.add_memory(memory_to_add, importance=1.0)
                return "Acknowledged. I'll remember that."
            else:
                return "Remember what? Provide the information after the command."

        decay_chance = self.memory_settings.get("decay_chance", 0.1)
        if random.random() < decay_chance:
            self.long_term_memory.decay_memories()

        self.mood_eq.add_to_conversation_history("user", user_input)
        self._detect_and_apply_mood_changes(user_input)
        self.mood_eq.drift_mood()
        silver_response = self._generate_response(user_input)
        self.mood_eq.add_to_conversation_history("assistant", silver_response)
        
        turn_summary = self._summarize_turn_for_ltm(user_input, silver_response)
        importance_score = self._assess_importance(turn_summary)
        self.long_term_memory.add_memory(turn_summary, importance=importance_score)
        
        return silver_response