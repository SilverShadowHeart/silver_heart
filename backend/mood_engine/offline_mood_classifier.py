import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, List, Optional, Tuple

class OfflineMoodClassifier:
    def __init__(self, semantic_mood_data: List[Dict],
                 embedder: SentenceTransformer,
                 n_neighbors: int = 3):
        
        self.embedder = embedder
        self.classifier: Optional[KNeighborsClassifier] = None
        self.semantic_mood_data: List[Dict] = semantic_mood_data
        self.X_train_text: List[str] = []
        self.y_train_labels: List[str] = []
        self.mood_label_to_delta_map: Dict[str, Dict[str, float]] = {}

        for mood_entry in self.semantic_mood_data:
            label = mood_entry.get('label')
            mood_delta = mood_entry.get('mood_delta')
            training_phrases = mood_entry.get('training_phrases', [])
            if label and mood_delta and training_phrases:
                for phrase in training_phrases:
                    self.X_train_text.append(phrase)
                    self.y_train_labels.append(label)
                self.mood_label_to_delta_map[label] = mood_delta

        if not self.X_train_text:
            print("[OfflineMoodClassifier] Warning: No valid training data found. Semantic classification will not function.")
            return

        self._train_classifier(n_neighbors)

    def _train_classifier(self, n_neighbors: int):
        print("[OfflineMoodClassifier] Encoding training data...")
        X_train_vec: np.ndarray = self.embedder.encode(self.X_train_text, convert_to_tensor=False)
        print(f"[OfflineMoodClassifier] Training KNN classifier with {len(self.X_train_text)} samples...")
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.classifier.fit(X_train_vec, self.y_train_labels)
        print("[OfflineMoodClassifier] Classifier trained successfully.")

    def predict_eq_delta(self, user_input: str) -> Dict[str, float]:
        if not self.classifier:
            return {}
        try:
            input_vec: np.ndarray = self.embedder.encode([user_input], convert_to_tensor=False)
            predicted_label: str = self.classifier.predict(input_vec)[0]
            eq_delta = self.mood_label_to_delta_map.get(predicted_label, {})
            return eq_delta
        except Exception as e:
            print(f"[OfflineMoodClassifier] Error during semantic prediction: {e}")
            return {}