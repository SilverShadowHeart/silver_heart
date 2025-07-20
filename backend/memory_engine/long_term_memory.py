import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import time
from typing import List, Dict, Optional

class LongTermMemory:
    def __init__(self, session_id: str,
                 embedder: SentenceTransformer,
                 db_path: str = "backend/memory_engine/db",
                 memory_config: Optional[Dict] = None):
        
        self.session_id = session_id
        self.embedder = embedder

        if memory_config is None: memory_config = {}
        self.decay_factor = memory_config.get("decay_factor", 1.5e-6)
        self.forget_threshold = memory_config.get("forget_threshold", 0.1)

        self.client = chromadb.PersistentClient(path=db_path)
        
        collection_name = f"silver_memory_{self.session_id.replace('-', '_')}"
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
        )

    def add_memory(self, memory_text: str, importance: float = 0.5):
        """
        Adds a new memory with a calculated or specified importance score.
        """
        if not memory_text or not memory_text.strip(): return

        try:
            memory_id = str(uuid.uuid4())
            current_time = int(time.time())
            
            metadata = {
                "timestamp": current_time,
                "last_accessed": current_time,
                "salience": 1.0,
                "importance": max(0.1, min(1.0, importance))
            }
            
            self.collection.add(
                documents=[memory_text],
                metadatas=[metadata],
                ids=[memory_id]
            )
            print(f"[LTM-{self.session_id}] Added new memory (Importance: {importance:.2f}): '{memory_text[:50]}...'")
        except Exception as e:
            print(f"[LTM-{self.session_id}] Error adding memory: {e}")

    # CORRECTED INDENTATION: This method is now at the class level
    def fetch_memories(self, query_text: str, n_results: int = 3) -> List[Dict]:
        """
        Fetches the most relevant memories that are not "forgotten",
        and reinforces them by boosting their salience.
        Returns a list of dictionaries with document and metadata.
        """
        try:
            if self.collection.count() == 0: return []
            
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(n_results, self.collection.count()),
                where={"salience": {"$gte": self.forget_threshold}},
                include=["documents", "metadatas"]
            )
            
            if not results or not results['ids'][0]: return []

            fetched_ids = results['ids'][0]
            fetched_documents = results['documents'][0]
            fetched_metadatas = results['metadatas'][0]
            
            updated_metadatas = []
            for meta in fetched_metadatas:
                meta["last_accessed"] = int(time.time())
                meta["salience"] = 1.0
                updated_metadatas.append(meta)
            
            self.collection.update(ids=fetched_ids, metadatas=updated_metadatas)
            
            return [{"document": doc, "metadata": meta} for doc, meta in zip(fetched_documents, fetched_metadatas)]
        except Exception as e:
            print(f"[LTM-{self.session_id}] Error fetching memories: {e}")
            return []
        
    def get_most_important_memories(self, n_results: int = 2) -> List[str]:
        """
        Retrieves the memories with the highest importance scores, regardless of query.
        """
        try:
            if self.collection.count() == 0: return []

            memories = self.collection.get(
                where={"salience": {"$gte": self.forget_threshold}},
                include=["documents", "metadatas"]
            )
            
            sorted_memories = sorted(
                zip(memories['documents'], memories['metadatas']),
                key=lambda item: item[1].get('importance', 0.1),
                reverse=True
            )
            
            return [doc for doc, meta in sorted_memories[:n_results]]
        except Exception as e:
            print(f"[LTM-{self.session_id}] Error getting most important memories: {e}")
            return []

    def decay_memories(self):
        """
        Applies decay to salience, influenced by the memory's importance.
        Important memories decay slower.
        """
        try:
            if self.collection.count() == 0: return

            memories = self.collection.get(include=["metadatas"])
            
            ids_to_update = []
            metadatas_to_update = []
            current_time = int(time.time())

            for i in range(len(memories['ids'])):
                memory_id = memories['ids'][i]
                metadata = memories['metadatas'][i]
                
                last_accessed = metadata.get("last_accessed", current_time)
                current_salience = metadata.get("salience", 1.0)
                importance = metadata.get("importance", 0.5)

                time_since_accessed = current_time - last_accessed
                decay_amount = time_since_accessed * self.decay_factor * (1.1 - importance)
                
                if decay_amount > 0:
                    new_salience = max(0.0, current_salience - decay_amount)
                    metadata['salience'] = new_salience
                    ids_to_update.append(memory_id)
                    metadatas_to_update.append(metadata)

            if ids_to_update:
                self.collection.update(ids=ids_to_update, metadatas=metadatas_to_update)
                print(f"[LTM-{self.session_id}] Decayed salience for {len(ids_to_update)} memories.")
        except Exception as e:
            print(f"[LTM-{self.session_id}] Error during memory decay: {e}")