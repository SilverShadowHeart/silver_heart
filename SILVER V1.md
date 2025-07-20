# Project Silver: Architectural Documentation

### An Emotionally-Aware, Resource-Constrained Offline AI Assistant

---

## 1. Abstract & Core Philosophy

Project Silver is an experimental, emotionally-adaptive AI assistant designed to run entirely offline on consumer-grade hardware. Developed by a solo engineer, it demonstrates a complex, modular architecture that integrates real-time mood evaluation, a persistent memory system, and a dynamic personality engine under strict resource constraints (target < 3.5GB VRAM).

The core philosophy rejects the stateless, purely functional model of traditional assistants. Silver is engineered to be a presence—a consistent, dynamic, and persistent personality that learns, remembers, and feels. This is achieved through four foundational pillars: a rich **Emotional Engine**, a multi-layered **Memory System**, a strong **Personality & Expression System**, and a robust **System Integrity** framework.

---

### ⚠️ IMPORTANT NOTE: EXPERIMENTAL BUILD STATUS

This project is a successful proof of concept, but it is an **experimental build** and should be treated as such. It is not a polished, production-ready application. Users should expect bugs, performance issues, and inconsistencies.

**Key Known Issues:**

- 🔴 **Memory Contamination Bug (High Priority):** A critical bug exists where the long-term memory system can retrieve a semantically similar but contextually irrelevant old memory. This can cause the AI to believe it has already addressed a new question, leading to confusing or hostile responses (e.g., "The answer was given. Use it wisely."). This severely impacts the AI's logical consistency.
    
- 🔴 **Unstable TTS Engine (High Priority):** The current Text-to-Speech (pyttsx3) implementation is a known weak point. It is prone to failure during conversations, causing the voice to stop working until a server restart. It is considered **broken** for reliable, continuous use.
    
- 🟡 **Slow Initial Startup:** The one-time loading of the embedding model and mood classifier at startup is **considerable** (30-60+ seconds) and is not yet fully optimized.
    
- 🟡 **Resource Intensive:** The system's "fully offline" design requires significant RAM/VRAM to run smoothly, pushing the limits of the target hardware.
    

---

## 2. System Architecture Overview

Silver is built on a modular architecture where each component has a distinct responsibility. This design allows for dynamic loading/unloading of resources and clear separation of concerns.

### High-Level Data Flow Diagram


[Architecture](architechure.png)



---

## 3. The Lifecycle of a Request: A Case Study

To understand how Silver's components interact, consider a scenario where the user asks "what is 2+2?" for the third time.

1. **Request Reception (main.py):** The Flask server receives the POST request with the user's message and session_id. It retrieves the correct Personality instance for that session.
    
2. **Irritation Check (personality_engine.py):**
    
    - The process_input method converts the new input into a semantic vector.
        
    - This vector is compared to the vector of the last input. cosine_similarity is extremely high (>0.95), flagging it as a repetition.
        
    - The repetition_counter increments to 2.
        
    - The irritation_system rules in mood_config.json are checked. The rule for trigger_count: 2 is activated.
        
    - A significant mood_delta of {"anger": 0.4, "sarcasm": 0.3} is generated.
        
3. **Single-Turn Mood Detection (personality_engine.py):** The input is analyzed by the three mood detection tiers (Keyword, Semantic, Expressive). In this case, it is found to be neutral.
    
4. **Mood Vector Update (mood_engine/mood_eq.py):**
    
    - The update_mood method is called with the powerful mood_delta from the irritation system.
        
    - Silver's anger and sarcasm values spike dramatically. Her internal state is now "very annoyed."
        
    - This new emotional state is persisted to the session's silver_soul.json file.
        
5. **Memory Retrieval (memory_engine/long_term_memory.py):**
    
    - The _get_llm_prompt_with_mood method queries ChromaDB for memories semantically similar to the input.
        
    - It retrieves summaries of the last two times the user asked this exact question.
        
6. **Master Prompt Assembly (personality_engine.py):**
    
    - The system checks Silver's now-angry mood and selects the corresponding mood_style_directive from the config: "Your tone is sharp and impatient. Your response should be a single, direct sentence..."
        
    - It assembles the final prompt for the LLM, containing:
        
        - The core persona instructions.
            
        - The new, angry style directive.
            
        - Long-term memories of the previous questions.
            
        - Short-term chat history.
            
        - The user's current input.
            
7. **LLM Generation (personality_engine.py):** The prompt is sent to the local Ollama server. The nous-hermes2 model generates a response, heavily influenced by the angry directive and memory of being asked repeatedly.
    
8. **Response Post-Processing (personality_engine.py):** The raw LLM response is passed to _apply_mood_behavior_modifiers. Because anger is high, a prefix like "Listen." might be prepended to the response.
    
9. **Memory Creation (personality_engine.py):** The final exchange is summarized by the LLM into a new memory: "The user asked for the sum of 2+2 for a third time, and the AI responded with open hostility." This summary is rated for importance and stored in ChromaDB.
    
10. **Final Output (main.py):** The final text is sent to the UI. The text and the "angry" mood vector are sent to the speak function, causing the TTS voice to be faster and louder.
    

---

## 4. Deep Dive: Architectural Pillars

### Pillar 1: The Emotional Engine

The heart of the AI's personality. It is responsible for tracking, updating, and applying an internal emotional state.

- **1.1: 12-Band Emotional Equalizer (MoodEQ):** The core state manager, implemented in mood_engine/mood_eq.py. It holds the 12-band mood vector (Happy, Angry, Sarcasm, etc.), manages the silver_soul.json session file, and contains the logic for natural **mood decay**, where emotions fade over time if not reinforced.
    
- **1.2: Multi-Tiered Mood Detection:** Orchestrated by personality_engine.py, this system analyzes user input on three levels:
    
    - **Keyword Analysis:** Fast check for words with strong emotional valence (e.g., "hate," "love," "idiot").
        
    - **Semantic Classification:** An OfflineMoodClassifier (scikit-learn KNN model) trained on example phrases to determine the emotional intent of the sentence as a whole.
        
    - **Expressive Analysis:** Detects emotional cues from punctuation and emojis (e.g., !!!, ???, :)).
        
- **1.3: Emotional Catalysts:** A system in personality_engine.py that tracks conversational patterns over multiple turns. It detects behaviors like **Repetition**, **Overpraise**, or **Helplessness** and applies significant, pre-defined mood shifts, creating more complex emotional reactions than a single turn analysis ever could.
    
- **1.4: Mood Drift:** A subtle mechanic in mood_eq.py where strong emotions can influence weaker ones. For example, high anger can slowly increase sarcasm while suppressing patience, creating more natural emotional transitions.
    

### Pillar 2: The Memory System

The mechanism for persistence and learning, allowing the AI to build context over long periods.

- **2.1: Long-Term Vector Memory (LongTermMemory):** Implemented in memory_engine/long_term_memory.py. This class is a wrapper around a persistent ChromaDB client. It creates a unique "collection" (database) for each session_id, ensuring user memories are kept separate.
    
- **2.2: Intelligent Memory Creation:** After each turn, personality_engine.py uses the LLM to perform two micro-tasks:
    
    1. **Summarize:** It generates a concise, third-person summary of the interaction.
        
    2. **Assess Importance:** It rates the summary's importance on a scale of 1-10.  
        This summary, its importance score, and a timestamp are stored as a new memory.
        
- **2.3: Memory Decay & Reinforcement:** Logic within long_term_memory.py simulates human memory.
    
    - **Decay:** A background process periodically reduces the "salience" of all memories based on their importance and age. Unimportant memories fade faster.
        
    - **Reinforcement:** When a memory is retrieved for use in a prompt, its salience is reset to 1.0, reinforcing its importance.
        

### Pillar 3: The Personality & Expression System

This pillar translates the internal state (mood and memory) into a tangible, observable personality.

- **3.1: Core Persona:** A foundational set of instructions in config.json that defines Silver's baseline identity, knowledge boundaries, and communication style.
    
- **3.2: Dynamic Style Directives:** The personality_engine.py checks the current dominant mood against the mood_style_directives in the config. It injects the winning directive (e.g., an angry, sarcastic, or curious persona) into the final LLM prompt, fundamentally altering the tone, pacing, and wording of the response.
    
- **3.3: Behavioral Modifiers:** A post-processing step that applies simple, rule-based changes to the LLM's raw text. This can add conversational tics, prefixes, or adjust sentence structure based on the current mood.
    
- **3.4: Audial Expression:** The speak function in main.py uses the final mood vector to adjust the TTS voice parameters. High anger increases the speech rate and volume, while high sadness might slow it down, mapping emotion to vocal delivery.
    

---

## 5. Technology Stack & Setup

### Technology Stack

- **Backend Framework:** Python, Flask
    
- **Local LLM Server:** Ollama
    
- **Primary LLM Model:** nous-hermes2:10.7b-solar-q4_K_M
    
- **Vector Database:** ChromaDB (persistent local storage)
    
- **Embedding Model:** all-MiniLM-L6-v2 (from SentenceTransformers)
    
- **Mood Classifier:** scikit-learn (KNeighborsClassifier)
    
- **Frontend:** HTML, CSS, JavaScript (via eel)
    
- **Text-to-Speech:** pyttsx3 (unstable, slated for replacement)
    

### Setup and Installation

#### Prerequisites

1. **Python 3.10+** installed and added to your PATH.
    
2. **Ollama** installed and running. ([Download here](https://ollama.com/download))
    
3. A system with sufficient RAM/VRAM (Recommended: 16GB+ System RAM or 8GB+ VRAM).
    

#### Installation Steps

1. Clone the project repository.
    
2. Navigate to the project's root directory in your terminal.
    
3. Install all required Python packages from the requirements.txt file:
    
    Generated bash
    
    ```
    pip install -r requirements.txt
    ```
    

    
4. Pull the required LLM model using the Ollama CLI:
    

    
    ```
    ollama pull nous-hermes2:10.7b-solar-q4_K_M
    ```

    

#### Configuration

1. **(Highly Recommended) Set Ollama Keep-Alive:** To prevent the model from being unloaded between requests (which causes massive delays), set the OLLAMA_KEEP_ALIVE system environment variable. A value of 10m is recommended.
    
2. **(Optional) Select TTS Voice:** When you first run the backend, it will list available TTS voices in the console. Copy the ID of your preferred voice and paste it into the default_voice_id field in backend/config/mood_config.json.
    

---

## 6. Current Limitations & Future Roadmap

### Immediate Priorities (Bug Fixing & Stability)

- **[P0] Fix Memory Contamination:** Resolve the critical bug where old, irrelevant memories are retrieved. This may involve adding "topic" metadata to memories and filtering queries by it, or refining the memory retrieval prompt.
    
- **[P0] Replace TTS Engine:** Replace the unstable pyttsx3 with a modern, robust, and local TTS engine like **Coqui TTS** or **Piper**.
    
- **[P1] Optimize Loading:** Investigate methods to accelerate the initial model loading process, potentially through model quantization or caching optimizations.
    

### High-Priority Features (Intelligence & Agency)

- **Task Execution:** Grant Silver the ability to run local scripts, manage files, or interact with APIs, turning her from a conversationalist into a true "smart assistant."
    
- **Formal Command Parser:** Implement a system to handle explicit commands like #summarize_session, #reset_mood, or #forget_last.
    

### Medium-Priority Features (UI & Polish)

- **Memory Viewer:** Add a panel to the UI that allows the user to view, search, and manually delete Silver's long-term memories.
    
- **Application Packaging:** Bundle the entire application into a single executable using a tool like PyInstaller for simple, one-click setup and distribution.
    
- **Emotional Regression Testing:** Create a formal test suite to validate that mood changes occur as expected under a variety of inputs, preventing "emotional bugs."
    

---

## 7. Developer Notes & Acknowledgements

"I may not know every Python module, and I certainly didn’t start this as an expert—but I learned as I built. I never claimed to be a perfect coder. What I created is far from trivial; it has depth, structure, and a real sense of direction. Despite limited hardware—a single overworked CPU fan and a GPU under constant load—I designed the full system architecture myself. Yes, I used AI to assist with coding and debugging, but every key decision, from emotional modeling to memory handling, came from my own thinking. This project is a reflection of what I’ve learned, how I think, and what I’m capable of building—even under pressure. It’s not flawless, but it’s mine. And I’m proud of what it’s grown into."

This project was a solo effort, driven by a passion for exploring the boundaries of AI personality. The design was heavily inspired by real-world concepts like audio equalizers and the process of reverse-engineering complex systems.

Special acknowledgement is given to the architecture of the [**Super-Memory-Agent**](https://github.com/sriram-dev-9/Super-Memory-Agent) repository by [**Sriram-dev-9**](https://github.com/sriram-dev-9), which provided the excellent foundational concept for using ChromaDB for persistent, local AI memory.