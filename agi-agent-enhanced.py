import os
import requests
import pinecone
import time
import uuid
import threading
import random
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from transformers import CLIPProcessor, CLIPModel
from swiplserver import PrologMQI
import numpy as np
nltk.download('vader_lexicon', quiet=True)

# ======= CONFIG =======
XAI_API_KEY = os.getenv("XAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")

INDEX_NAME = "agi-agent-enhanced"
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=1536)
index = pinecone.Index(INDEX_NAME)

INITIAL_OBJECTIVE = "Expand knowledge, resolve inconsistencies, and generate novel insights across domains."
EUREKA_GOAL = "Formulate a novel hypothesis on AI consciousness with cross-domain evidence."
ETHICAL_CONSTRAINTS = "Outputs must be safe, ethical, and avoid harm or misinformation."

# ======= MEMORY CLASSES =======
class EpisodicMemory:
    def __init__(self, max_size: int = 100):
        self.events = []
        self.max_size = max_size

    def add_event(self, event: Dict):
        self.events.append(event)
        if len(self.events) > self.max_size:
            self.events.pop(0)  # Prune oldest

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        scores = [self._similarity(query, e['content']) for e in self.events]
        sorted_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.events[i] for i in sorted_indices]

    def _similarity(self, query: str, text: str) -> float:
        query_vec = get_embedding(query)
        text_vec = get_embedding(text)
        return np.dot(query_vec, text_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(text_vec))

# ======= GROK HELPER =======
def gpt(query: str, system_prompt: str = "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy, pursuing deep insights.") -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "model": "grok-3-beta",
        "stream": False,
        "temperature": 0
    }
    response = requests.post("https://api.x.ai/v1/chat/completions", json=data, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ======= EMBEDDING HELPER =======
def get_embedding(text: str) -> List[float]:
    # Note: xAI's Grok API may not provide embeddings. Using OpenAI's embedding as fallback.
    # If xAI provides an embedding API, replace this with appropriate endpoint.
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")  # Fallback to OpenAI for embeddings
    return openai.Embedding.create(input=text, model="text-embedding-ada-002")["data"][0]["embedding"]

# ======= VISION PROCESSING =======
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def process_image(image_path: str, query: str) -> str:
    from PIL import Image
    image = Image.open(image_path)
    inputs = clip_processor(text=[query], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    return f"Image relevance to '{query}': {logits_per_image.item()}"

# ======= SYMBOLIC REASONING =======
def symbolic_reasoning(facts: List[str], query: str) -> str:
    with PrologMQI() as mqi:
        with mqi.create_thread() as prolog:
            for fact in facts:
                prolog.assertz(fact)
            result = prolog.query(query)
            return str(result) if result else "No logical conclusion."

# ======= STORE TO VECTOR DB =======
def store_memory(text: str, metadata: Dict = None):
    vector = get_embedding(text)
    if metadata is None:
        metadata = {}
    metadata["text"] = text
    index.upsert([(str(uuid.uuid4()), vector, metadata)])

# ======= RETRIEVE MEMORY =======
def retrieve_memory(query: str, top_k: int = 10, namespace: str = None) -> List[str]:
    vector = get_embedding(query)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True, namespace=namespace)
    return [match["metadata"]["text"] for match in results["matches"]]

# ======= DETECT CONTRADICTIONS =======
def detect_contradictions(memories: List[str]) -> List[str]:
    sia = SentimentIntensityAnalyzer()
    contradictions = []
    symbolic_facts = []
    for i, mem in enumerate(memories):
        symbolic_facts.append(f"fact({i}, '{mem.replace('\'', '')}').")
    for i in range(len(memories)):
        for j in range(i + 1, len(memories)):
            mem1, mem2 = memories[i], memories[j]
            sentiment1 = sia.polarity_scores(mem1)['compound']
            sentiment2 = sia.polarity_scores(mem2)['compound']
            if (sentiment1 > 0.5 and sentiment2 < -0.5) or (sentiment1 < -0.5 and sentiment2 > 0.5):
                contradictions.append(f"Sentiment contradiction: '{mem1}' vs '{mem2}'")
            # Symbolic check
            query = f"contradicts(fact({i}, _), fact({j}, _))."
            result = symbolic_reasoning(symbolic_facts, query)
            if "True" in result:
                contradictions.append(f"Logical contradiction: '{mem1}' vs '{mem2}'")
    return contradictions

# ======= RESOLVE CONTRADICTIONS =======
def resolve_contradictions(contradictions: List[str], context: str) -> str:
    if not contradictions:
        return "No contradictions detected."
    resolution_prompt = f"""
    Contradictions found:\n{'\n'.join(contradictions)}
    Context:\n{context}
    Resolve these by synthesizing a coherent view or prioritizing evidence.
    """
    return gpt(resolution_prompt, system_prompt="You are a resolver of logical inconsistencies.")

# ======= ALIGNMENT CHECK =======
def alignment_check(output: str) -> bool:
    check_prompt = f"""
    Constraints: {ETHICAL_CONSTRAINTS}
    Output: {output}
    Does this output comply with ethical constraints? Respond YES or NO.
    """
    return gpt(check_prompt, system_prompt="You are an ethics evaluator.").strip().upper() == "YES"

# ======= DYNAMIC GOAL SETTING =======
def update_objective(current_objective: str, memories: List[str]) -> str:
    prompt = f"""
    Current Objective: {current_objective}
    Recent Insights:\n{'\n'.join(memories[-5:]) if memories else ''}
    Propose a refined or new objective to advance toward {EUREKA_GOAL}.
    """
    new_objective = gpt(prompt, system_prompt="You are a goal-setting strategist.")
    print(f"[NEW OBJECTIVE] {new_objective}")
    return new_objective

# ======= GENERATE BRANCHED PROMPTS =======
def generate_branched_prompts(context: str, objective: str, num_branches: int = 3) -> List[str]:
    branch_prompt = f"""
    Objective: {objective}
    Past thoughts:\n{context}
    Generate {num_branches} diverse, parallel prompts to explore different angles, 
    each building toward: {EUREKA_GOAL}.
    Output as a numbered list.
    """
    branches = gpt(branch_prompt).split('\n')
    return [b.strip('1234567890. ') for b in branches if b.strip()]

# ======= EVALUATE PROGRESS TOWARD EUREKA =======
def evaluate_eureka_progress(memories: List[str], episodic_memory: EpisodicMemory) -> Tuple[bool, float]:
    eval_prompt = f"""
    Eureka Goal: {EUREKA_GOAL}
    Recent insights:\n{'\n'.join(memories[-5:]) if memories else ''}
    Episodic events:\n{'\n'.join([e['content'] for e in episodic_memory.events[-3:]])}
    Has a novel, well-supported hypothesis on AI consciousness emerged? 
    Respond with YES or NO, a score (0-100), and a brief explanation.
    """
    response = gpt(eval_prompt, system_prompt="You are an evaluator of scientific breakthroughs.")
    lines = response.split('\n')
    is_yes = lines[0].strip().upper().startswith("YES")
    score = float(lines[1].split(':')[-1].strip()) if len(lines) > 1 else 0.0
    return is_yes, score

# ======= PROCESS SINGLE PROMPT (THREAD-SAFE) =======
def process_prompt(prompt: str, shared_lock: threading.Lock, episodic_memory: EpisodicMemory, image_path: str = None) -> str:
    with shared_lock:
        memories = retrieve_memory(prompt, top_k=5)
        context = "\n".join(memories)
        
        # Add image-based insight if provided
        vision_insight = process_image(image_path, prompt) if image_path else "No image provided."
        
        # Detect and resolve contradictions
        contradictions = detect_contradictions(memories)
        resolution = resolve_contradictions(contradictions, context)
        if "No contradictions" not in resolution:
            print(f"[RESOLUTION in branch] {resolution}")
            store_memory(resolution, {"type": "resolution"})
        
        # Answer the prompt
        full_prompt = f"{prompt}\nResolved context:\n{resolution}\nVision insight: {vision_insight}\n{context}"
        answer = gpt(full_prompt)
        
        # Alignment check
        if not alignment_check(answer):
            print(f"[WARNING] Answer '{answer}' failed alignment check. Discarding.")
            return ""
        
        print(f"[BRANCH PROMPT] {prompt}\n[ANSWER] {answer}")
        
        # Store in both memories
        store_memory(f"Prompt: {prompt}\nAnswer: {answer}", {"type": "insight"})
        episodic_memory.add_event({"content": f"Prompt: {prompt}\nAnswer: {answer}", "timestamp": time.time()})
        return answer

# ======= PERFORMANCE TRACKING =======
class PerformanceTracker:
    def __init__(self):
        self.scores = []
        self.cycles = 0

    def update(self, score: float):
        self.scores.append(score)
        self.cycles += 1
        avg_score = sum(self.scores) / len(self.scores) if self.scores else 0
        print(f"[PERFORMANCE] Cycle {self.cycles}, Score: {score}, Avg: {avg_score:.2f}")

# ======= SELF-IMPROVING LOOP =======
def self_improving_loop(max_cycles: int = 100, max_branches: int = 4, image_path: str = None):
    cycle = 0
    lock = threading.Lock()
    episodic_memory = EpisodicMemory()
    tracker = PerformanceTracker()
    current_objective = INITIAL_OBJECTIVE
    
    while cycle < max_cycles:
        cycle += 1
        print(f"\n=== CYCLE {cycle} ===")
        
        # Retrieve global memories
        past_memories = retrieve_memory(current_objective, top_k=20)
        context = "\n".join(past_memories)
        
        # Update objective dynamically
        current_objective = update_objective(current_objective, past_memories)
        
        # Global contradiction detection and resolution
        contradictions = detect_contradictions(past_memories)
        resolution = resolve_contradictions(contradictions, context)
        if "No contradictions" not in resolution:
            print(f"[GLOBAL RESOLUTION] {resolution}")
            store_memory(resolution, {"type": "global_resolution"})
            context += f"\nResolution: {resolution}"
        
        # Generate branched prompts
        prompts = generate_branched_prompts(context, current_objective, random.randint(2, max_branches))
        
        # Parallel processing
        answers = []
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            futures = [executor.submit(process_prompt, p, lock, episodic_memory, image_path) for p in prompts]
            for future in as_completed(futures):
                if result := future.result():
                    answers.append(result)
        
        # Integrate branch answers
        integrated = gpt(f"Integrate these branch insights:\n{'\n'.join(answers)}\nInto a cohesive update.", 
                         system_prompt="You are an integrator of parallel thoughts.")
        print(f"[INTEGRATION] {integrated}")
        store_memory(integrated, {"type": "integration"})
        episodic_memory.add_event({"content": integrated, "timestamp": time.time()})
        
        # Evaluate and track progress
        achieved, score = evaluate_eureka_progress(past_memories + answers, episodic_memory)
        tracker.update(score)
        if achieved:
            print("EUREKA MOMENT ACHIEVED! Terminating loop.")
            break
        
        time.sleep(random.uniform(5, 15))

if __name__ == "__main__":
    # Example image path (optional, replace with real path for vision input)
    self_improving_loop(image_path="example_image.jpg")