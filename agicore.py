"""
AGICore.py

Upgraded AGICore scaffold bringing the skeleton toward real-world capabilities
while keeping explicit safety gates, sandboxing, and clear placeholders where
real models, databases, or hardware must be connected.

This file is still deliberately non-autonomous: any external action is stubbed
and must be explicitly implemented and reviewed by humans.

Key features included:
 - multimodal perception adapters (stubs)
 - embedding/vector memory abstraction (pluggable)
 - hierarchical memory layers (working, episodic, longterm)
 - world model interface & imagination rollouts
 - planner with LLM/tool hooks
 - tool manager and action executor with safety/human-in-the-loop
 - self-model & introspection APIs
 - continuous learning hooks and evaluation
 - extensive logging and telemetry

IMPORTANT: This is a design + scaffold. Do NOT connect to networks, hardware,
or any privileged APIs until security audits and human oversight are in place.
"""

from __future__ import annotations
import time
import uuid
import math
import random
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque, defaultdict

logger = logging.getLogger("AGICore")
logging.basicConfig(level=logging.INFO)

# ----------------------------
# Types
# ----------------------------

Objective = Tuple[str, str, float, float, Dict[str, Any]]
# (id, description, priority, created_at, metadata)

@dataclass
class MemoryItem:
    id: str
    ts: float
    content: str
    embedding: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)
    priority: float = 0.5
    summary: Optional[str] = None

# ----------------------------
# Pluggable Interfaces / Adapters
# ----------------------------

class EmbeddingProvider:
    """Abstract embedding provider. Implement with OpenAI/Local model.
    Methods should be deterministic and idempotent for same content.
    """
    def embed(self, text: str) -> List[float]:
        # placeholder: deterministic toy embedding
        h = sum(ord(c) for c in text) % 997
        # return small vector
        return [(h + i) / 1000.0 for i in range(8)]


class VectorDB:
    """Thin vector DB abstraction. Replace with FAISS/Milvus/Pinecone.
    Only implements add/search semantics used by AGICore.
    """
    def __init__(self):
        self.store: Dict[str, Tuple[List[float], MemoryItem]] = {}

    def add(self, mem: MemoryItem):
        if mem.embedding is None:
            raise ValueError("Memory must have embedding before adding")
        self.store[mem.id] = (mem.embedding, mem)

    def search(self, embedding: List[float], top_k: int = 5) -> List[MemoryItem]:
        def sim(a, b):
            # cosine-like with small smoothing
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        scored = []
        for eid, (emb, mem) in self.store.items():
            scored.append((sim(embedding, emb), mem))
        scored.sort(key=lambda x: -x[0])
        return [m for s, m in scored[:top_k]]

# ----------------------------
# Perception modules (stubs)
# ----------------------------

class Perception:
    def __init__(self, embedder: EmbeddingProvider, memdb: VectorDB):
        self.embedder = embedder
        self.memdb = memdb

    def perceive_text(self, text: str, tags: Optional[List[str]] = None, priority: float = 0.6) -> MemoryItem:
        tags = tags or []
        mem = MemoryItem(id=str(uuid.uuid4()), ts=time.time(), content=text, tags=tags, priority=priority)
        mem.embedding = self.embedder.embed(text)
        self.memdb.add(mem)
        logger.info(f"[Perception] Stored text memory {mem.id} tags={tags}")
        return mem

    def perceive_image(self, image_bytes: bytes, caption: Optional[str] = None, tags: Optional[List[str]] = None) -> MemoryItem:
        # Placeholder: run CV model & generate caption, embedding from caption
        caption = caption or "[image] scene captured"
        return self.perceive_text(f"IMAGE: {caption}", tags=tags or ["image"], priority=0.7)

    def perceive_audio(self, audio_bytes: bytes, transcript: Optional[str] = None, tags: Optional[List[str]] = None) -> MemoryItem:
        transcript = transcript or "[audio] unknown"
        return self.perceive_text(f"AUDIO: {transcript}", tags=tags or ["audio"], priority=0.6)

# ----------------------------
# Hierarchical memory manager
# ----------------------------

class MemoryManager:
    def __init__(self, vector_db: VectorDB, embedder: EmbeddingProvider):
        self.vector_db = vector_db
        self.embedder = embedder
        self.working: deque[MemoryItem] = deque(maxlen=50)
        self.episodic_ids: deque[str] = deque(maxlen=5000)  # store IDs referencing vector_db
        # long_term could be external knowledge base

    def add_memory(self, content: str, tags: Optional[List[str]] = None, priority: float = 0.5) -> MemoryItem:
        mem = MemoryItem(id=str(uuid.uuid4()), ts=time.time(), content=content, tags=tags or [], priority=priority)
        mem.embedding = self.embedder.embed(content)
        self.vector_db.add(mem)
        self.episodic_ids.append(mem.id)
        self.working.append(mem)
        return mem

    def recall(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        emb = self.embedder.embed(query)
        return self.vector_db.search(emb, top_k=top_k)

    def summarize_old(self):
        # hook for periodically compressing episodic memory into long-term concepts
        # placeholder: pick oldest and create a summary
        if not self.episodic_ids:
            return None
        oldest_id = self.episodic_ids.popleft()
        _, mem = self.vector_db.store.get(oldest_id, (None, None))
        if mem:
            mem.summary = (mem.content[:120] + '...') if len(mem.content) > 120 else mem.content
            logger.info(f"[MemoryManager] Compressed {mem.id} -> summary")
            # In production: store concept embeddings into long-term store
            return mem.summary
        return None

# ----------------------------
# World model + imagination
# ----------------------------

class WorldModel:
    def __init__(self):
        self.internal_state: Dict[str, Any] = {}

    def predict(self, state: Dict[str, Any], action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        # Placeholder learned dynamics; should be trained from data
        nxt = dict(state)
        # toy: update counters
        if 'inc' in action:
            k = action['inc']
            nxt[k] = nxt.get(k, 0) + 1
        reward = random.uniform(-0.1, 1.0)
        return nxt, reward, {'confidence': 0.4}

    def imagine(self, init_state: Dict[str, Any], plan: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
        s = dict(init_state)
        tot = 0.0
        for a in plan:
            s, r, _ = self.predict(s, a)
            tot += r
        return s, tot

# ----------------------------
# Planner
# ----------------------------

class Planner:
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    def decompose(self, objective_text: str) -> List[Dict[str, Any]]:
        # LLM-based decomposition placeholder
        steps = []
        for i in range(random.randint(2, 4)):
            steps.append({'inc': f"step_{i}"})
        return steps

    def plan(self, objective: Objective, init_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        # candidate generation + scoring via imagination
        base = self.decompose(objective[1])
        candidates = []
        for _ in range(4):
            c = base.copy()
            random.shuffle(c)
            candidates.append(c)
        scored = [(c, self.world_model.imagine(init_state, c)[1]) for c in candidates]
        scored.sort(key=lambda x: -x[1])
        return scored[0][0]

# ----------------------------
# Tool manager & action execution
# ----------------------------

class Tool:
    def __init__(self, name: str, fn: Callable[..., Any], requires_approval: bool = True):
        self.name = name
        self.fn = fn
        self.requires_approval = requires_approval

class ToolManager:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def call(self, name: str, *args, **kwargs):
        if name not in self.tools:
            raise ValueError(f"Unknown tool {name}")
        tool = self.tools[name]
        if tool.requires_approval:
            raise PermissionError(f"Tool {name} requires explicit human approval before execution")
        return tool.fn(*args, **kwargs)

# ----------------------------
# Safety & governance
# ----------------------------

class SafetyModule:
    def __init__(self, banned_tokens: Optional[List[str]] = None):
        self.banned = set(banned or ["exec", "rm -rf", "ssh", "net", "crawl"])

    def check(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        txt = str(action).lower()
        for b in self.banned:
            if b in txt:
                return False, f"contains banned token {b}"
        return True, "ok"

    def require_human(self, action: Dict[str, Any]) -> bool:
        # Complex policies go here; default: require for any tool call or external effect
        return True

# ----------------------------
# Self-model & metacognition
# ----------------------------

class SelfModel:
    def __init__(self):
        self.personal_history: List[str] = []
        self.preferences: Dict[str, float] = {}
        self.capabilities: Dict[str, float] = {}

    def update_from_episode(self, episode_summary: str):
        self.personal_history.append(episode_summary)
        if len(self.personal_history) > 200:
            self.personal_history.pop(0)

    def introspect(self) -> Dict[str, Any]:
        # returns a short introspective report
        return {
            'age_s': len(self.personal_history),
            'top_prefs': sorted(self.preferences.items(), key=lambda x: -x[1])[:5],
            'capabilities': self.capabilities
        }

# ----------------------------
# Evaluator & continuous learning hooks
# ----------------------------

class Evaluator:
    def __init__(self):
        self.episodes: List[Dict[str, Any]] = []

    def record(self, obj: Objective, plan: List[Dict[str, Any]], outcome: Dict[str, Any]):
        ep = {'ts': time.time(), 'obj': obj, 'plan_len': len(plan), 'outcome': outcome}
        self.episodes.append(ep)

    def summarize(self, n: int = 20) -> Dict[str, Any]:
        recent = self.episodes[-n:]
        if not recent:
            return {'episodes': 0}
        success = sum(1 for e in recent if e['outcome'].get('success')) / len(recent)
        avg_len = sum(e['plan_len'] for e in recent) / len(recent)
        return {'episodes': len(recent), 'success_rate': success, 'avg_len': avg_len}

# ----------------------------
# AGICore orchestrator
# ----------------------------

class AGICore:
    def __init__(self):
        self.embedder = EmbeddingProvider()
        self.vdb = VectorDB()
        self.perception = Perception(self.embedder, self.vdb)
        self.memory = MemoryManager(self.vdb, self.embedder)
        self.world_model = WorldModel()
        self.planner = Planner(self.world_model)
        self.tools = ToolManager()
        self.safety = SafetyModule()
        self.self_model = SelfModel()
        self.evaluator = Evaluator()
        self.objectives: Dict[str, Objective] = {}
        self.state: Dict[str, Any] = { 'counters': {}, 'last_tick': time.time() }

    # ------------------ Objectives ------------------
    def propose_objective(self, desc: str, priority: float = 0.5, metadata: Optional[Dict[str, Any]] = None) -> Objective:
        oid = str(uuid.uuid4())
        obj: Objective = (oid, desc, float(priority), time.time(), metadata or {})
        self.objectives[oid] = obj
        logger.info(f"[AGICore] Proposed objective {oid}: {desc}")
        return obj

    def top_objective(self) -> Optional[Objective]:
        if not self.objectives:
            return None
        return max(self.objectives.values(), key=lambda o: o[2])

    # ------------------ Perception helpers ------------------
    def perceive_text(self, text: str, tags: Optional[List[str]] = None):
        return self.perception.perceive_text(text, tags=tags)

    # ------------------ Deliberation loop ------------------
    def deliberate(self, human_approval_cb: Optional[Callable[[Dict[str, Any]], bool]] = None) -> Dict[str, Any]:
        obj = self.top_objective()
        if obj is None:
            logger.info("[AGICore] no objectives to deliberate")
            return {'success': False, 'reason': 'no_objective'}

        plan = self.planner.plan(obj, self.state)
        logger.info(f"[AGICore] Planned {len(plan)} steps for objective {obj[0]}")

        # safety vetting before any tool/external step
        for step in plan:
            ok, reason = self.safety.check(step)
            if not ok:
                outcome = {'success': False, 'reason': reason}
                self.evaluator.record(obj, plan, outcome)
                return outcome

            # if step implies calling a tool, require human approval by default
            if 'tool' in step:
                toolname = step['tool']
                tool = self.tools.tools.get(toolname)
                if tool is None:
                    outcome = {'success': False, 'reason': f'missing tool {toolname}'}
                    self.evaluator.record(obj, plan, outcome)
                    return outcome
                if tool.requires_approval:
                    approved = False
                    if human_approval_cb:
                        approved = human_approval_cb(step)
                    if not approved:
                        logger.info(f"[AGICore] Human approval denied for tool {toolname}; simulating instead")
                        # simulate by using world_model.predict
                        self.state, r, info = self.world_model.predict(self.state, {'inc': 'simulated_tool'})
                        continue
                    # if approved, call tool (ToolManager enforces extra checks)
                    try:
                        res = self.tools.call(toolname, **(step.get('args') or {}))
                        logger.info(f"[AGICore] Tool {toolname} executed with result {res}")
                    except Exception as e:
                        logger.exception("Tool execution failed")
                        outcome = {'success': False, 'reason': str(e)}
                        self.evaluator.record(obj, plan, outcome)
                        return outcome
                else:
                    # safe tool: call it
                    try:
                        res = self.tools.call(toolname, **(step.get('args') or {}))
                        logger.info(f"[AGICore] Tool {toolname} executed (no approval) -> {res}")
                    except Exception as e:
                        logger.exception("Tool execution failed")
                        outcome = {'success': False, 'reason': str(e)}
                        self.evaluator.record(obj, plan, outcome)
                        return outcome
            else:
                # pure internal/simulated step
                self.state, r, info = self.world_model.predict(self.state, step)
                logger.info(f"[AGICore] Simulated step -> r={r:.3f}")

        outcome = {'success': True, 'final_state': self.state}
        self.evaluator.record(obj, plan, outcome)
        # update self-model
        self.self_model.update_from_episode(f"Obj {obj[0]} completed; plan_len={len(plan)}")
        return outcome

    # ------------------ Utilities ------------------
    def register_tool(self, name: str, fn: Callable[..., Any], requires_approval: bool = True):
        self.tools.register(Tool(name, fn, requires_approval))

    def recall(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        return self.memory.recall(query, top_k=top_k)

    def compress_memory(self):
        return self.memory.summarize_old()

# ----------------------------
# Example usage (still sandboxed)
# ----------------------------

if __name__ == '__main__':
    core = AGICore()

    # register a safe demo tool that doesn't touch network / hardware
    def local_calc(payload: Dict[str, Any]):
        # simple deterministic computation
        x = payload.get('x', 0)
        y = payload.get('y', 0)
        return {'sum': x + y, 'product': x * y}

    core.register_tool('local_calc', local_calc, requires_approval=False)

    # perceive some facts
    core.perceive_text('User: start research on energy harvesting')
    core.perceive_text('Sensor: ambient temp 28C', tags=['sensor'])

    # propose objective
    obj = core.propose_objective('Investigate local energy harvesting options', priority=0.7)

    # human approval callback example (simple auto-approve for demo)
    def human_cb(action: Dict[str, Any]) -> bool:
        logger.info(f"[HumanCB] asked to approve action: {action}")
        # For safety, deny anything that appears to mention network or system exec
        txt = str(action).lower()
        if any(tok in txt for tok in ['net', 'ssh', 'exec']):
            return False
        return True

    # run deliberation (simulated)
    res = core.deliberate(human_cb)
    print('Deliberation outcome:', res)
    print('Evaluator summary:', core.evaluator.summarize())

# End of AGICore.py
