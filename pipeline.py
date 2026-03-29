"""
Operational Knowledge RAG Pipeline — Master (JSON-driven)
==========================================================
Neela Thiagarajan

This pipeline automatically loads ALL How-To JSON files from the
knowledge_base/ folder and builds the corpus dynamically.

To add a new process:
  1. Create a new JSON file in knowledge_base/
  2. Run the pipeline — it picks it up automatically
  3. No code changes needed

Usage:
  python pipeline.py                    # run all demo queries
  python pipeline.py --query "..."      # ask a specific question
  python pipeline.py --interactive      # interactive mode
  python pipeline.py --list             # list loaded docs and chunks
"""

import os
import sys
import json
import glob
import textwrap
import argparse
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import anthropic


# ── KnowledgeChunk ────────────────────────────────────────────────────────

@dataclass
class KnowledgeChunk:
    chunk_id: str
    process: str
    doc_id: str
    section: str
    intent: str
    audience: str
    content: str
    keywords: list[str] = field(default_factory=list)
    business_capability: str = ""
    business_outcome: str = ""
    process_stage: str = ""
    systems: list[str] = field(default_factory=list)
    data_entities: list[str] = field(default_factory=list)
    data_triggers: str = ""
    source: str = ""


# ── Load corpus from JSON files ───────────────────────────────────────────

def load_corpus(kb_folder: str = "knowledge_base") -> tuple[list[KnowledgeChunk], dict]:
    """
    Load all How-To JSON files from the knowledge_base folder.
    Returns (corpus, process_map) where process_map is auto-built
    from all process_keywords in the JSON files.
    """
    corpus = []
    process_map = {}
    docs_loaded = []

    # Find all JSON files
    json_files = glob.glob(os.path.join(kb_folder, "*.json"))

    if not json_files:
        print(f"⚠️  No JSON files found in '{kb_folder}/'")
        print(f"   Create How-To JSON files there and re-run.")
        return corpus, process_map

    for filepath in sorted(json_files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                doc = json.load(f)

            doc_id  = doc.get('doc_id', os.path.basename(filepath))
            process = doc.get('process', 'UNKNOWN')
            systems = doc.get('systems', [])
            data_entities = doc.get('data_entities', [])
            source  = doc.get('source', '')
            sections = doc.get('sections', {})

            # Build process_map from keywords
            for kw in doc.get('process_keywords', []):
                process_map[kw.lower()] = process

            chunks_added = 0
            for section_name, section in sections.items():
                if not section.get('content', '').strip():
                    continue

                chunk = KnowledgeChunk(
                    chunk_id        = f"{doc_id}-{section_name}",
                    process         = process,
                    doc_id          = doc_id,
                    section         = section_name,
                    intent          = section.get('intent', 'LEARN'),
                    audience        = section.get('audience', 'all'),
                    content         = section.get('content', ''),
                    keywords        = section.get('keywords', []),
                    business_capability = section.get('business_capability', ''),
                    business_outcome    = section.get('business_outcome', ''),
                    process_stage       = section.get('process_stage', ''),
                    systems             = systems,
                    data_entities       = data_entities,
                    data_triggers       = section.get('data_triggers', ''),
                    source              = source,
                )
                corpus.append(chunk)
                chunks_added += 1

            docs_loaded.append({
                'file': os.path.basename(filepath),
                'doc_id': doc_id,
                'process': process,
                'chunks': chunks_added
            })

        except Exception as e:
            print(f"⚠️  Failed to load {filepath}: {e}")

    return corpus, process_map, docs_loaded


# ── Routing ───────────────────────────────────────────────────────────────

INTENT_MAP = {
    "how do i": "DO", "how to": "DO", "steps to": "DO", "step by step": "DO",
    "what is": "LEARN", "explain": "LEARN", "difference": "LEARN", "what happens": "LEARN", "why": "LEARN",
    "error": "FIX", "not working": "FIX", "fails": "FIX", "problem": "FIX",
    "troubleshoot": "FIX", "missing": "FIX", "not appearing": "FIX", "not visible": "FIX",
    "not closing": "FIX", "not created": "FIX", "not generating": "FIX",
    "rule": "CONSTRAINT", "critical": "CONSTRAINT", "must": "CONSTRAINT", "cannot": "CONSTRAINT",
    "before i start": "CHECK", "prerequisite": "CHECK", "checklist": "CHECK",
    "what do i need": "CHECK", "before creating": "CHECK",
}


def route_query(query: str, process_map: dict) -> dict:
    q = query.lower()
    intent = None
    for trigger, i in INTENT_MAP.items():
        if trigger in q:
            intent = i
            break
    process = None
    for keyword, p in process_map.items():
        if keyword in q:
            process = p
            break
    return {"intent": intent, "process": process}


# ── Retrieval ─────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, corpus: list, process_map: dict, top_k: int = 3) -> list:
    routing = route_query(query, process_map)
    q_lower = query.lower()
    scored = []

    for chunk in corpus:
        score = 0
        if routing["process"] and chunk.process in (routing["process"], "all"):
            score += 10
        elif not routing["process"]:
            score += 2
        if routing["intent"] and chunk.intent == routing["intent"]:
            score += 8
        for kw in chunk.keywords:
            if kw.lower() in q_lower:
                score += 3
        query_words = set(q_lower.split())
        content_words = set(chunk.content.lower().split())
        score += len(query_words & content_words)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# ── 5-Map Context Envelope ────────────────────────────────────────────────

def build_context_envelope(chunks: list) -> str:
    sections = []
    capabilities = list({c.business_capability for c in chunks if c.business_capability})
    stages       = list({c.process_stage for c in chunks if c.process_stage})
    all_systems  = list({s for c in chunks for s in c.systems})
    all_entities = list({e for c in chunks for e in c.data_entities})
    all_triggers = list({c.data_triggers for c in chunks if c.data_triggers})

    if capabilities or stages:
        ctx = "BUSINESS + PROCESS CONTEXT:"
        if capabilities: ctx += f"\n  Business Capability: {', '.join(capabilities)}"
        if stages:       ctx += f"\n  Process Stage: {', '.join(stages)}"
        sections.append(ctx)
    if all_systems:
        sections.append(f"SYSTEMS INVOLVED: {', '.join(all_systems)}")
    if all_entities or all_triggers:
        ctx = "DATA LAYER:"
        if all_entities: ctx += f"\n  Key Entities: {', '.join(all_entities)}"
        if all_triggers: ctx += f"\n  Data Triggers: {'; '.join(all_triggers)}"
        sections.append(ctx)

    chunk_texts = []
    for c in chunks:
        chunk_texts.append(f"[{c.doc_id} | {c.section.upper()} | {c.intent}]\n{c.content.strip()}")
    sections.append("KNOWLEDGE CONTENT:\n" + "\n\n---\n\n".join(chunk_texts))
    return "\n\n".join(sections)


# ── Response Contracts ────────────────────────────────────────────────────

RESPONSE_CONTRACTS = {
    "DO": {
        "structure": ["Numbered steps", "Required inputs", "Verification step"],
        "format": "numbered list",
        "rules": ["Use numbered steps only", "Be concise and actionable", "End with verification"],
        "avoid": "long explanation, narrative prose",
    },
    "FIX": {
        "structure": ["Problem (symptom)", "Cause", "Fix steps"],
        "format": "labeled sections: Problem / Cause / Fix",
        "rules": ["Start with symptom", "State root cause explicitly", "Provide exact fix steps"],
        "avoid": "vague language, missing the cause layer",
    },
    "CHECK": {
        "structure": ["Checklist of prerequisites"],
        "format": "checkbox-style bullet checklist",
        "rules": ["Use checkbox-style bullets", "Each item independently verifiable"],
        "avoid": "mixing steps with prerequisites",
    },
    "CONSTRAINT": {
        "structure": ["Rule statement", "Impact if violated"],
        "format": "bullet list of rules",
        "rules": ["State each rule directly", "State what happens if violated"],
        "avoid": "softening language",
    },
    "LEARN": {
        "structure": ["Short explanation", "Comparison if applicable"],
        "format": "prose, 3-5 sentences max per concept",
        "rules": ["Focus on clarity", "Keep concise"],
        "avoid": "over-explanation, step-by-step format",
    },
}


def build_system_prompt(intent: Optional[str]) -> str:
    base = (
        "You are a precise operational knowledge assistant for an Enterprise Asset Management system. "
        "Answer based ONLY on the provided context. Do not add information not present in the context.\n\n"
    )
    if intent and intent in RESPONSE_CONTRACTS:
        contract = RESPONSE_CONTRACTS[intent]
        base += (
            f"RESPONSE CONTRACT — Intent: {intent}\n"
            f"Required Structure: {' → '.join(contract['structure'])}\n"
            f"Output Format: {contract['format']}\n"
            f"Rules:\n" + "\n".join(f"  - {r}" for r in contract["rules"]) + "\n"
            f"Avoid: {contract['avoid']}\n\n"
            "Follow this contract exactly."
        )
    return base


# ── Answer query ──────────────────────────────────────────────────────────

def answer_query(query: str, corpus: list, process_map: dict, show_routing: bool = True) -> str:
    routing = route_query(query, process_map)
    chunks  = retrieve_chunks(query, corpus, process_map)

    if not chunks:
        return "No relevant content found for this query."

    context       = build_context_envelope(chunks)
    system_prompt = build_system_prompt(routing["intent"])

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}]
    )

    answer = response.content[0].text

    if show_routing:
        routing_info = (
            f"\n\033[90m  Routing → Process: {routing['process'] or 'universal'} | "
            f"Intent: {routing['intent'] or 'undetected'}\n"
            f"  Retrieved: {', '.join(c.chunk_id for c in chunks)}\033[0m"
        )
    else:
        routing_info = ""

    return f"{answer}{routing_info}"


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Operational Knowledge RAG Pipeline')
    parser.add_argument('--query', '-q', type=str, help='Ask a specific question')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--list', '-l', action='store_true', help='List loaded docs and chunks')
    parser.add_argument('--kb', type=str, default='knowledge_base', help='Knowledge base folder path')
    parser.add_argument('--demo', action='store_true', help='Run demo queries')
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY") and not args.list:
        print("\n❌ ANTHROPIC_API_KEY not set.")
        print("   Windows:   set ANTHROPIC_API_KEY=your-key-here")
        print("   Mac/Linux: export ANTHROPIC_API_KEY=your-key-here\n")
        sys.exit(1)

    # Load corpus
    result = load_corpus(args.kb)
    corpus, process_map, docs_loaded = result

    print("=" * 65)
    print("OPERATIONAL KNOWLEDGE RAG PIPELINE")
    print("Neela Thiagarajan")
    print("=" * 65)
    print(f"\nKnowledge base: {args.kb}/")
    print(f"Docs loaded:    {len(docs_loaded)}")
    print(f"Chunks loaded:  {len(corpus)}")
    print(f"Processes:      {', '.join(sorted(set(c.process for c in corpus)))}")
    print()

    # --list
    if args.list:
        print("LOADED DOCUMENTS:")
        for d in docs_loaded:
            print(f"  {d['doc_id']} ({d['process']}) — {d['chunks']} chunks  [{d['file']}]")
        print()
        return

    if not corpus:
        print("No chunks loaded. Add JSON files to knowledge_base/ and re-run.")
        return

    # --query
    if args.query:
        print(f"Q: {args.query}\n")
        answer = answer_query(args.query, corpus, process_map)
        for line in answer.split('\n'):
            if line.startswith('\033'):
                print(line)
            else:
                print(textwrap.fill(line, width=65) if len(line) > 65 else line)
        print()
        return

    # --interactive
    if args.interactive:
        print("Interactive mode — type a question and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                query = input("Q: ").strip()
                if not query: continue
                if query.lower() in ('quit', 'exit', 'q'): break
                print()
                answer = answer_query(query, corpus, process_map)
                for line in answer.split('\n'):
                    if line.startswith('\033'):
                        print(line)
                    else:
                        print(textwrap.fill(line, width=65) if len(line) > 65 else line)
                print()
            except KeyboardInterrupt:
                print("\nExiting.")
                break
        return

    # Default: demo queries
    demo_queries = []
    for doc in docs_loaded:
        doc_id = doc['doc_id']
        doc_chunks = [c for c in corpus if c.doc_id == doc_id]
        intents_present = {c.intent for c in doc_chunks}
        process = doc['process'].lower().replace('-', ' ')
        if 'DO' in intents_present:
            demo_queries.append(("DO", f"How do I create a {process}?"))
        if 'FIX' in intents_present:
            demo_queries.append(("FIX", f"Something is not working in {process}"))
        if 'CONSTRAINT' in intents_present:
            demo_queries.append(("CONSTRAINT", f"What are the critical rules for {process}?"))
        if 'CHECK' in intents_present:
            demo_queries.append(("CHECK", f"What do I need before starting {process}?"))

    if not demo_queries:
        demo_queries = [
            ("DO",         "How do I create a Task Plan?"),
            ("FIX",        "Checklist items not appearing on work orders"),
            ("CONSTRAINT", "What are the critical rules I must know?"),
            ("CHECK",      "What do I need before I start?"),
            ("LEARN",      "What is a billing relief code?"),
        ]

    for intent_label, query in demo_queries[:8]:
        print(f"{'─' * 65}")
        print(f"Q [{intent_label}]: {query}\n")
        answer = answer_query(query, corpus, process_map)
        for line in answer.split('\n'):
            if line.startswith('\033'):
                print(line)
            else:
                print(textwrap.fill(line, width=65) if len(line) > 65 else line)
        print()

    print("=" * 65)
    print(f"DONE — {len(demo_queries[:8])} queries across {len(docs_loaded)} docs")
    print("=" * 65)
    print()
    print("Run modes:")
    print("  python pipeline.py --interactive   # ask your own questions")
    print("  python pipeline.py --query 'text'  # single question")
    print("  python pipeline.py --list          # show loaded docs")
    print()


if __name__ == "__main__":
    main()
