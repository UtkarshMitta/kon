"""
Tier classification rubric for the Mistral Query Router.
Defines complexity tiers, criteria, and seed examples used for
both synthetic data generation and documentation.


TIER_RUBRIC = {
    "small": {
        "tier": 1,
        "label": "small",
        "target_model": "ministral-3b-latest",
        "description": "Simple, factual, or trivial queries that any small model can handle.",
        "criteria": [
            "Single-hop factual lookups (capitals, dates, definitions)",
            "Basic arithmetic (add, subtract, multiply, divide)",
            "Greetings and simple conversational turns",
            "Yes/No questions with obvious answers",
            "Simple format conversions (units, temperatures)",
            "Spelling or grammar of a single word",
        ],
        "seed_examples": [
            "What is the capital of France?",
            "What is 2 + 2?",
            "Hello, how are you?",
            "Is water wet?",
            "Convert 100 Fahrenheit to Celsius",
            "What year did World War II end?",
            "How do you spell 'necessary'?",
            "What color is the sky?",
            "What is the square root of 4?",
            "Who wrote Romeo and Juliet?",
            "How many days are in a week?",
            "What does 'RSVP' stand for?",
            "Name the largest planet in our solar system.",
            "What is the boiling point of water?",
            "Goodbye, have a nice day!",
        ],
    },
    "medium": {
        "tier": 2,
        "label": "medium",
        "target_model": "mistral-small-latest",
        "description": "Moderate complexity queries requiring some reasoning, synthesis, or structured output.",
        "criteria": [
            "Summarize a short text or article",
            "Translate a paragraph between languages",
            "Explain a well-known concept in 2–3 sentences",
            "Simple coding tasks (write a function, fix a syntax error)",
            "List-based outputs (top 5, pros and cons)",
            "Basic data extraction or reformatting",
            "Multi-step arithmetic or word problems",
            "Simple creative writing (haiku, short poem)",
        ],
        "seed_examples": [
            "Summarize the plot of The Great Gatsby in 3 sentences.",
            "Translate 'The quick brown fox jumps over the lazy dog' to French.",
            "What are the pros and cons of remote work?",
            "Write a Python function that reverses a string.",
            "Explain what photosynthesis is in simple terms.",
            "List the top 5 most populated countries.",
            "What is the difference between HTTP and HTTPS?",
            "Write a haiku about autumn.",
            "If a train travels 60 mph for 2.5 hours, how far does it go?",
            "Convert this JSON to a Python dictionary: {\"name\": \"Alice\", \"age\": 30}",
            "What are three healthy breakfast options?",
            "Explain the difference between a list and a tuple in Python.",
            "Rewrite this sentence in passive voice: 'The cat chased the mouse.'",
            "What is the time complexity of binary search?",
            "Give me a recipe for scrambled eggs.",
        ],
    },
    "large": {
        "tier": 3,
        "label": "large",
        "target_model": "mistral-medium-latest",
        "description": "Complex queries requiring multi-step reasoning, analysis, or substantial coding.",
        "criteria": [
            "Debug a non-trivial code snippet with logical errors",
            "Compare and contrast two concepts with nuance",
            "Multi-paragraph explanations with examples",
            "Code that requires understanding of algorithms or data structures",
            "Analysis of trade-offs in technical decisions",
            "Chain-of-thought reasoning problems",
            "Writing with specific constraints (tone, audience, structure)",
            "Data analysis or interpretation questions",
        ],
        "seed_examples": [
            "Debug this Python code that's supposed to find duplicates in a list but returns wrong results:\ndef find_dupes(lst):\n    seen = []\n    for i in lst:\n        if i in lst:\n            seen.append(i)\n    return seen",
            "Compare and contrast REST APIs vs GraphQL. When would you choose each?",
            "Explain how a hash map works internally, including collision resolution strategies.",
            "Write a Python function to find the longest common subsequence of two strings.",
            "Analyze the trade-offs between SQL and NoSQL databases for an e-commerce platform.",
            "A farmer has 100 meters of fencing. What dimensions maximize the enclosed rectangular area? Show your work.",
            "Write a technical blog post introduction about microservices vs monolithic architecture.",
            "Implement a binary search tree with insert, delete, and search operations in Python.",
            "What are the SOLID principles in software engineering? Explain each with an example.",
            "Given a dataset of customer purchases, describe how you would build a recommendation system.",
            "Explain the CAP theorem and its implications for distributed systems.",
            "Write an async Python function that fetches data from 3 APIs concurrently.",
            "How does garbage collection work in Java vs Python? Compare approaches.",
            "Design the database schema for a social media application with posts, comments, and likes.",
            "Explain backpropagation in neural networks step by step with a simple example.",
        ],
    },
    "xlarge": {
        "tier": 4,
        "label": "xlarge",
        "target_model": "mistral-large-latest",
        "description": "Expert-level queries requiring deep reasoning, novel problem-solving, or extensive generation.",
        "criteria": [
            "Mathematical proofs or formal logic",
            "System design for large-scale applications",
            "Research-level analysis or literature review",
            "Novel algorithm design or optimization",
            "Long-form content generation (essays, reports, documentation)",
            "Multi-domain synthesis (combining knowledge from different fields)",
            "Ambiguous or open-ended problems requiring creativity + expertise",
            "Security analysis, threat modeling, or architectural review",
        ],
        "seed_examples": [
            "Prove that the square root of 2 is irrational.",
            "Design a scalable real-time chat system that handles 10 million concurrent users. Include architecture diagram description, technology choices, and failure handling.",
            "Write a comprehensive literature review on transformer architectures in NLP, covering attention mechanisms, positional encoding variants, and recent efficiency improvements.",
            "Design an algorithm to solve the traveling salesman problem for up to 20 cities using dynamic programming with bitmask, and analyze its complexity.",
            "Write a 2000-word essay analyzing the ethical implications of artificial general intelligence, covering alignment, economic displacement, and governance.",
            "Given a legacy monolithic e-commerce application, create a detailed migration plan to microservices including service boundaries, data migration strategy, and rollback plan.",
            "Explain the proof of the Fundamental Theorem of Calculus and its connection to Riemann integration.",
            "Perform a security threat model for a healthcare application handling PHI data under HIPAA, including attack vectors, mitigations, and compliance checklist.",
            "Design a distributed consensus protocol that improves on Raft for geo-replicated databases. Describe the algorithm, prove its safety properties, and analyze liveness.",
            "Write a comprehensive technical specification for a programming language with dependent types, including syntax, type-checking rules, and example programs.",
            "Analyze the complexity classes P, NP, and NP-Complete. Explain the P vs NP problem and discuss the implications of its resolution.",
            "Create a full curriculum for teaching machine learning to software engineers, including 12 weekly modules with learning objectives, readings, and projects.",
            "Design a recommendation engine that combines collaborative filtering, content-based filtering, and knowledge graphs. Provide the full system architecture.",
            "Write a research proposal for using reinforcement learning to optimize energy consumption in data centers.",
            "Explain the mathematical foundations of public-key cryptography, including RSA and elliptic curve cryptography, with proofs of correctness.",
        ],
    },
}

# ──────────────────────────────────────────────
# Query categories for diverse generation
# ──────────────────────────────────────────────
QUERY_CATEGORIES = [
    "mathematics",
    "coding_and_programming",
    "science_and_technology",
    "creative_writing",
    "business_and_finance",
    "education_and_learning",
    "health_and_medicine",
    "history_and_geography",
    "language_and_translation",
    "data_analysis",
    "system_design",
    "debugging",
    "general_knowledge",
    "philosophy_and_ethics",
    "legal_and_compliance",
]


def get_seed_examples_flat():
    #Return all seed examples as a flat list of (query, tier_label) tuples.
    examples = []
    for tier_label, rubric in TIER_RUBRIC.items():
        for query in rubric["seed_examples"]:
            examples.append((query, tier_label))
    return examples


def get_tier_description(tier_label: str) -> str:
    #Get a human-readable description for a tier.
    rubric = TIER_RUBRIC.get(tier_label)
    if not rubric:
        raise ValueError(f"Unknown tier: {tier_label}")
    criteria_str = "\n".join(f"  - {c}" for c in rubric["criteria"])
    return f"Tier {rubric['tier']} ({tier_label}): {rubric['description']}\nCriteria:\n{criteria_str}"


if __name__ == "__main__":
    # Print rubric summary
    print("=" * 60)
    print("MISTRAL QUERY ROUTER - TIER RUBRIC")
    print("=" * 60)
    for label in ["small", "medium", "large", "xlarge"]:
        print(f"\n{get_tier_description(label)}")
        print(f"  Seed examples: {len(TIER_RUBRIC[label]['seed_examples'])}")
    print(f"\nTotal seed examples: {len(get_seed_examples_flat())}")
    print(f"Query categories: {len(QUERY_CATEGORIES)}")



"""