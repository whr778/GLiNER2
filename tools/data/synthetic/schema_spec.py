"""Broad annotation ontology for synthetic base-model training data.

Deliberately BROAD, general-domain labels (not fine-grained) so a base model
learns wide coverage of the five GLiNER2 task types before any downstream
fine-tuning. Event types + roles follow the ACE 2005 / ERE / KAIROS style
(document-level triggers and typed arguments), matching RAMS / WikiEvents /
ACE 2005.

Each domain rotates the generator through a different register (news, business,
legal, ...) so the corpus is stylistically diverse, mirroring the real+synthetic
mix described in the GLiNER2 paper (Zaratiana et al., 2025).

Everything here is plain data. To broaden or retarget the labels, edit these
lists -- no other module hard-codes an ontology.

Pools are deliberately larger than any one document uses: `generate.py` samples a
per-document SUBSET (see `generation.sample_per_doc` in the config), which keeps the
prompt about its old size while the vocabulary a model sees across the corpus grows.
Varying the label set per document is also what teaches span-to-type-description
matching rather than a fixed vocabulary.
"""

from __future__ import annotations

# =============================================================================
# ENTITIES -- 23 -> 125, broad and general-domain
# =============================================================================
ENTITY_TYPES = [
    # people and roles (14)
    "person", "job title", "nationality", "ethnic group", "religious group",
    "political affiliation", "academic degree", "military rank", "occupation",
    "family relationship", "pseudonym", "honorific", "team role", "witness",
    # organizations (12)
    "organization", "company", "government agency", "political party",
    "non-profit organization", "educational institution", "sports team",
    "media outlet", "military unit", "criminal organization", "trade union",
    "regulatory body",
    # places (12)
    "location", "geopolitical entity", "facility", "address", "region",
    "body of water", "mountain", "road", "airport", "landmark", "border",
    "postal code",
    # time (7)
    "date", "time", "duration", "frequency", "age", "era", "deadline",
    # numbers and measures (13)
    "money", "quantity", "percent", "ordinal", "cardinal", "measurement",
    "temperature", "distance", "weight", "area", "speed", "voltage", "dosage",
    # products and works (13)
    "product", "work of art", "book", "film", "song", "software", "brand",
    "model number", "vehicle", "aircraft", "vessel", "video game", "publication",
    # legal (10)
    "law", "court case", "contract", "patent", "regulation", "crime",
    "sentence", "license", "treaty", "verdict",
    # science, medicine, nature (16)
    "disease", "symptom", "medication", "chemical", "organism", "gene",
    "protein", "medical procedure", "anatomical structure", "species",
    "mineral", "clinical trial", "diagnosis", "pathogen", "vaccine", "allergen",
    # technology (10)
    "programming language", "algorithm", "hardware component", "protocol",
    "file format", "database", "operating system", "api", "framework",
    "security vulnerability",
    # finance (9)
    "currency", "stock ticker", "financial instrument", "tax", "credit rating",
    "market index", "interest rate", "fiscal quarter", "budget line",
    # events and misc (8)
    "event", "holiday", "conference", "sports competition", "natural disaster",
    "war", "award", "language", "weapon",
]

# =============================================================================
# RELATIONS — 20 -> 72
# =============================================================================
RELATION_TYPES = [
    # employment and org structure (12)
    "works_for", "founded", "subsidiary_of", "employer_of", "member_of",
    "leader_of", "affiliated_with", "board_member_of", "advisor_to",
    "successor_of", "predecessor_of", "reports_to",
    # location (9)
    "located_in", "headquartered_in", "based_in", "born_in", "died_in",
    "capital_of", "borders", "adjacent_to", "operates_in",
    # ownership and commerce (11)
    "owns", "acquired", "merged_with", "invested_in", "supplier_of",
    "customer_of", "competitor_of", "distributor_of", "licensed_to",
    "manufactured_by", "sold_to",
    # personal (7)
    "spouse_of", "parent_of", "sibling_of", "child_of", "relative_of",
    "colleague_of", "mentor_of",
    # creation and authorship (7)
    "produced_by", "authored_by", "directed_by", "composed_by", "designed_by",
    "developed_by", "published_by",
    # civic and legal (9)
    "citizen_of", "represents", "charged_with", "convicted_of", "sued_by",
    "regulated_by", "signatory_of", "party_to", "testified_against",
    # science and technology (9)
    "treats", "causes", "interacts_with", "derived_from", "measured_in",
    "depends_on", "implements", "supersedes", "compatible_with",
    # miscellaneous (8)
    "participated_in", "won", "sponsored_by", "named_after", "part_of",
    "member_state_of", "allied_with", "opposed_to",
]

# =============================================================================
# EVENTS — 22 -> 56 types, ACE/ERE/KAIROS style, unchanged shape
# =============================================================================
EVENT_ONTOLOGY = {
    # conflict and life (9) — existing, kept verbatim
    "Conflict.Attack": ["Attacker", "Target", "Instrument", "Place", "Time"],
    "Conflict.Demonstrate": ["Demonstrator", "Place", "Time"],
    "Conflict.Ceasefire": ["Party", "Mediator", "Place", "Time"],
    "Life.Die": ["Victim", "Agent", "Instrument", "Place", "Time"],
    "Life.Injure": ["Victim", "Agent", "Instrument", "Place", "Time"],
    "Life.BeBorn": ["Person", "Place", "Time"],
    "Life.Marry": ["Participant", "Place", "Time"],
    "Life.Divorce": ["Participant", "Place", "Time"],
    "Life.Evacuate": ["Evacuee", "Agent", "Origin", "Destination", "Time"],
    # movement (4)
    "Movement.Transport": ["Agent", "Entity", "Origin", "Destination", "Vehicle", "Time"],
    "Movement.Migrate": ["Migrant", "Origin", "Destination", "Time"],
    "Movement.Deport": ["Agent", "Person", "Origin", "Destination", "Time"],
    "Movement.Arrive": ["Traveler", "Destination", "Vehicle", "Time"],
    # transaction (4)
    "Transaction.TransferMoney": ["Giver", "Recipient", "Beneficiary", "Money", "Time"],
    "Transaction.TransferOwnership": ["Buyer", "Seller", "Artifact", "Price", "Time"],
    "Transaction.Donate": ["Donor", "Recipient", "Artifact", "Time"],
    "Transaction.Lend": ["Lender", "Borrower", "Money", "Time"],
    # business (7)
    "Business.StartOrg": ["Agent", "Organization", "Place", "Time"],
    "Business.MergeOrg": ["Organization", "Place", "Time"],
    "Business.DeclareBankruptcy": ["Organization", "Place", "Time"],
    "Business.EndOrg": ["Organization", "Place", "Time"],
    "Business.LaunchProduct": ["Organization", "Product", "Place", "Time"],
    "Business.Layoff": ["Organization", "Employee", "Quantity", "Place", "Time"],
    "Business.Strike": ["Striker", "Organization", "Place", "Time"],
    # personnel (4)
    "Personnel.StartPosition": ["Person", "Organization", "Position", "Time"],
    "Personnel.EndPosition": ["Person", "Organization", "Position", "Time"],
    "Personnel.Elect": ["Person", "Organization", "Position", "Place", "Time"],
    "Personnel.Nominate": ["Person", "Agent", "Position", "Time"],
    # contact (3)
    "Contact.Meet": ["Participant", "Place", "Time"],
    "Contact.Communicate": ["Communicator", "Recipient", "Place", "Time"],
    "Contact.Negotiate": ["Participant", "Topic", "Place", "Time"],
    # justice (8)
    "Justice.Arrest": ["Person", "Agent", "Crime", "Place", "Time"],
    "Justice.ChargeIndict": ["Defendant", "Prosecutor", "Adjudicator", "Crime", "Time"],
    "Justice.TrialHearing": ["Defendant", "Prosecutor", "Adjudicator", "Place", "Time"],
    "Justice.Sentence": ["Defendant", "Adjudicator", "Crime", "Sentence", "Time"],
    "Justice.Sue": ["Plaintiff", "Defendant", "Adjudicator", "Crime", "Time"],
    "Justice.Acquit": ["Defendant", "Adjudicator", "Crime", "Time"],
    "Justice.Appeal": ["Appellant", "Adjudicator", "Crime", "Time"],
    "Justice.Fine": ["Entity", "Adjudicator", "Money", "Crime", "Time"],
    # disaster and public safety (4)
    "Disaster.NaturalDisaster": ["Type", "Affected", "Place", "Time"],
    "Disaster.Accident": ["Vehicle", "Victim", "Place", "Time"],
    "Disaster.Outbreak": ["Disease", "Affected", "Place", "Time"],
    "Disaster.Rescue": ["Rescuer", "Victim", "Place", "Time"],
    # science, health, technology (7)
    "Medical.Diagnose": ["Patient", "Diagnosis", "Physician", "Place", "Time"],
    "Medical.Treat": ["Patient", "Treatment", "Physician", "Time"],
    "Research.Publish": ["Author", "Publication", "Venue", "Time"],
    "Research.Discover": ["Researcher", "Finding", "Place", "Time"],
    "Research.Fund": ["Funder", "Recipient", "Money", "Time"],
    "Tech.Release": ["Organization", "Product", "Version", "Time"],
    "Tech.Breach": ["Attacker", "Target", "Data", "Place", "Time"],
    # government and civic (4)
    "Government.EnactLaw": ["Agent", "Law", "Place", "Time"],
    "Government.Vote": ["Voter", "Proposal", "Result", "Place", "Time"],
    "Government.Sanction": ["Agent", "Target", "Reason", "Time"],
    "Government.Protest": ["Demonstrator", "Target", "Place", "Time"],
    # award and sport (2)
    "Award.ReceiveAward": ["Recipient", "Award", "Awarder", "Place", "Time"],
    "Sport.Compete": ["Competitor", "Competition", "Result", "Place", "Time"],
}

# =============================================================================
# CLASSIFICATION — 3 tasks -> 12
# =============================================================================
CLASSIFICATION_TASKS = {
    "topic": [
        "business", "politics", "technology", "science", "health", "sports",
        "entertainment", "world", "finance", "education", "environment", "legal",
        "travel", "agriculture", "energy", "defense",
    ],
    "sentiment": ["positive", "negative", "neutral"],
    "formality": ["formal", "informal"],
    "urgency": ["routine", "elevated", "urgent", "critical"],
    "audience": ["general public", "expert", "internal", "regulatory"],
    "document_genre": [
        "news report", "opinion", "advertisement", "legal filing",
        "technical documentation", "correspondence", "review", "announcement",
    ],
    "certainty": ["asserted", "hedged", "speculative", "denied"],
    "temporal_orientation": ["past", "present", "future", "timeless"],
    "subjectivity": ["objective", "subjective"],
    "actionability": ["informational", "advisory", "requires action"],
    "risk_level": ["none", "low", "moderate", "high"],
    "language_register": ["technical", "plain", "legal", "colloquial"],
}
MULTI_LABEL_TASKS = {"topic", "audience"}

# =============================================================================
# STRUCTURES — 4 templates -> 14
# =============================================================================
STRUCTURE_TEMPLATES = {
    # existing four, unchanged
    "product": {
        "name": None, "brand": None, "price": None,
        "condition": ["new", "used", "refurbished", "unknown"],
    },
    "person_profile": {
        "name": None, "role": None, "employer": None, "location": None,
    },
    "transaction": {
        "item": None, "amount": None, "buyer": None, "seller": None, "date": None,
    },
    "job_posting": {
        "title": None, "company": None, "location": None,
        "employment_type": ["full-time", "part-time", "contract", "internship", "unknown"],
    },
    # new
    "company_profile": {
        "name": None, "industry": None, "headquarters": None, "founded": None,
        "size": ["startup", "small", "medium", "large", "unknown"],
    },
    "incident_report": {
        "incident_type": None, "location": None, "date": None, "casualties": None,
        "severity": ["minor", "moderate", "severe", "unknown"],
    },
    "clinical_finding": {
        "condition": None, "patient_group": None, "treatment": None, "outcome": None,
    },
    "legal_case": {
        "case_name": None, "court": None, "plaintiff": None, "defendant": None,
        "status": ["filed", "ongoing", "settled", "decided", "unknown"],
    },
    "research_paper": {
        "title": None, "authors": None, "venue": None, "year": None, "topic": None,
    },
    "event_listing": {
        "name": None, "venue": None, "date": None, "organizer": None, "price": None,
    },
    "real_estate_listing": {
        "address": None, "price": None, "bedrooms": None, "area": None,
        "property_type": ["apartment", "house", "commercial", "land", "unknown"],
    },
    "vehicle_listing": {
        "make": None, "model": None, "year": None, "mileage": None, "price": None,
    },
    "financial_report": {
        "organization": None, "period": None, "revenue": None, "net_income": None,
    },
    "software_release": {
        "product": None, "version": None, "release_date": None, "platform": None,
    },
}

# =============================================================================
# DOMAINS — 12 -> 24
# =============================================================================
DOMAINS = [
    # existing twelve
    "breaking news article", "business and financial news",
    "legal notice or court report", "scientific or medical abstract",
    "corporate press release", "social media thread",
    "product listing or e-commerce description",
    "email or professional correspondence", "sports report",
    "government or policy statement", "biographical encyclopedia entry",
    "technology review",
    # new twelve
    "investigative long-form report", "regulatory filing",
    "clinical trial summary", "engineering incident postmortem",
    "conference talk abstract", "customer support transcript",
    "real estate listing", "travel guide entry",
    "opinion column", "internal company memo",
    "patent abstract", "agricultural or environmental bulletin",
    # requested additions — phrased as document REGISTERS, matching the existing
    # convention ("breaking news article", not "a crime"). The register framing is
    # also what keeps generation clear of the cyber/conflict safety classifiers.
    "cybersecurity incident news report",
    "military conflict news dispatch",
    "disaster response situation report",
    "severe weather bulletin",
]

ALL_TASKS = ["entities", "relations", "events", "classifications", "structures"]


# --- Per-document label sampling ------------------------------------------
def sample_labels(rng, tasks, per_doc):
    """This document's SUBSET of each pool, as ``{task: [labels]}``.

    ``rng`` is seeded per document index by the caller, so a run is reproducible and
    the asked-about set can be recovered later.

    Sampled types the document turns out not to contain become NEGATIVES, but only
    for entities. An entity record is a dict (``{type: [spans]}``), so an absent type
    is expressible as ``{type: []}`` and its query is still emitted -- verified
    end-to-end: one present and two absent types gives three queries and one gold
    mention. `relations` and `events` records are LISTS of instances, so an absent
    type simply is not in the list and no query is ever emitted for it. Negatives
    there need a record-format change, not a knob.
    """
    pools = {
        "entities": ENTITY_TYPES,
        "relations": RELATION_TYPES,
        "events": list(EVENT_ONTOLOGY),
        "classifications": list(CLASSIFICATION_TASKS),
        "structures": list(STRUCTURE_TEMPLATES),
    }
    out = {}
    for task in tasks:
        pool = pools.get(task)
        if not pool:
            continue
        k = min(per_doc.get(task, len(pool)), len(pool))
        out[task] = rng.sample(pool, k)
    return out
