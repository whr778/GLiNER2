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
"""

from __future__ import annotations

# --- Entities: broad, general-domain types --------------------------------
ENTITY_TYPES = [
    "person", "organization", "location", "geopolitical entity", "facility",
    "product", "event", "work of art", "law", "language", "nationality",
    "date", "time", "money", "quantity", "percent", "ordinal",
    "job title", "vehicle", "weapon", "disease", "chemical", "organism",
]

# --- Relations: broad binary relations ------------------------------------
RELATION_TYPES = [
    "works_for", "founded", "subsidiary_of", "located_in", "headquartered_in",
    "part_of", "owns", "acquired", "member_of", "affiliated_with",
    "spouse_of", "parent_of", "sibling_of", "employer_of", "produced_by",
    "based_in", "citizen_of", "leader_of", "competitor_of", "supplier_of",
]

# --- Events: ACE/ERE/KAIROS-style types with typed argument roles ----------
# Each entry: event_type -> allowed argument roles. Broad top-level ontology.
EVENT_ONTOLOGY = {
    "Conflict.Attack": ["Attacker", "Target", "Instrument", "Place", "Time"],
    "Conflict.Demonstrate": ["Demonstrator", "Place", "Time"],
    "Life.Die": ["Victim", "Agent", "Instrument", "Place", "Time"],
    "Life.Injure": ["Victim", "Agent", "Instrument", "Place", "Time"],
    "Life.BeBorn": ["Person", "Place", "Time"],
    "Movement.Transport": ["Agent", "Entity", "Origin", "Destination", "Vehicle", "Time"],
    "Transaction.TransferMoney": ["Giver", "Recipient", "Beneficiary", "Money", "Time"],
    "Transaction.TransferOwnership": ["Buyer", "Seller", "Artifact", "Price", "Time"],
    "Business.StartOrg": ["Agent", "Organization", "Place", "Time"],
    "Business.MergeOrg": ["Organization", "Place", "Time"],
    "Business.DeclareBankruptcy": ["Organization", "Place", "Time"],
    "Personnel.StartPosition": ["Person", "Organization", "Position", "Time"],
    "Personnel.EndPosition": ["Person", "Organization", "Position", "Time"],
    "Personnel.Elect": ["Person", "Organization", "Position", "Place", "Time"],
    "Contact.Meet": ["Participant", "Place", "Time"],
    "Contact.Communicate": ["Communicator", "Recipient", "Place", "Time"],
    "Justice.Arrest": ["Person", "Agent", "Crime", "Place", "Time"],
    "Justice.ChargeIndict": ["Defendant", "Prosecutor", "Adjudicator", "Crime", "Time"],
    "Justice.TrialHearing": ["Defendant", "Prosecutor", "Adjudicator", "Place", "Time"],
    "Justice.Sentence": ["Defendant", "Adjudicator", "Crime", "Sentence", "Time"],
    "Disaster.NaturalDisaster": ["Type", "Affected", "Place", "Time"],
    "Award.ReceiveAward": ["Recipient", "Award", "Awarder", "Place", "Time"],
}

# --- Classification: task -> candidate label vocabulary --------------------
# The model only picks the true label(s); the full vocabulary is fixed here so
# every record shares a consistent candidate set per task.
CLASSIFICATION_TASKS = {
    "topic": [
        "business", "politics", "technology", "science", "health", "sports",
        "entertainment", "world", "finance", "education", "environment", "legal",
    ],
    "sentiment": ["positive", "negative", "neutral"],
    "formality": ["formal", "informal"],
}
MULTI_LABEL_TASKS = {"topic"}  # tasks that may carry >1 true label

# --- Structures: hierarchical templates (field -> optional choice set) ------
# A field mapped to a list is a choice field (value must be one of the choices);
# a field mapped to None is a free-text extractive field.
STRUCTURE_TEMPLATES = {
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
}

# --- Domains: rotate the generator across registers ------------------------
DOMAINS = [
    "breaking news article",
    "business and financial news",
    "legal notice or court report",
    "scientific or medical abstract",
    "corporate press release",
    "social media thread",
    "product listing or e-commerce description",
    "email or professional correspondence",
    "sports report",
    "government or policy statement",
    "biographical encyclopedia entry",
    "technology review",
]

# --- Task keys understood by the generator ---------------------------------
ALL_TASKS = ["entities", "relations", "events", "classifications", "structures"]
