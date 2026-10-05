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

# DRAFT 2026-10-05, FOR REVIEW -- NOT YET IN ANY PROMPT. One line of meaning per event type, plus
# the boundary against its nearest sibling ("Not: ..."). Measured the same day: Haiku and Sonnet
# agree on events at F1 ~0.40 but on the arguments of a SHARED event at ~0.70, so the
# disagreement is which events exist and their type, and the menu gave the annotator bare names.
# Several types overlap by construction (Demonstrate/Protest, LaunchProduct/Tech.Release, the
# money transfers, Attack/Tech.Breach, Elect/Vote, the movements); each boundary below is a
# DECISION, flagged for review, not a fact about the data.
EVENT_DEFINITIONS = {
    "Conflict.Attack": "A physical act of violence or armed force against people, places or property. "
                       "Not: a cyberattack (Tech.Breach), a threat that is not carried out, or a verbal attack.",
    "Conflict.Demonstrate": "A public gathering, march or rally to express a view. "
                            "Not: a labour stoppage (Business.Strike). REVIEW: same concept as Government.Protest.",
    "Conflict.Ceasefire": "Parties to an armed conflict agree to stop or pause fighting. "
                          "Not: a trade or diplomatic agreement with no fighting (Contact.Negotiate).",
    "Life.Die": "A person dies, from any cause. Not: a death that is only feared or projected.",
    "Life.Injure": "A person is physically harmed but not killed. "
                   "Not: property damage, or harm that is only threatened.",
    "Life.BeBorn": "A person is born. Not: the founding of an organisation (Business.StartOrg).",
    "Life.Marry": "Two people marry. Not: an engagement or a business partnership.",
    "Life.Divorce": "A marriage legally ends. Not: a separation of business partners.",
    "Life.Evacuate": "People are moved out of a place to escape danger. "
                     "Not: routine travel (Movement.Transport) or permanent relocation (Movement.Migrate).",
    "Movement.Transport": "Someone moves people or goods from one place to another. "
                          "Not: emergency evacuation (Life.Evacuate) or forced removal (Movement.Deport).",
    "Movement.Migrate": "People relocate to live somewhere else, by their own choice. "
                        "Not: forced removal (Movement.Deport) or a short trip (Movement.Arrive).",
    "Movement.Deport": "An authority forcibly removes a person from a country or place. "
                       "Not: a voluntary move (Movement.Migrate) or an arrest (Justice.Arrest).",
    "Movement.Arrive": "A person or vehicle reaches a destination. "
                       "Not: the act of moving someone else (Movement.Transport).",
    "Transaction.TransferMoney": "Money passes from one party to another as payment, investment or transfer. "
                                 "Not: a gift (Transaction.Donate), a loan (Transaction.Lend), or research funding (Research.Fund).",
    "Transaction.TransferOwnership": "Ownership of an asset or company changes hands, usually by sale or acquisition. "
                                     "Not: two companies combining as equals (Business.MergeOrg).",
    "Transaction.Donate": "Money or goods are given without expecting return. "
                          "Not: a paid transfer (Transaction.TransferMoney) or a grant for research (Research.Fund).",
    "Transaction.Lend": "Money is lent on the expectation of repayment. "
                        "Not: a payment or investment (Transaction.TransferMoney).",
    "Business.StartOrg": "A company, organisation or venture is founded. "
                         "Not: a new product (Business.LaunchProduct) or a new office of an existing company.",
    "Business.MergeOrg": "Two or more organisations combine into one. "
                         "Not: one buying another outright (Transaction.TransferOwnership).",
    "Business.DeclareBankruptcy": "An organisation formally declares it cannot pay its debts. "
                                  "Not: closing for other reasons (Business.EndOrg).",
    "Business.EndOrg": "An organisation ceases to exist or shuts down. "
                       "Not: a formal insolvency filing (Business.DeclareBankruptcy) or job cuts (Business.Layoff).",
    "Business.LaunchProduct": "A company makes a new physical product or service available. "
                              "Not: a software or version release (Tech.Release), or a pre-order page or announcement of a future launch.",
    "Business.Layoff": "An employer dismisses workers, usually in numbers. "
                       "Not: one person leaving a role (Personnel.EndPosition).",
    "Business.Strike": "Workers stop work collectively to press demands. "
                       "Not: a public protest that is not a work stoppage (Conflict.Demonstrate).",
    "Personnel.StartPosition": "A person takes up a job or role. "
                               "Not: winning an election (Personnel.Elect) or being proposed for a role (Personnel.Nominate).",
    "Personnel.EndPosition": "A person leaves a job or role, by resigning, retiring or being fired. "
                             "Not: mass job cuts (Business.Layoff).",
    "Personnel.Elect": "A person is chosen for a position by a vote. "
                       "Not: a vote on a proposal or law (Government.Vote).",
    "Personnel.Nominate": "A person is proposed or named for a position that is not yet theirs. "
                          "Not: taking up the position (Personnel.StartPosition).",
    "Contact.Meet": "People meet in person. Not: a remote call or message (Contact.Communicate).",
    "Contact.Communicate": "Someone conveys information to someone else, by speech, writing or call. "
                           "Not: every reporting verb ('said', 'told reporters') -- only a communication the text is about.",
    "Contact.Negotiate": "Parties bargain toward an agreement. "
                         "Not: a meeting with no bargaining (Contact.Meet), or an agreed ceasefire (Conflict.Ceasefire).",
    "Justice.Arrest": "Authorities detain a person. Not: a charge without detention (Justice.ChargeIndict).",
    "Justice.ChargeIndict": "A person or organisation is formally accused of a crime. "
                            "Not: a civil lawsuit (Justice.Sue) or an arrest (Justice.Arrest).",
    "Justice.TrialHearing": "A court holds a trial or hearing. Not: the verdict's penalty (Justice.Sentence).",
    "Justice.Sentence": "A court imposes a punishment such as prison time. "
                        "Not: a monetary penalty alone (Justice.Fine).",
    "Justice.Sue": "A party files a civil lawsuit against another. Not: a criminal charge (Justice.ChargeIndict).",
    "Justice.Acquit": "A defendant is found not guilty. Not: charges simply being dropped before trial.",
    "Justice.Appeal": "A party asks a higher court to review a decision.",
    "Justice.Fine": "An authority orders a party to pay a monetary penalty. "
                    "Not: a sanction between states or bodies (Government.Sanction).",
    "Disaster.NaturalDisaster": "A natural hazard occurs: earthquake, flood, storm, wildfire, drought. "
                                "Not: an accident caused by people or machines (Disaster.Accident).",
    "Disaster.Accident": "An unintended crash, collision, fire or failure caused by people or machines. "
                         "Not: a deliberate attack (Conflict.Attack) or a natural hazard (Disaster.NaturalDisaster).",
    "Disaster.Outbreak": "A disease spreads among people or animals. Not: one patient's diagnosis (Medical.Diagnose).",
    "Disaster.Rescue": "People are saved from danger by rescuers. "
                       "Not: a planned evacuation (Life.Evacuate).",
    "Medical.Diagnose": "A patient is found to have a condition. Not: an outbreak across a population (Disaster.Outbreak).",
    "Medical.Treat": "A patient receives medical treatment.",
    "Research.Publish": "A work is published: a paper, book, article or story. Not: a product release (Tech.Release).",
    "Research.Discover": "Researchers find something new. Not: publishing a known result (Research.Publish).",
    "Research.Fund": "Money is granted for research. Not: a commercial payment (Transaction.TransferMoney).",
    "Tech.Release": "A software product, version or technology is released. "
                    "Not: a physical product launch (Business.LaunchProduct).",
    "Tech.Breach": "Unauthorised access to computer systems or data: a hack, intrusion or data leak. "
                   "Not: a physical attack (Conflict.Attack).",
    "Government.EnactLaw": "A government passes or brings a law or regulation into force. "
                           "Not: a vote that does not enact anything (Government.Vote).",
    "Government.Vote": "A body or electorate votes on a proposal. Not: electing a person (Personnel.Elect).",
    "Government.Sanction": "A state or international body imposes penalties on another state, body or person. "
                           "Not: a court's fine (Justice.Fine).",
    "Government.Protest": "Public opposition directed at a government or policy. "
                          "REVIEW: same concept as Conflict.Demonstrate -- merge one into the other, or keep this only for protests whose Target is named.",
    "Award.ReceiveAward": "A person or organisation is given an award or prize.",
    "Sport.Compete": "Competitors take part in a sporting contest. Not: a business competition.",
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
