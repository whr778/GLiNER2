# Arguments inside the trigger

**Status:** measured 2026-11-20. A count of gold spans, so the same file reprints the same fraction. ACE 2005 is not on this machine, and its inside share is not in the table.

This is the record for annotation guidelines. The inside share of a corpus is fixed by how that corpus defines the trigger. A training mix that never says which definition it used is mixing two tasks under one role label.

## 1. Definition

An argument is **inside** when both it and a trigger of the same event carry a non-empty character span, and the argument span is contained in that trigger. Edges count. A span equal to the trigger counts as inside. Containment in any one trigger of that event is enough.

An argument is **outside** when both spans exist and the argument is not contained in any trigger of that event. A partial overlap is outside: the argument may start inside the trigger and end past it, or start before the trigger and end inside it.

An argument is **unmeasured** when the argument has no integer character span, or the event has no trigger span. String gold is unmeasured. Unmeasured arguments stay out of the rate.

The inside share is `inside / (inside + outside)`. It is a share of placed gold spans, not an F1.

The test is character offsets on the gold, and it is this test:

```23:35:tools/data/measure_argument_inside_trigger.py
def span_of(value) -> Optional[Span]:
    """``(start, end)`` when ``value`` is an offset dict with a non-empty span."""
    if not isinstance(value, dict):
        return None
    start, end = value.get("start"), value.get("end")
    if isinstance(start, int) and isinstance(end, int) and start < end:
        return start, end
    return None


def contained(argument: Span, trigger: Span) -> bool:
    """True when ``argument`` lies inside ``trigger``, edges included."""
    return trigger[0] <= argument[0] and argument[1] <= trigger[1]
```

A span counts only when `entity` (and the trigger) is a dict with integer `start` and `end` and `start < end`. Spans on `coref_clusters` are not read.

## 2. How to count it

The command takes a JSONL path as-is. A stem with no extension expands to `{stem}.jsonl` and `{stem}.*.jsonl`.

```bash
uv run python tools/data/measure_argument_inside_trigger.py data/wojood_hadath.train.jsonl
uv run python tools/data/measure_argument_inside_trigger.py data/casie_off.train.jsonl data/cmnee_ner_off.train.jsonl
uv run python tools/data/measure_argument_inside_trigger.py data/wojood_hadath
```

Per-role lines are printed only for roles that have at least one placed span. A file of string gold prints a single unmeasured total and no role lines.

## 3. The two geometries

**Inside.** The trigger is the event mention, and the role is a piece of that mention. Hadath train, `hasAgent`: the trigger is `اتفاقية الأمم المتحدة لحظر استعمال وتخزين وإنتاج ونقل الألغام المضادة للأفراد` (characters 7–84) and the argument is `الأمم المتحدة` (15–28). `hasLocation` `اوسلو` sits inside `اتفاقية اوسلو 1`. `hasDate` `1987` sits inside `الانتفاضة الأولى 1987`.

**Outside.** The trigger is the predicate, and the argument is another mention. CASIE train, `Place`: the trigger is `ransomware campaigns` (1245–1265) and the argument is `New Zealand` (1211–1222), earlier in the same sentence. Every placed CASIE argument is disjoint from its trigger.

**Equal span, which counts as inside.** WikiEvents offset train has four inside arguments, and each one is the trigger word used again as the argument: `injured` / Victim, `jail` / Place (twice), `bomb` / ExplosiveDevice. That is 4/4,435 (0.09%), and it is span equality, not a participant nested in a longer trigger. Hadath train has one equal span in 1,533 placed arguments.

**Partial overlap, which counts as outside.** The argument overlaps the trigger and sticks out of it. Hadath train has 4 of these in 1,533 placed arguments. One is the trigger `انتفاضة 2011` against the `hasDate` argument `أعقاب انتفاضة 2011`. CMNEE offset train has 205 of 62,284 (0.33%). The usual shape there is a two-character trigger sitting inside a longer argument, such as trigger `练习` inside `训练练习`. CASIE offset train and WikiEvents offset train have zero partial overlaps.

## 4. Counts

Cross-corpus comparison is train files only. The CASIE, CMNEE, and WikiEvents offset files on disk are the train splits. Their val and test stay string gold, so they cannot enter this table. Hadath's three splits together are the figure already in the catalog.

| Corpus | Language | Placed arguments | Inside share | Equal spans | Partial overlap | Disjoint | Unmeasured |
|---|---|---:|---:|---:|---:|---:|---:|
| Wojood Hadath, all splits | Arabic | 2,477 | 1,814/2,477 (73.23%) | — | — | — | 0 |
| Wojood Hadath train | Arabic | 1,533 | 1,145/1,533 (74.69%) | 1 | 4 | 384 | 0 |
| CASIE offset train | English | 17,911 | 0/17,911 (0.00%) | 0 | 0 | 17,911 | 81 |
| CMNEE offset train | Chinese | 62,284 | 2/62,284 (0.003%) | 0 | 205 | 62,077 | 289 |
| WikiEvents offset train | English | 4,435 | 4/4,435 (0.09%) | 4 | 0 | 4,431 | 0 |

Hadath's inside share is high on every role, on the three splits together: `hasLocation` 703/811 (86.68%), `hasDate` 866/1,290 (67.13%), `hasAgent` 245/376 (65.16%). Train alone: `hasLocation` 427/486 (87.86%), `hasDate` 551/802 (68.70%), `hasAgent` 167/245 (68.16%). Dropping one role does not remove the geometry.

CMNEE's two inside arguments are single characters inside a short trigger: `核` inside `核试验` (Equipment), and `油` inside `加过油` (Materials). They are real nesting, and they are two cases.

DuEE train and RAMS train have arguments and no character spans: 28,875 and 17,026 unmeasured, inside share unknown. `maven`, `docee`, `docee_zh`, `turkish_event`, `events_biotech`, `docfee`, and `scierc` train files returned inside 0/0 and unmeasured 0. The command found no argument dicts under `output.events`, so those files are absent from the table.

## 5. What sets the share

The share follows the annotation's definition of the trigger.

Hadath's source JSON has no offsets. `tools/data/convert_wojood_hadath.py` writes a span only when the trigger occurs once in the sentence, and it places an argument inside that trigger when that landing is unique. The trigger it writes is the event mention, the named occurrence. The participants are pieces of that mention, which is why the inside share is about three quarters, in Arabic, on agent, location, and date alike.

CASIE, CMNEE, and WikiEvents mark the trigger as the predicate and the argument as a separate mention. Their inside share is about zero in English news, English cybersecurity, and Chinese news. The same zero appears in both languages that use the predicate definition. The high share appears in the one corpus whose trigger is the event mention.

Subject matter does not set it either. Hadath's events are named occurrences (a battle, an election, a disaster, a holiday). WikiEvents and CMNEE are news. CASIE is cybersecurity. The three predicate corpora agree with each other, and Hadath does not agree with them.

ACE 2005 uses one anchor-versus-mention scheme for English, Arabic, and Chinese (`tools/data/convert_ace2005.py`). That is the predicate geometry. The rate on the licensed splits has not been counted here.

## 6. A file with no offsets

String gold is a legal GLiNER2 shape, and it is unmeasured by this command. `wikievents.train` is the control already in the catalog: inside 0/0, unmeasured 4,435, because its triggers and argument entities are bare strings. The offset twin, `wikievents_off.train`, is the row in the table above.

An edit that drops `start` and `end` produces the same report, including a file a coworker saved without the offsets. The command then prints all unmeasured and an inside share of 0/0. That report means the spans are absent. It does not mean the arguments sit outside the trigger.

On an ACE conversion, `coref_clusters` still carry `{"text", "start", "end"}` when `--offsets` was left off. Event triggers and argument entities are strings in that file. This command does not read the clusters, so the file still comes back all unmeasured. The rebuild in `tools/data/run_all_converters.sh` passes `--offsets`.

## 7. What a guideline has to decide

`ENGLISH_ANNOTATION_SPEC.md` §7 still lists argument extent as an open decision. These are the facts that decision has to absorb.

1. Say what the trigger is. A predicate trigger produces outside arguments. An event-mention trigger produces inside arguments. Those are different supervision signals for the same role name. Hadath train is 1,145/1,533 (74.69%) inside. CASIE offset train is 0/17,911 (0.00%) inside. A mix of the two trains one argument head on both geometries unless the guideline names the one it wants.

2. Count the inside share on the offset file before the corpus joins a training mix, and write the fraction down. Language and domain do not predict it. Section 4 is the method: the command in §2, train split against train split, unmeasured kept out of the rate.

3. Keep integer character offsets on the trigger and on `entity`. A guideline that emits strings, or a later edit that strips the offsets, makes the balance unmeasurable. Cluster offsets are not a substitute.

4. If the guideline allows an argument to stick out of the trigger, count that partial overlap on its own. This definition files it under outside. It is 205/62,284 (0.33%) of CMNEE offset train and 4/1,533 (0.26%) of Hadath train, and it is zero on CASIE and WikiEvents offset train. A target mix that wants "the argument may contain the trigger" is asking for this class.

5. Say whether a span equal to the trigger is an argument. This definition counts it as inside. WikiEvents' entire inside count is that case. Treating 4/4,435 (0.09%) as nested participants overstates what those four rows are.

6. Check every role. Hadath is high on `hasAgent`, `hasLocation`, and `hasDate`. A guideline that fixes one role and leaves the others nested has not chosen a geometry.

## 8. Where else this is written

The definition and the command live in `tools/data/measure_argument_inside_trigger.py`. The Hadath placement rule is in `tools/data/convert_wojood_hadath.py`. The 1,814/2,477 (73.23%) count is in `tools/train/config/wojood/wojood-hadath-eb23a.yaml` and in the 2026-11-20 catalog row for the command. The cross-corpus train table, the equal-span and partial-overlap splits, and the guideline consequences are in this paper.
