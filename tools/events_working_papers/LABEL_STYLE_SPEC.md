# One label style: Title_Snake for every training label

**Status:** SPEC 2026-10-07, not built. Decisions so far are the user's (section 3); the four review
files in `label_style/` are awaiting the user's `decision` column. Labels are an INPUT to GLiNER2, so
this changes what the model reads: it needs a new label file (v3) and a NEW BASE. No warm start from
eb19 or earlier.

## 1. Why (measured 2026-10-07)

**The unified maps have no style.** `build_label_maps.py` groups spellings that are identical once
case and separators are removed (`fold`), and then picks the winner by count, or by the base corpus's
own spelling. Nothing about the winner's form is enforced. The targets in `unified-full-v2.yaml`:

| category | forms in use (count, example) |
|---|---|
| entities | `Person` 166, `Regulatory Body` 23, `streetAddress` 66, `chemical` 24, `programming_language` 20, `CARDINAL` 6, `Military_Rank` 5, `amino-acid` 2 |
| relations | `member of` 7, `PART-OF` 2, `bind` 1 |
| events | `defendant` 26, `Location` 19 |
| structures | `asteroid_name` 71, `regionName` 3, `CC` 1 |

**The pick destroys word boundaries.** RAMS writes roles and event types squashed and lowercase. Where
RAMS has the most uses, its form becomes canonical. Every model on `unified-full` (eb17-eb20) therefore
reads these English event labels as unsegmented strings:

| group | RAMS, uses | others, uses | map picks |
|---|---|---|---|
| place of employment | `placeofemployment` 214 | `PlaceOfEmployment` 3 (WikiEvents) | squashed |
| judge/court | `judgecourt` 95 | `JudgeCourt` 42 | squashed |
| contact.threaten-coerce.broadcast | 37 | `Contact.ThreatenCoerce.Broadcast` 3 | squashed |
| giver | `giver` 756 | `Giver` 542 | `giver` |
| victim | `victim` 762 | `Victim` 5,127 | `Victim` |

The effect on scores is UNMEASURED. English arguments are the weakest head (eb19: 0.0522 strict F1), and
these are English argument labels, so a measurement belongs in the next base's readout.

**Different files map the same pair in opposite directions.** `unified.yaml` (eb16 and earlier) against
`unified-full.yaml` (eb17+), as strict inversions:
- events: `Law` / `law`, `Voter` / `voter`, `death` / `Death`, `detainee` / `Detainee`;
- entities: `Metric` / `metric`.

A model warm-started across the two files is shown the opposite spelling of a label it learned.

**A live wrong merge.** The events map has `Proposal -> proposal`. Same letters, different concepts:

| spelling | corpus | text it tags |
|---|---|---|
| `Proposal` (ROLE) | cc_news (Haiku) | the measure voted on: "governorship election" |
| `proposal` (event TYPE) | DuEE | 求婚, a marriage proposal |

## 2. The rule

`Title_Snake(label)`:
1. **Non-Latin** labels are unchanged (`人名`).
2. **Dots** are kept as the hierarchy separator. Each dotted segment is styled on its own:
   `Business.StartOrg -> Business.Start_Org`.
3. **Word split** on spaces, `_`, `-`, `/` and camelCase boundaries (`HTTPServer -> HTTP_Server`).
4. **Case.** Each word is capitalised, except ACRONYMS, which are kept. A token is an acronym if it is
   ALL-CAPS and either has <= 3 letters (`PER`, `LOC`, `GPE`) or is not an English word (`NORP`, `MISC`).
   A longer English word in capitals is SHOUTED, not an acronym: `PERSON -> Person`, `PART-OF -> Part_Of`.
   Every all-caps token and its verdict is in `label_style/all_caps_tokens.tsv`.
5. **Join** with `_`: `street address`, `streetAddress` and `STREET-ADDRESS` all become `Street_Address`.

**The style source is the group's boundary-bearing spelling.** Word boundaries in `placeofemployment` cannot
be recovered by rule, so a group is styled from whichever of its spellings carries boundaries
(`PlaceOfEmployment -> Place_Of_Employment`). A group with no such spelling uses the reviewed SQUASHED table
(section 4), or stays one word.

**Scope: every label in every category, not just map targets.** Today a label with one spelling never
enters the map and trains as written. Under v3 the map is TOTAL over the training vocabulary, so nothing
trains raw. **Exception, unchanged:** open-vocabulary corpora (Pile-NER, NuNER, gliner_multilingual,
knowledgator) stay raw through `data.labels_passthrough`; their free-text labels are the lesson.

**Inference applies the same function.** `apply_label_map` maps known spellings. Any label NOT in the map is
styled by the same `Title_Snake` (config.json carries `label_style: title_snake`), so `military rank` typed
in the viewer reaches `Military_Rank`. Checkpoints without `label_style` are unchanged (bit-identical).

## 3. Decisions

**Made by the user, 2026-10-07:**
- Title_Snake everywhere: each word capitalised, joined by `_`.
- Acronyms keep their capitals.
- Event types keep their dots.
- Open-vocabulary corpora stay raw.
- A collision gate.
- Underscores. We move forward; earlier encoders don't matter.

**Proposed here, awaiting the user:**
- The acronym test: <= 3 letters or not an English word.
- The style source: the boundary-bearing spelling.
- The total map.
- Styling at inference.

## 4. Review files (`label_style/`, fill the `decision` column)

| file | rows | what to decide |
|---|---|---|
| `suspect_groups.tsv` | 104 | groups that may be TWO concepts: role+type in events (`Proposal`, `Competition`, `Award`), and ALL-CAPS beside other casings. The text behind every spelling is in `example`. Write `merge`, or `keep apart: <new name>` |
| `squashed_segments.tsv` | 224 (events 79, structures 120, entities 15, relations 10) | the split for each squashed segment. `proposed` comes from a dictionary segmenter; `CHECK` marks doubtful splits, but the flag misses some (`database -> Data_Base`, `timestamp -> Times_Tamp`, `spacecraft -> Space_Craft`), so read every row. Write the styled form, or `one word` |
| `all_caps_tokens.tsv` | 232 | acronym or shouted word, where the verdict is wrong (`LAW`, kaznerd's OntoNotes tag, is kept as an acronym by the <= 3 rule) |
| `structure_dot_pairs.tsv` | 61 | `hotel.name` vs `hotel_name`: fewest dots wins, as today, then styled |

**A collision is a group whose spellings mean different things.** Grouping is by letters only, so this
cannot be detected; it can only be SUSPECTED (role grouped with type, all-caps beside words) and decided by
reading the tagged text. Title_Snake changes only case and separators, so it CANNOT merge two groups that
are not merged today. The collision risk is today's groups.

## 5. Build (trace, code, trace, test)

1. **`label_style.py`:** `title_snake()` plus the acronym test, shared by the generator and `apply_label_map`.
2. **`build_unified_full.py --v3`:**
   - groups as today;
   - then the pinned decisions: SYNONYMS, MERGES, KEEP_APART, and SQUASHED from the reviewed files;
   - then the style source and Title_Snake;
   - writes `labels/unified-full-v3.yaml` as a TOTAL map.
3. **Inference:** `apply_label_map` styles unmapped labels when config.json says `label_style: title_snake`.
4. **Trace** before and after on real records of every category, with the label spellings printed through
   `read_transformed`.

## 6. Gates (each must be able to fail)

1. **Style:** zero v3 targets fail `Title_Snake(t) == t`.
2. **Total:** zero training labels (outside passthrough corpora) reach the model unstyled. Counted through
   `read_transformed` on every split, not on the map.
3. **No loss:** zero label uses lost in any category; the map is CLOSED (no target is a key).
4. **Every suspect has a decision:** zero rows of `suspect_groups.tsv` or `squashed_segments.tsv` without a
   `decision`, or the generator refuses.
5. **Keep-apart holds:** `Proposal` (role) and the DuEE type end on different v3 labels.
6. **Eval agrees with train:** val/test pass through the same map, so gold and predictions share spellings.
   Proven by scoring one checkpoint's predictions before and after on the same records.
7. **Off is unchanged:** a checkpoint without `label_style` maps and decodes bit-identically.

## 7. Cost and order

- **A new base.** v3 is a different vocabulary. eb20, if not yet started, or eb21 trains on
  `labels_file: labels/unified-full-v3.yaml`; models warm-started from eb19 stay on v1.
- **The scores are not comparable** with eb19 on any label-spelling-sensitive key without the same map
  applied to both sides. The blind-test gold is mapped, so the comparison holds if both are scored through
  their own label files; state it in the readout.
- **Measuring the label effect itself** needs two arms on identical data, differing only in the label
  file (a fast A/B, ~$20). Otherwise the new base's English arguments mix the label fix with everything
  else that changed.
