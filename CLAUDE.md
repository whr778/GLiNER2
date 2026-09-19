## VERY IMPORTANT

- Be simple. Approach tasks in a simple, incremental way.
- Work incrementally ALWAYS. Small simple steps. Validate and check each increment before moving on.
- Use LATEST apis as of NOW.
- Label every metric with its type, in tables and in prose: F1, NRMSE, RMSE, TVD, MSE.
- Instead of using /private/tmp use /Volumes/Development/tmp which will persist across reboots.
- Periodically cleanup and compact /Volumes/Development/tmp.

## MANDATORY CODE STYLE

- Do not overengineer. Do not program defensively. Use exception managers only when necessary.
- Use `uv` as Python package manager. Always `uv run xxx` never `python3 xxx`, always `uv add xxx` never `pip install xxx`
- Favor clear, concise docstring comments. Be sparing with comments outside docstrings.
- Favor short modules, short methods and functions. Name things clearly.
- Never use emojis in code or in print statements or logging.
- Keep README.md concise.

## Important -- debugging and fixing

- When troubleshooting problems, ALWAYS identify root cause BEFORE fixing
- Reproduce consistently.
- PROVE THE PROBLEM FIRST - don't guess.
- Try one test at a time. Be methodical.
- Don't jump to conclusions. Don't apply workarounds.
- Don't trust comments. Check the code and if the comments don't reflect the code, don't use them.
- Fully review the documents and code before recommending and making changes.

## EXPERIMENTS AND MEASUREMENT

- EVERY EXPERIMENT GOES IN THE CATALOG. `tools/events_working_papers/EXPERIMENT_CATALOG.md`
  is the chronological record, newest at the top. Add a row when the run ENDS -- with its
  branch era, purpose, where it ran, the OUTCOME, the real billed cost, and any working-paper
  link -- and mark it RUNNING at launch if it will take hours. Record nulls, negatives and
  runs that died: they are the majority and they are the point, and an absent row reads as an
  experiment nobody ran. Quote the measured floor beside any delta, and reference the paper
  rather than restating its argument.
- A GATE MUST BE ABLE TO FAIL. Before trusting any check, ask what it would print if the
  thing it guards were broken. Real cases: a form gate scored best-over-range rewarded
  indiscriminate firing and inverted a verdict; a gate that admitted nothing scored a
  PERFECT false-positive rate and hid for two runs; `create_repo(exist_ok=True)` passed as
  a "write test" for a read-only token; a capture pattern of `[0-9.e+-]*` could not match
  `nan`, so the gate printed the word `norm`; a credential check printed its failure and
  let the run continue.
- PAIR EVERY FORM GATE WITH A CORRECTNESS COMPANION. Counting firings is not counting hits.
- PROVE THE TREATMENT APPLIED, from inside the run. Emit one deterministic line per arm
  showing the treatment is live -- and emit it AFTER logging exists, from a point where the
  thing has actually happened. A once-only flag captured a pre-injector forward, a
  composition line logged before `__getitem__`, and a dropout line logged before
  `_setup_logging`; all three were gates that could not fail. Note that `__getitem__` runs
  in forked workers, so it cannot be the source of a printed line.
- NO ARM WITHOUT A CONTROL, and prove the arms DIFFER. "+0.049 precision" with no no-menu
  baseline became -30% recall once measured. Two arms whose caps both exceeded the corpus
  ran the identical data and their identical failure read as reproduction.
- A DELTA IS NOT A RESULT WITHOUT A NOISE FLOOR. Relations carry a +/-0.041 run-to-run
  floor and single runs +/-0.02; quote the floor beside the delta or do not claim the win.
- PICK ON VALIDATION, SCORE THE BLIND TEST ONCE. Sweeping on test and quoting the best is
  fitting the test set. A "+0.049 win" on this programme turned out to be a
  checkpoint-selection defect.
- NEVER COMPARE ACROSS A CHANGED TEST SET. If the eval data moved, the delta is confounded
  and the comparison is void -- say so rather than reporting the number.
- MEASURE A PROPERTY BEFORE CLASSIFYING ON IT. `candidate_pool` was assumed structural for
  months and blocked a real experiment; building the head both ways showed 340 tensors
  either side, none added, removed or reshaped. "Changes the shapes" and "invalidates a
  warm start scientifically" are different claims -- do not let one justify the other.
- DEBUG WITH A CONTROL ARM TOO. When hunting a non-finite, flag only modules whose OUTPUT is
  non-finite while ALL INPUTS WERE FINITE; without that conjunction a propagating NaN names
  dozens of modules and localizes nothing. Vary ONE axis at a time -- steps, device,
  precision -- because a single-step repro will miss a fault that first appears at step 1.

## GPU runs (tools/lambda/)

- I do not pay for idle time. Every job carries three independent stops: a `timeout` on the
  job, a detached hard-deadline watchdog, and terminate on the normal path.
- PROVE CREDENTIALS BEFORE SPENDING. Ship `$HF_TOKEN` from the environment -- the library's
  default cache token cannot upload -- and probe it with a real upload-then-delete, locally
  and again on the box. An empty token surfaces only as `Illegal header value b'Bearer '`.
- PUBLISH METRICS BEFORE THE MODEL: the metrics are the finding and are kilobytes; the model
  is 600MB and can fail on quota. Verify against the Hub's file list, never a clean return.
  A named-but-never-written artefact is a FAILED job, not an empty publish.
- Check ECC at launch, `pkill -9` a runner (its trap catches SIGTERM), and filter every pipe
  through a login shell -- the greeting goes to stdout and has corrupted credential files.

## Data converters (tools/data/)

- Emit normalized, UTF-8 JSONL via `_split.dumps_record` (the `SplitWriter` write path): NFKC plus stray line-separator stripping (NEL U+0085, U+2028, U+2029 -> space, so records never fragment across lines).
- New converters must route every record write through `dumps_record`, never raw `json.dumps`, and open all files with `encoding="utf-8"`.
- When using json.dump or json.dumps ensure_ascii should always be set to False unless I have directed otherwise.

## TRAINING DATA LABELS

- Labels are an INPUT to GLiNER2 at inference, so one concept must have one spelling
  across every corpus a model trains on. `LOC`/`Loc`/`Location`/`LOCATION` are one
  label; so are `CompanyName`/`Company Name`. Same for relations, events, structures
  and classifications.
- Unify in the CONFIG, not by rewriting corpora: `labels_file: labels/unified.yaml`,
  regenerated by `tools/train/build_label_maps.py`. Every model warm-started from a
  base must reference the SAME file, or the warm start is against a vocabulary the
  base never learned. The BASE's spelling is canonical, chosen by the base's OWN
  counts: a downstream corpus must never outvote the base among the base's own labels.
  An EMPTY inline `labels:` block silently OVERRIDES `labels_file` -- delete it when
  wiring a config, or the wiring is a no-op that looks done.
- Rewrite the data only to fix the LANGUAGE of a label, or to split labels a
  translation collapsed. Check for many-to-one collisions after translating: two
  distinct DuEE event types both came back as `withdrawal`.
- Only TAXONOMY corpora may drive the map. Measure first: `mix_natural` is 58%
  placeholder labels (`e_0` x30,034) and 72% singletons, and `zh_multitask` is 70%
  singletons because the annotator invented labels per document. Feeding those to a
  label map manufactures thousands of meaningless merges.
- Do NOT merge on string similarity alone. Prove two labels mean the same thing by
  reading the surfaces they tag. Known keeps: `GPE` != `Location` (Thailand vs Indian
  Ocean), `NORP` != `Organization` (nationalities), `FAC` != `Location` (buildings),
  redocred's `TIME` holds dates, and docee's `Target` (who an attack hit) !=
  bio_ner_relations' `target` (a kinase substrate).
- Scan labels at EVERY nesting depth and check the label MENU separately from the
  answers. Reading `rec["events"]` hid 147,456 Chinese labels nested under
  `rec["output"]`, and a corpus can present a Chinese menu while its answers are English.
- Roll-up runs BEFORE the map, so a map key containing the separator can never fire.
- Verify a unification three ways: zero variant clusters remain, zero label uses were
  lost in any category, and the map is CLOSED -- no target is itself a key. The map is
  applied once, so `LOC: Location` beside `Location: location` leaves both alive.

## TRAINING

- Always verify all samples of aggregated train and test and validation are unique, and report if they are not. I do not want cross-set contamination, as that invalidates results. I do not want samples from validation or test in the train dataset. Test is blind-test and needs to be unique no duplicates from train or validation.
