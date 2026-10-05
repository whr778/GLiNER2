# Junction layer: trigger x argument links for natural-mode events

**Status:** BUILT (21f357e) and A/B'd 2026-10-04 (p5link): argument F1 +0.035*, per-column junction AUC 0.58 -> 0.74 / 0.77, English arguments move for the first time. Going into eb19.

## 1. Why: the assignment has no join

In a relational database, `Person` and `Address` are related many-to-many through a third table that holds both primary keys. That table makes the link queryable in **both** directions. Our event head has no such table.

Today, instance *i* (a trigger candidate) scores candidate *j* for role field *f* as (`records.py` `_assign_logits`):

$$s_{ijf} = \underbrace{\mathrm{inst\_proj}(t_i)\cdot\mathrm{cand\_proj}(a_j)}_{P_{ij}\ \text{pairwise}} + \underbrace{\mathrm{field\_proj}(r_f)\cdot\mathrm{cand\_proj}(a_j)}_{R_{fj}\ \text{role, identical for every trigger}}$$

**Measured** (`measure_assign_decomposition.py`; 16 training-shaped batches, 456 gold arguments, 3,648 false-trigger comparisons). The gate held: P + R reproduces the model's logits to 6.9e-05.

| Readout | eb18 | p3arg-hard |
|---|---|---|
| Variance across triggers vs across candidates | 18.7 vs 44.1 | 4.3 vs 12.8 |
| **Junction AUC (per column):** gold vs false trigger, same gold argument | **0.631** | **0.576** |
| Gold argument vs rest of its row (per row): R only / P only / P + R | 0.947 / **0.405** / 0.948 | 0.953 / **0.371** / 0.954 |

*Corrected 2026-10-05: first reported as 0.609 / 0.560 etc. from a POOLED AUC (all gold vs all false scores across columns); the per-column / per-row values above are the measure the text describes. Conclusions unchanged.*

**The assignment is role fit with no join.** This one fact explains three findings we couldn't explain:
- argument evidence cannot separate gold from false triggers (AUC ~0.5);
- every negative-instance setting cost recall (pushing false rows down lowers the shared R);
- false triggers take 52-74% of wrong-trigger arguments.

## 2. The relational model

| Relational | Here |
|---|---|
| `Trigger` table, key = span | trigger instances $t_i$: the natural group's anchor candidates (gold-seeded + top-scoring) |
| `Argument` table, key = span | argument candidates $a_j$. Their states are **role-independent** span encodings (`candidate_encoder` over start/end boundary states, `model.py:606`). |
| `Link(trigger_id, argument_id)` | a learned **link vector** $\ell_{ij} = g(t_i, a_j, \text{span geometry})$ |
| a column on the link row | the **role**, read as a query: $s_{ijf} = \langle \ell_{ij}, q_f \rangle$ |
| `SELECT ... WHERE trigger_id = i` | the trigger's **row**: its arguments (the king knows his subjects) |
| `SELECT ... WHERE argument_id = j` | the argument's **column**: its triggers (the subjects know their king) |

## 3. Architecture

A new setting `record_link_mode: additive | junction`. The default `additive` is today's code, bit-identical.

Under `junction`, the natural-mode assignment score becomes

$$s_{ijf} = P_{ij} + R_{fj} + \underbrace{\langle (U\,t_i) \odot (V\,a_j) \odot \sigma(G\,q_f),\ \mathbf 1\rangle / \sqrt d}_{\text{gated bilinear link}} + \underbrace{w_f^\top e_{\mathrm{geom}}(i,j)}_{\text{order + distance}}$$

- **Gated bilinear link.** This is the junction row, and the same pattern as `SparseRelationScorer`'s `use_biaffine_content`: head ⊙ σ(gate(label)) ⊙ tail. The role gates the link, so role and pair identity interact instead of adding.
- **Geometry.** Relative order and a bucketed token distance between the trigger and argument spans. It's available for free from `instance_spans` and `field_spans`, and the relation scorer already uses both features.
- **Zero init of $V$ and $w$.** At a warm start, $s_{ijf}$ equals today's score exactly, so decode from eb18 is bit-identical at step 0 and the link is learned from data. P and R are kept, so role fit stays expressible.

**Cost.** The extra term costs one $[N_i, C_f]$ elementwise product per field, the same shape as today's assignment matrix. There is no new candidate enumeration.

## 4. Training: train the join, not the absences

What p4neg taught us: absolute "no argument" targets on false rows **lower the shared role term** and cost recall. The junction is instead trained **contrastively in the column**, which only pair identity can satisfy:

1. **Row loss (unchanged):** for each gold-seeded instance, the role BCE over gold + the K hardest wrong candidates (`record_role_hard_negatives`).
2. **Column loss (new):** for every gold argument *j* of role *f*, a listwise softmax over the triggers in its column:

   $$\mathcal L_{col} = -\log \frac{\sum_{i \in G_j} e^{s_{ijf}}}{\sum_{i \in G_j \cup H_j} e^{s_{ijf}}}$$

   - $G_j$ = the instance(s) seeded by its gold trigger. That's every alternative, so it stays forward-compatible with coreference.
   - $H_j$ = the K hardest **false** triggers: highest-scoring, non-overlapping a gold trigger, reusing `_negative_instances`.

   Because R is identical across a column, **this loss can only be lowered through the link term**. It is a direct optimiser of the junction AUC, and it never pushes role fit down.
3. **Weight:** `record_link_column_weight` (default 0), its own mean. As with every term since `absent_reduction`, it never enters another term's denominator.

Settings, all opt-in with defaults that are bit-identical:

| Setting | Default | Meaning |
|---|---|---|
| `record_link_mode` | `additive` | `junction` adds the gated bilinear + geometry terms |
| `record_link_column_weight` | 0 | weight of the column (which-trigger-owns-this-argument) loss |
| `record_link_column_negatives` | 8 | K false triggers per gold argument |

## 5. Decode

- **Phase 1 (this spec):** decode unchanged. Exclusive allocation now ranks a contested argument by a score that **knows which trigger it belongs to**. That alone targets the false-trigger attachments (52-74%).
- **Phase 2 (later, measured separately):** many-to-many decode (per-link threshold, no exclusive allocation), and a trigger's **row aggregate** feeding its existence score. That's the existence head built on a signal that can carry it, replacing the precondition that failed.

## 6. Gates, measured from inside the run

- **Junction AUC** (gold vs false trigger, same gold argument) logged every N steps from the training forward, with the usual backoff. It must **rise clearly** (target >= 0.8 from 0.58-0.63, per column) or the layer is not learning the join, whatever the blind test says.
- **Row AUCs** by P, R and the link term: shows which term carries the separation.
- **Step 0:** with $V$ and $w$ zero-initialised, $s_{ijf}$ must equal `additive` exactly (a test that can fail).
- **Proof line:** the cumulative count of column-loss terms and false triggers used, so an arm that never trained the column is visible.

## 7. Build order (trace, code, trace, test)

1. **Trace before:** on the real CASIE batch, the column scores of each gold argument across gold vs false triggers (the `measure_assign_decomposition.py` machinery).
2. **Code:** the gated link + geometry in `_assign_logits` (natural mode, `junction`), and the column loss in `compute_group_loss`.
3. **Trace after:**
   - step-0 equality;
   - column-loss scale against the row loss;
   - the gradient reaching U, V, G and w;
   - a few CPU steps on one document showing junction AUC rising.
4. **Tests:** zero-init equality; the column loss lowered only through the link term (R cancels); shared pool refused; mutation checks.
5. **Fast A/B** (p3arg setup, every arm on top of the hard role loss):
   - control (×2, for the floor);
   - junction architecture only;
   - junction + column loss.

   **Readouts:**
   - junction AUC (in-run and post-hoc);
   - argument P/R/F1 by language;
   - the wrong-trigger split (false-trigger share);
   - below-gate share;
   - **every other head**.
6. **If it holds:** eb19 = eb18 + hard role loss + `proposal_gold: identity` + the junction.

## 8. Open items and risks

- **Strings, not offsets:** gold trigger and argument strings mark every occurrence, so some "gold" links are polluted. That's tolerable for a first test, and offset-anchored mentions remain the clean fix.
- **The relation head** uses the same pattern and scores low (~0.12 F1). The pattern isn't proof of success. The junction AUC gate is what decides.
- **English arguments** (~0.005 F1) may be limited by recall in the scorer, not the join. The A/B reads them separately.
- **Shared pool:** refuse the settings there, as for the other record options.
- **Design C** (anchorless slots) stays parked. The junction keeps trigger-anchored instances and adds the missing join.
