---
Status: accepted
---

# The two rural definitions are deliberately different and are not reconciled

A rural label is applied to two different things, by two different rules, and the two rules are
allowed to disagree.

**A listing is rural when the PSA barangay containing its coordinate is PSA-rural.** This rule
selects the **training population**: whether a listing teaches the model anything about rural STR
behaviour is a question about the listing's own local context, not about the modal character of the
square kilometre it happens to fall in.

**A grid cell is rural when at least half of its barangay-covered area lies in PSA-rural
barangays.** This rule selects the **scoring domain**: which cells are mapped and scored.

The two can disagree, and **30 of the 245 fitted rural listings sit in cells whose majority area is
urban**. This is not a defect to reconcile. Relabelling a listing to match its cell would silently
discard 30 real rural observations; relabelling a cell to match a minority of its area would corrupt
the prediction domain. The listing label is authoritative for who trains the model; the cell label
is authoritative for which cells are predicted and scored.

## Considered options

- **Reconcile the rules** (drop the 30 mismatching listings, or reclassify their cells). Rejected:
  it trades away either real observations or a correct prediction domain for a tidier story.
- **Use one rule for both.** Rejected: a single rule cannot be authoritative for two different
  questions — "is this observation rural?" and "is this place rural?".
- **Keep both and document the disagreement.** Chosen.

## Consequences

The disagreement is a stable, reported quantity (`listings_in_urban_cells` in the rural model
summary), not a hidden edge case. A reader who reconciles the two rural counts will find them
different on purpose, and this record explains why.
