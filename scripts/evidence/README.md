# Evidence for the decision agenda

The scripts behind the measured numbers in `docs/decision_agenda.md`, kept as a record of how
each number was produced. They import the pre-rebuild package and read the pre-rebuild data
folders, so they are not expected to run after the rebuild; git keeps them.

- Top level: the audits behind steps 2–4 (raw text problems, apostrophes, the lemma map,
  compounds, mentions per abstract, sentence lengths, `DocBin` size, small against transformer
  model).
- `first_review/`: the checks behind the first review's amendments (check 1a's mention
  positions, permutation floors and J's chance level, the `DocBin` round trip, the lemma review's
  coverage).
- `second_review/`: the second review's experiments (J's identity with the mean-vector form and
  alternative measures on DWUG EN, bootstrap coverage and the control bar, the bar on round X's
  vectors, frequency matching of controls, NFKC and repeated sentences, adjusted counting against
  the stratified estimator); `decision_rule_output.txt` is the output of the longest run.
