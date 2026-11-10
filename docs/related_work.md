# Related work, and what it means for the next design

*Recorded 2026-11-10, after rounds C–C″ (senses) and round 5 (the grid) and before any redesign.
Each source is marked by how it was read: **read** (the paper's text, through its arXiv or ar5iv
page), or **summary** (search results and abstracts only). Numbers from a summary should be
checked against the paper before they are relied on.*

## 1. What the field already knows about our results

| Our result | Related finding | Source |
|---|---|---|
| Rounds C–C″: the encoder placed every known sense, but no rule for turning clusters into a meaning's share held on unseen data | Clustering usages into senses is the field's weak point. On English, the best models match human sense clusters with an adjusted Rand index of 0.27 (XL-LEXEME) and 0.34 (GPT-4), against 0.63 between human annotators; BERT, the family our sense instrument used, scores 0.14 | Periti and Tahmasebi 2024 (read) |
| We encoded usages with off-the-shelf SciBERT and ModernBERT | Encoders trained on word-in-context judgements do far better. XL-LEXEME reaches a Spearman correlation of 0.89 with human graded-change scores on English, where BERT reaches 0.56; GPT-4 used as a pair judge reaches 0.82, about XL-LEXEME's 0.80 in the same setting | Periti and Tahmasebi 2024 (read); Cassotti et al. 2023 (summary) |
| We measured a meaning as a share of clusters | The best-performing measure of graded change needs no clusters: the average pairwise distance (APD) between usages of two periods. It outranks the distance between mean vectors and both clustering-based measures | Periti and Tahmasebi 2024 (read) |
| Round 5: the grid's flags are evidence and its silences are not; controls matter | Several published "laws" of semantic change were artefacts of the representation and of word frequency, visible only against a control condition in which no change can have occurred | Dubossarsky et al. 2017 (summary) |
| A change in a measure can mean something else | Contextual models give high change scores to words whose range of contexts changed without any change of sense, and blend syntax with meaning | Kutuzov et al. 2022 (summary) |
| Our known answers were few and chosen by the author | Human gold exists. DWUG EN holds roughly 100,000 human judgements of usage pairs for English, with graded-change scores per word; the LSCD benchmark standardises evaluation on it. For differences between domains rather than periods, SURel applies the same pair-judging protocol to general against specialist text (German, cooking) | Schlechtweg et al. 2021, 2024; Hätty et al. 2019 (summaries) |
| Our question: what a word means in each field | Discipline-specific senses were measured across 293 subfields and 12 million abstracts: a masked model's top five substitutes per occurrence, grouped into senses by community detection, and each sense scored against fields by normalised pointwise mutual information. Validation is recall against Wiktionary's field-labelled senses (11,548 word–subfield pairs); precision is not computed | Lucy et al. 2023 (read) |
| Readable evidence survived | Usages can be represented by what a model would put in their place (substitutes) or by a generated definition; both are readable and both perform competitively on change detection | Card 2023; Giulianelli et al. 2023; Fedorova et al. 2024 (summaries) |
| A sense appearing in one field (the *hallucination* question) | Global measures such as APD average over all usages; a usage-level nearest-match measure (AMD) was proposed for one-sided changes such as a sense emerging or disappearing | Goworek and Dubossarsky 2026 (summary) |
| A language model as annotator | Results are mixed and prompt-dependent: GPT-4 agreed with two human annotators at Spearman 0.45 and 0.6 where they agreed with each other at 0.87, with a one-sentence prompt; small changes to the human guidelines improved it; a shared task (CoMeDi 2025) now evaluates models as annotators of usage pairs | Karjus 2023; Schlechtweg's group 2024; CoMeDi 2025 (summaries) |

Two conclusions follow. The sense rounds' stopping rule fired for a reason the field recognises:
we used the weakest configuration in the literature (a general encoder and clustering) with no
external benchmark. And the nearest prior study to this project, Lucy et al., rests on the same
clustering step and validates it by recall alone.

## 2. The design this points to

**Primitive: the usage pair.** For a word, two cells and a judge, the quantity is how much more
distant pairs drawn across the cells are than pairs drawn within each. Across two fields in one
decade it measures a difference of meaning between communities; across two decades of one field,
a change. The within-cell baseline removes each cell's own spread of contexts, the false-positive
source Kutuzov et al. describe. There is no clustering step.

**Judges, three on the same pairs.** XL-LEXEME as the main instrument; a language model as pair
judge on a subsample; the author on a small gold set from this corpus, on the four-point scale
the benchmarks use (the openly supervised element `docs/known_answers.md` says a new attempt
needs).

**Readable evidence.** For any divergent pair of cells, the most distant sentence pairs, and
optionally a generated definition per usage.

**A calibration ladder**, each rung pre-registered with its own stopping rule:

1. Reproduce XL-LEXEME's published result on DWUG EN. A human benchmark nobody on this project
   chose; if the pipeline cannot reproduce it, nothing else proceeds.
2. This project's known answers over time, with the configuration frozen from rung 1:
   *attention, hallucination, prompt, alignment, platform, network*, and the quiet transitions.
3. Known answers between fields: *embodiment* between HCI and philosophy must diverge, random
   control words must not, and Wiktionary's field-labelled senses give a larger set.
4. Only then open questions: a survey of field-specific meanings, and one-sided changes such as
   the language-model sense of *hallucination* reaching psychology.

**Carried over:** word-targeted OpenAlex cells, the quality rules, the nested-field rule, the
label-noise table, and the pre-registration protocol.

**Not done:** replicating the clustering approach of Lucy et al.; repairing the grid; reopening
the direction of HCI's change.

## 3. Caveats recorded with the plan

- XL-LEXEME was trained on general text; whether it holds on research abstracts is what rungs 2
  and 3 test.
- DWUG EN may be downloaded but not redistributed, so it stays out of the repository.
- A language-model judge costs money per pair; volumes are set after checking current prices.
- McMahan and Evans (2018), on ambiguity of scientific terms, may be relevant; its method could
  not be read (paywalled) and nothing here relies on it.

## 4. Added with the sense-label instrument (2026-11-10)

- **Substitution-based sense induction.** A masked language model's fillers for a target in
  context, grouped into senses, induce a word's senses without a fixed inventory. Amrami and
  Goldberg (2018) used dynamic symmetric patterns ("X and Y") with a recurrent language model;
  their 2019 follow-up moved to BERT with the pattern "target (or even [MASK])". Eyal et al.
  (2022) scaled the approach with community detection over substitute co-occurrence, giving a
  variable number of senses each named by its representative substitutes. Round Q's step A uses
  this, with ModernBERT-large because BERT, trained on older text, does not offer the attention
  mechanism's substitutes.
- **Decision models.** JEV (TypeSafe AI) answers typed questions with probabilities and no free
  text. Rao and Callison-Burch (2026) compare decision models with language-model judges on
  evaluation benchmarks: comparable on binary criteria, weaker on graded ones, much cheaper, and
  wrong in many of the same places. No published use for word senses was found; this project's
  comparison is in [`labellers.md`](labellers.md).

## Sources

- Cassotti, Siciliani, de Gemmis, Semeraro and Basile (2023). XL-LEXEME: WiC Pretrained Model
  for Cross-Lingual LEXical sEMantic changE. ACL. <https://aclanthology.org/2023.acl-short.135>
- Card (2023). Substitution-based Semantic Change Detection using Contextual Embeddings. ACL.
  <https://aclanthology.org/2023.acl-short.52>
- Dubossarsky, Weinshall and Grossman (2017). Outta Control: Laws of Semantic Change and Inherent
  Biases in Word Representation Models. EMNLP. <https://aclanthology.org/D17-1118/>
- Fedorova, Kutuzov and Scherrer (2024). Definition generation for lexical semantic change
  detection. <https://arxiv.org/abs/2406.14167>
- Giulianelli, Luden, Fernandez and Kutuzov (2023). Interpretable Word Sense Representations via
  Definition Generation. ACL. <https://arxiv.org/abs/2305.11993>
- Goworek and Dubossarsky (2026). Rethinking Metrics for Lexical Semantic Change Detection.
  LChange workshop. <https://arxiv.org/abs/2602.15716>
- Hätty, Schlechtweg and Schulte im Walde (2019). SURel: A Gold Standard for Incorporating
  Meaning Shifts into Term Extraction. *SEM.
  <https://www.ims.uni-stuttgart.de/en/research/resources/experiment-data/surel>
- Karjus (2023). Machine-assisted quantitizing designs. <https://arxiv.org/abs/2309.14379>
- Kutuzov, Velldal and Øvrelid (2022). Contextualized embeddings for semantic change detection:
  Lessons learned. NEJLT. <https://aclanthology.org/2022.nejlt-1.9>
- Lucy, Dodge, Bamman and Keith (2023). Words as Gatekeepers: Measuring Discipline-specific Terms
  and Meanings in Scholarly Publications. Findings of ACL. <https://arxiv.org/abs/2212.09676>
- Periti and Tahmasebi (2024). A Systematic Comparison of Contextualized Word Embeddings for
  Lexical Semantic Change. NAACL. <https://arxiv.org/abs/2402.12011>
- Schlechtweg, Hätty, Del Tredici and Schulte im Walde (2019). A Wind of Change: Detecting and
  Evaluating Lexical Semantic Change across Times and Domains. ACL.
  <https://arxiv.org/abs/1906.02979>
- Schlechtweg et al. (2021). DWUG: A large Resource of Diachronic Word Usage Graphs in Four
  Languages. EMNLP. <https://arxiv.org/abs/2104.08540>; DWUG EN, <https://zenodo.org/record/5544443>
- Schlechtweg, Virk and Arefyev (2024). The LSCD Benchmark: a Testbed for Diachronic Word
  Meaning Tasks. <https://arxiv.org/abs/2404.00176>
- Liu, Hu and Liu (2024). An entry to the CoMeDi shared task (2025), which evaluates models as
  annotators of usage pairs. <https://arxiv.org/abs/2411.12147>
- Amrami and Goldberg (2018). Word Sense Induction with Neural biLM and Symmetric Patterns.
  EMNLP. <https://arxiv.org/abs/1808.08518>
- Amrami and Goldberg (2019). Towards better substitution-based word sense induction.
  <https://arxiv.org/abs/1905.12598>
- Eyal, Sadde, Taub-Tabib and Goldberg (2022). Large Scale Substitution-based Word Sense
  Induction. ACL. <https://aclanthology.org/2022.acl-long.325>
- Rao and Callison-Burch (2026). JEV vs. LLMs as Rubric Judges: Cheaper, Faster, and Wrong in the
  Same Places. <https://arxiv.org/abs/2609.29769>
- Forman (2008). Quantifying counts and costs via classification. Data Mining and Knowledge
  Discovery 17(2).
- Northcutt, Jiang and Chuang (2021). Confident Learning: Estimating Uncertainty in Dataset Labels.
  Journal of Artificial Intelligence Research 70. <https://arxiv.org/abs/1911.00068>
- Pilehvar and Camacho-Collados (2019). WiC: the Word-in-Context Dataset for Evaluating
  Context-Sensitive Meaning Representations. NAACL. <https://arxiv.org/abs/1808.09121>
- Ratner, Bach, Ehrenberg, Fries, Wu and Ré (2017). Snorkel: Rapid Training Data Creation with Weak
  Supervision. VLDB. <https://arxiv.org/abs/1711.10160>
