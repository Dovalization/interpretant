# Guidelines for judging usage pairs

Adapted from the DURel annotation guidelines (Schlechtweg et al. 2018), used for check 2b in
`docs/diagnostics.md`.

Each item shows one word, marked ⟦like this⟧, in two sentences from research abstracts. Judge
how related **the word's meaning** is in the two sentences, not whether the sentences are about
the same topic.

| Score | Meaning | Example (*network*) |
|---|---|---|
| 4 | **Identical**: the word means the same thing in both | "a neural ⟦network⟧ classifier" / "training neural ⟦networks⟧ on images" |
| 3 | **Closely related**: same sense, different nuance or use | "a neural ⟦network⟧ classifier" / "a ⟦network⟧ of neuron-like units" |
| 2 | **Distantly related**: different senses that share a link | "a neural ⟦network⟧ classifier" / "a semantic ⟦network⟧ of concepts" |
| 1 | **Unrelated**: different meanings with nothing in common | "a neural ⟦network⟧ classifier" / "a PERT ⟦network⟧ for project scheduling" |
| 0 | **Cannot decide**: the sentence is too short, garbled or ambiguous | |

- Judge the marked word only. Topic, field and quality of writing do not matter.
- A plural or a different part of speech of the same meaning still counts as identical.
- A fixed phrase (*pay ⟦attention⟧*, *received ⟦attention⟧*) has its own meaning; compare it
  with the other sentence as it is.
- Go with your first considered judgement; do not revise earlier items.
- Write one number per item after `Judgement:`. Sittings of 15–20 minutes are fine; the order
  does not matter.
