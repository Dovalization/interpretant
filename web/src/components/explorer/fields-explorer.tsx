"use client";

import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";

import { Segmented } from "@/components/explorer/controls";
import { FallBars } from "@/components/explorer/fall-bars";
import { PairChart } from "@/components/explorer/pair-chart";
import { PerplexityMatrix } from "@/components/explorer/perplexity-matrix";
import { FigurePanel } from "@/components/figure-panel";
import { decadeLabel, fieldLabel } from "@/lib/fields";
import { pairSeries, perplexityMatrix } from "@/lib/perplexity";

import type { Falls, Perplexity } from "@/lib/schema";

export function FieldsExplorer({ table, falls }: { table: Perplexity; falls: Falls }) {
  const decades = table.decades.filter((decade) => decade >= 1980);
  const [state, setState] = useQueryStates({
    decade: parseAsInteger.withDefault(decades[decades.length - 1] ?? 2020),
    model: parseAsString.withDefault("philosophy"),
    text: parseAsString.withDefault("hci"),
    gained: parseAsString.withDefault("hci"),
  });
  const gainTexts = [...new Set(falls.rows.map((row) => row.text))].sort();
  const gainRows = falls.rows.filter((row) => row.text === state.gained);
  const span = falls.rows[0];
  const matrix = perplexityMatrix(table, state.decade);
  const pair = { model: state.model, text: state.text };

  return (
    <div className="flex flex-col gap-8">
      <FigurePanel
        heading="Whose model gained on a field's text, and why"
        subheading={
          span
            ? `Fall in excess cross-entropy from the ${decadeLabel(span.early)} to the ${decadeLabel(span.late)}, split into the text's word mix moving and the model's per-word skill changing`
            : "Fall in excess cross-entropy between two decades, split into the text's word mix moving and the model's per-word skill changing"
        }
        actions={
          gainTexts.length > 0 && (
            <Segmented
              label="Text"
              value={state.gained}
              options={gainTexts.map((text) => ({ value: text, label: fieldLabel(text) }))}
              onChange={(gained) => setState({ gained })}
            />
          )
        }
        legend="Predictable is not borrowed: a fall can come from the text's frequencies moving towards words the model already handled (mix) or from the model getting better at the text's words (rate). For HCI the mix dominates: the field's own prose left computing's register, and every non-computing model gained on it. Corrected design, mean of five seeds."
      >
        {gainRows.length > 0 ? (
          <FallBars rows={gainRows} highlight={state.model} />
        ) : (
          <p className="py-6 text-sm text-muted-foreground">
            The decomposition has not been exported yet (`interpretant drift decompose`, then
            `interpretant export web`).
          </p>
        )}
      </FigurePanel>
      <FigurePanel
        heading="Who predicts whom"
        subheading="Excess cross-entropy of each field's held-out abstracts under every other field's bigram model, bits per token"
        actions={
          <Segmented
            label="Decade"
            value={String(state.decade)}
            options={decades.map((decade) => ({
              value: String(decade),
              label: decadeLabel(decade),
            }))}
            onChange={(value) => setState({ decade: Number(value) })}
          />
        }
        legend="Rows: the field whose model predicts. Columns: the field whose text is predicted. A low value means the column's language is predictable from the row's. Kneser-Ney bigrams, 200,000 training tokens per cell, shared abstracts removed, 1,000 held-out abstracts, mean of five seeds. Fields with fewer than 200,000 training tokens in a decade are left out. Click a cell to follow that pair over time."
      >
        <PerplexityMatrix matrix={matrix} selected={pair} onSelect={(next) => setState(next)} />
      </FigurePanel>
      <FigurePanel
        heading={`${fieldLabel(state.model)} and ${fieldLabel(state.text)} over time`}
        subheading="Both directions of the pair; points are the five resampled runs, lines their mean"
        legend="E(A → B) falls when B's language moves towards A's, or when A's language broadens to cover B's; the two readings are told apart by the other direction and by the words behind the change."
      >
        <PairChart
          a={state.model}
          b={state.text}
          points={pairSeries(table, state.model, state.text)}
        />
      </FigurePanel>
    </div>
  );
}
