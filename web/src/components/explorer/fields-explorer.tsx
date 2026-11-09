"use client";

import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";

import { Segmented } from "@/components/explorer/controls";
import { PairChart } from "@/components/explorer/pair-chart";
import { PerplexityMatrix } from "@/components/explorer/perplexity-matrix";
import { FigurePanel } from "@/components/figure-panel";
import { decadeLabel, fieldLabel } from "@/lib/fields";
import { pairSeries, perplexityMatrix } from "@/lib/perplexity";

import type { Perplexity } from "@/lib/schema";

export function FieldsExplorer({ table }: { table: Perplexity }) {
  const decades = table.decades.filter((decade) => decade >= 1980);
  const [state, setState] = useQueryStates({
    decade: parseAsInteger.withDefault(decades[decades.length - 1] ?? 2020),
    model: parseAsString.withDefault("philosophy"),
    text: parseAsString.withDefault("hci"),
  });
  const matrix = perplexityMatrix(table, state.decade);
  const pair = { model: state.model, text: state.text };

  return (
    <div className="flex flex-col gap-8">
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
