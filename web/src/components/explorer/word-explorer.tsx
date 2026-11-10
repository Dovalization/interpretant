"use client";

import {
  parseAsArrayOf,
  parseAsInteger,
  parseAsString,
  parseAsStringLiteral,
  useQueryStates,
} from "nuqs";

import { Chips, Segmented } from "@/components/explorer/controls";
import { DivergenceMatrix } from "@/components/explorer/divergence-matrix";
import { FrequencyChart } from "@/components/explorer/frequency-chart";
import { NeighbourColumns } from "@/components/explorer/neighbour-columns";
import { TrajectoryChart } from "@/components/explorer/trajectory-chart";
import { FigurePanel } from "@/components/figure-panel";
import {
  defaultFields,
  divergence,
  divergenceDecades,
  fieldsInLayer,
  frequency,
  LAYERS,
  METHODS,
  NEIGHBOUR_COUNTS,
  neighbourColumns,
  sharedWords,
  trajectory,
} from "@/lib/explorer";
import { decadeLabel, fieldColor, fieldLabel, LAYER_LABELS, METHOD_LABELS } from "@/lib/fields";

import type { Layer, WordData } from "@/lib/schema";

interface WordExplorerProps {
  data: WordData;
  defaultLayer: Layer;
}

export function WordExplorer({ data, defaultLayer }: WordExplorerProps) {
  const [state, setState] = useQueryStates({
    layer: parseAsStringLiteral(LAYERS).withDefault(defaultLayer),
    method: parseAsStringLiteral(METHODS).withDefault("twec"),
    k: parseAsInteger.withDefault(12),
    fields: parseAsArrayOf(parseAsString),
    decade: parseAsInteger,
  });
  const available = fieldsInLayer(data, state.layer, state.method);
  const chosen = (state.fields ?? []).filter((field) => available.includes(field));
  // A layer or method switch can leave no chosen field in the new layer; show its defaults.
  const fields = chosen.length > 0 ? chosen : defaultFields(data, state.layer, state.method);
  const columns = neighbourColumns(data, state.layer, state.method, fields);
  const shared = sharedWords(columns, state.k);
  const decades = divergenceDecades(data);
  const decade = state.decade ?? decades[decades.length - 1];
  const matrix = decade === undefined ? undefined : divergence(data, decade);

  const layerOptions = LAYERS.map((layer) => ({
    value: layer,
    label: LAYER_LABELS[layer],
    disabled: fieldsInLayer(data, layer, "twec").length === 0,
  }));

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-lg border bg-card px-4 py-3">
        <Segmented
          label="Layer"
          value={state.layer}
          options={layerOptions}
          onChange={(layer) => setState({ layer, fields: null })}
        />
        <Segmented
          label="Method"
          value={state.method}
          options={METHODS.map((method) => ({ value: method, label: METHOD_LABELS[method] }))}
          onChange={(method) => setState({ method })}
        />
        <Segmented
          label="Neighbours"
          value={String(state.k)}
          options={NEIGHBOUR_COUNTS.map((count) => ({ value: String(count), label: count }))}
          onChange={(k) => setState({ k: Number(k) })}
        />
        <Chips
          label="Fields"
          value={fields}
          options={available.map((field) => ({ value: field, label: fieldLabel(field) }))}
          max={4}
          colorFor={fieldColor}
          onChange={(next) => setState({ fields: next })}
        />
      </div>

      <FigurePanel
        heading="Nearest neighbours by decade"
        subheading={`The ${state.k} words closest to ${data.word} in each cell; grey words appear under more than one field`}
        legend={`${METHOD_LABELS[state.method]} models, ${LAYER_LABELS[state.layer].toLowerCase()}; neighbours come from words with at least 50 occurrences in the cell, and a cell needs 150 occurrences of the word to be shown. Hover a word to find it in the other lists.`}
      >
        <NeighbourColumns columns={columns} shared={shared} k={state.k} />
      </FigurePanel>

      <div className="grid gap-8 lg:grid-cols-2">
        <FigurePanel
          heading="Drift between consecutive decades"
          subheading="Percentile ranks against each pair's whole-vocabulary noise floor"
          legend="Solid: cosine drift rank; dashed: neighbourhood shift rank. A large dot marks a candidate, above the 95th percentile on both. TWEC seed 42."
        >
          <div className="grid gap-6">
            {fields.map((field) => (
              <TrajectoryChart
                key={field}
                field={field}
                rows={trajectory(data, state.layer, field)}
              />
            ))}
          </div>
        </FigurePanel>
        <FigurePanel
          heading="Frequency"
          subheading="Occurrences per million tokens in each field's cells"
          legend="Counts come from the cell models; a field's series starts in the first decade with a cell."
        >
          <FrequencyChart points={frequency(data, state.layer, fields)} fields={fields} />
        </FigurePanel>
      </div>

      {decades.length === 0 && (
        <FigurePanel
          heading="Divergence between fields"
          subheading="How differently two fields use the word in one decade, main grid"
        >
          <p className="py-6 text-sm text-muted-foreground">
            No pair of main-grid cells has 150 occurrences of {data.word} in the same decade, so
            cross-field divergence is not measured for it. The full cells above carry its
            within-field story.
          </p>
        </FigurePanel>
      )}
      {matrix && decade !== undefined && (
        <FigurePanel
          heading="Divergence between fields"
          subheading="How differently two fields use the word in one decade, main grid"
          actions={
            <Segmented
              label="Decade"
              value={String(decade)}
              options={decades.map((item) => ({ value: String(item), label: decadeLabel(item) }))}
              onChange={(value) => setState({ decade: Number(value) })}
            />
          }
          legend="Each cell is the percentile rank of the word's cosine distance between the two fields' models against the pair's noise floor: 95 and above means the two fields use the word more differently than they use almost every other word of its frequency. A ring marks a candidate (top 5% on both measures). Only pairs where the word has 150 occurrences in both cells are measured."
        >
          <DivergenceMatrix matrix={matrix} highlight={fields} />
        </FigurePanel>
      )}
    </div>
  );
}
