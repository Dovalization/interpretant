import type { CrossRow, Layer, Method, WithinRow, WordData } from "@/lib/schema";

/** Pure helpers that turn one word's export into what the explorer panels draw. */

export interface NeighbourColumn {
  field: string;
  decade: number;
  count: number;
  words: string[];
}

export const LAYERS: readonly Layer[] = ["openalex", "openalex_full", "books"];
export const METHODS: readonly Method[] = ["twec", "ppmi"];
export const NEIGHBOUR_COUNTS = [12, 25] as const;

const unique = <T>(values: T[]): T[] => [...new Set(values)];

/** Fields that have at least one neighbour list for the word in this layer and method. */
export const fieldsInLayer = (data: WordData, layer: Layer, method: Method): string[] =>
  unique(
    data.neighbours
      .filter((entry) => entry.layer === layer && entry.method === method)
      .map((entry) => entry.field),
  ).sort();

/** The layer with the most neighbour lists for the word: full cells when they exist. */
export const defaultLayer = (data: WordData): Layer => {
  const counts = new Map<Layer, number>();
  for (const entry of data.neighbours.filter((item) => item.method === "twec")) {
    counts.set(entry.layer, (counts.get(entry.layer) ?? 0) + 1);
  }
  return LAYERS.reduce(
    (best, layer) => ((counts.get(layer) ?? 0) > (counts.get(best) ?? 0) ? layer : best),
    "openalex",
  );
};

/** Up to two fields: the ones with the most candidate transitions, then the most cells. */
export const defaultFields = (data: WordData, layer: Layer): string[] => {
  const available = fieldsInLayer(data, layer, "twec");
  const score = (field: string): [number, number] => [
    data.within.filter((row) => row.layer === layer && row.field === field && row.candidate).length,
    data.neighbours.filter((entry) => entry.layer === layer && entry.field === field).length,
  ];
  return [...available]
    .sort((a, b) => {
      const [candidatesA, cellsA] = score(a);
      const [candidatesB, cellsB] = score(b);
      return candidatesB - candidatesA || cellsB - cellsA || a.localeCompare(b);
    })
    .slice(0, 2);
};

/** Neighbour lists for the chosen fields, grouped by field in the order given, then by decade. */
export const neighbourColumns = (
  data: WordData,
  layer: Layer,
  method: Method,
  fields: string[],
): NeighbourColumn[] => {
  const countFor = (field: string, decade: number): number =>
    data.counts.find(
      (count) => count.layer === layer && count.field === field && count.decade === decade,
    )?.count ?? 0;
  return fields.flatMap((field) =>
    data.neighbours
      .filter((entry) => entry.layer === layer && entry.method === method && entry.field === field)
      .sort((a, b) => a.decade - b.decade)
      .map((entry) => ({
        field,
        decade: entry.decade,
        count: countFor(field, entry.decade),
        words: entry.words,
      })),
  );
};

/** Words that appear among the first `k` neighbours of more than one field. */
export const sharedWords = (columns: NeighbourColumn[], k: number): Set<string> => {
  const fieldsByWord = new Map<string, Set<string>>();
  for (const column of columns) {
    for (const word of column.words.slice(0, k)) {
      const fields = fieldsByWord.get(word) ?? new Set<string>();
      fields.add(column.field);
      fieldsByWord.set(word, fields);
    }
  }
  return new Set(
    [...fieldsByWord.entries()].filter(([, fields]) => fields.size > 1).map(([word]) => word),
  );
};

/** One field's consecutive-decade transitions, in order. */
export const trajectory = (data: WordData, layer: Layer, field: string): WithinRow[] =>
  data.within
    .filter((row) => row.layer === layer && row.field === field)
    .sort((a, b) => a.decadeA - b.decadeA);

export interface FrequencyPoint {
  decade: number;
  perMillion: Partial<Record<string, number>>;
}

/** Occurrences per million tokens per decade, one entry per field, for a line chart. */
export const frequency = (data: WordData, layer: Layer, fields: string[]): FrequencyPoint[] => {
  const decades = unique(
    data.counts.filter((count) => count.layer === layer).map((count) => count.decade),
  ).sort((a, b) => a - b);
  return decades.map((decade) => ({
    decade,
    perMillion: Object.fromEntries(
      fields.flatMap((field) => {
        const count = data.counts.find(
          (item) => item.layer === layer && item.field === field && item.decade === decade,
        );
        return count ? [[field, count.perMillion]] : [];
      }),
    ),
  }));
};

export interface DivergenceMatrix {
  fields: string[];
  cell: (a: string, b: string) => CrossRow | undefined;
}

/** Cross-field rows of one decade as a symmetric lookup over the fields measured in it. */
export const divergence = (data: WordData, decade: number): DivergenceMatrix => {
  const rows = data.cross.filter((row) => row.decade === decade);
  const fields = unique(rows.flatMap((row) => [row.fieldA, row.fieldB])).sort();
  const byPair = new Map(rows.map((row) => [`${row.fieldA}|${row.fieldB}`, row]));
  return {
    fields,
    cell: (a, b) => byPair.get(`${a}|${b}`) ?? byPair.get(`${b}|${a}`),
  };
};

export const divergenceDecades = (data: WordData): number[] =>
  unique(data.cross.map((row) => row.decade)).sort((a, b) => a - b);
