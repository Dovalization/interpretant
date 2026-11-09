import type { Perplexity, PerplexityRow } from "@/lib/schema";

/** Pure helpers over the corrected cross-perplexity table. */

const mean = (values: number[]): number =>
  values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;

export interface PerplexityMatrix {
  fields: string[];
  max: number;
  cell: (model: string, text: string) => PerplexityRow | undefined;
}

/** The fields that qualify in a decade and a lookup of E(model → text) rows. */
export const perplexityMatrix = (table: Perplexity, decade: number): PerplexityMatrix => {
  const rows = table.rows.filter((row) => row.decade === decade && row.model !== row.text);
  const fields = [...new Set(rows.flatMap((row) => [row.model, row.text]))].sort();
  const byPair = new Map(rows.map((row) => [`${row.model}|${row.text}`, row]));
  return {
    fields,
    max: Math.max(...rows.map((row) => mean(row.excess)), 0),
    cell: (model, text) => byPair.get(`${model}|${text}`),
  };
};

export interface PairPoint {
  decade: number;
  /** Mean excess of B's text under A's model, and the seed values behind it. */
  forward?: { mean: number; seeds: number[] };
  /** Mean excess of A's text under B's model. */
  backward?: { mean: number; seeds: number[] };
}

/** Both directions of a pair over the decades where either exists. */
export const pairSeries = (table: Perplexity, a: string, b: string): PairPoint[] => {
  const decades = [
    ...new Set(
      table.rows
        .filter((row) => (row.model === a && row.text === b) || (row.model === b && row.text === a))
        .map((row) => row.decade),
    ),
  ].sort((x, y) => x - y);
  const find = (model: string, text: string, decade: number) =>
    table.rows.find((row) => row.model === model && row.text === text && row.decade === decade);
  return decades.map((decade) => {
    const forward = find(a, b, decade);
    const backward = find(b, a, decade);
    return {
      decade,
      forward: forward && { mean: mean(forward.excess), seeds: forward.excess },
      backward: backward && { mean: mean(backward.excess), seeds: backward.excess },
    };
  });
};

export const perplexityMean = mean;
