import { z } from "zod";

/** Mirrors the shapes written by `interpretant export web` (src/interpretant/export.py). */

export const LayerSchema = z.enum(["books", "openalex", "openalex_full"]);
export type Layer = z.infer<typeof LayerSchema>;

export const MethodSchema = z.enum(["twec", "ppmi"]);
export type Method = z.infer<typeof MethodSchema>;

export const CellSchema = z.object({
  layer: LayerSchema,
  field: z.string(),
  decade: z.number().int(),
  documents: z.number().int().nullable(),
  tokens: z.number().int(),
  vocabulary: z.number().int(),
  pool: z.number().int(),
});
export type Cell = z.infer<typeof CellSchema>;

export const CorpusSchema = z.object({
  fields: z.array(z.object({ slug: z.string(), label: z.string() })),
  decades: z.array(z.number().int()),
  cells: z.array(CellSchema),
});
export type Corpus = z.infer<typeof CorpusSchema>;

export const WordSummarySchema = z.object({
  word: z.string(),
  tier: z.string(),
  measurableCells: z.number().int(),
});
export type WordSummary = z.infer<typeof WordSummarySchema>;

export const WordsSchema = z.object({ words: z.array(WordSummarySchema) });

export const CountSchema = z.object({
  layer: LayerSchema,
  field: z.string(),
  decade: z.number().int(),
  count: z.number().int(),
  perMillion: z.number(),
});
export type Count = z.infer<typeof CountSchema>;

export const NeighbourListSchema = z.object({
  layer: LayerSchema,
  field: z.string(),
  decade: z.number().int(),
  method: MethodSchema,
  words: z.array(z.string()),
});
export type NeighbourList = z.infer<typeof NeighbourListSchema>;

export const WithinRowSchema = z.object({
  layer: LayerSchema,
  field: z.string(),
  decadeA: z.number().int(),
  decadeB: z.number().int(),
  countA: z.number().int(),
  countB: z.number().int(),
  cosineDrift: z.number(),
  neighbourShift: z.number(),
  cosineRank: z.number(),
  shiftRank: z.number(),
  candidate: z.boolean(),
});
export type WithinRow = z.infer<typeof WithinRowSchema>;

export const CrossRowSchema = z.object({
  fieldA: z.string(),
  fieldB: z.string(),
  decade: z.number().int(),
  countA: z.number().int(),
  countB: z.number().int(),
  cosineRank: z.number(),
  shiftRank: z.number(),
  candidate: z.boolean(),
});
export type CrossRow = z.infer<typeof CrossRowSchema>;

export const WordDataSchema = z.object({
  word: z.string(),
  counts: z.array(CountSchema),
  neighbours: z.array(NeighbourListSchema),
  within: z.array(WithinRowSchema),
  cross: z.array(CrossRowSchema),
});
export type WordData = z.infer<typeof WordDataSchema>;

export const PerplexityRowSchema = z.object({
  decade: z.number().int(),
  model: z.string(),
  text: z.string(),
  excess: z.array(z.number()),
});
export type PerplexityRow = z.infer<typeof PerplexityRowSchema>;

export const PerplexitySchema = z.object({
  fields: z.array(z.string()),
  decades: z.array(z.number().int()),
  rows: z.array(PerplexityRowSchema),
});
export type Perplexity = z.infer<typeof PerplexitySchema>;

export const VerdictSchema = z.enum(["passed", "falsified", "untestable"]);
export type Verdict = z.infer<typeof VerdictSchema>;

export const PredictionSchema = z.object({
  id: z.string(),
  round: z.string(),
  title: z.string(),
  verdict: VerdictSchema.optional(),
  note: z.string().optional(),
  numbers: z.string().optional(),
  file: z.string().optional(),
});
export type Prediction = z.infer<typeof PredictionSchema>;

export const PredictionsSchema = z.object({
  rounds: z.array(z.object({ id: z.string(), title: z.string(), subtitle: z.string() })),
  predictions: z.array(PredictionSchema),
});
export type Predictions = z.infer<typeof PredictionsSchema>;
