import { readFileSync } from "node:fs";
import { existsSync } from "node:fs";
import path from "node:path";

import {
  type Corpus,
  CorpusSchema,
  type Falls,
  FallsSchema,
  type Perplexity,
  PerplexitySchema,
  type Predictions,
  PredictionsSchema,
  type WordData,
  WordDataSchema,
  WordsSchema,
  type WordSummary,
} from "@/lib/schema";

import type { ZodType } from "zod";

import "server-only";

/**
 * The explorer's data is the JSON written by `interpretant export web` into `web/data`,
 * read at build time (the site is a static export) and validated once per file.
 */
const DATA_DIR = path.join(process.cwd(), "data");

const cache = new Map<string, unknown>();

function readJson<T>(relativePath: string, schema: ZodType<T>): T {
  const cached = cache.get(relativePath);
  if (cached !== undefined) return cached as T;
  const raw = readFileSync(path.join(DATA_DIR, relativePath), "utf-8");
  const parsed = schema.parse(JSON.parse(raw));
  cache.set(relativePath, parsed);
  return parsed;
}

export const getCorpus = (): Corpus => readJson("corpus.json", CorpusSchema);

export const getWords = (): WordSummary[] => readJson("words.json", WordsSchema).words;

export const getWord = (word: string): WordData =>
  readJson(path.join("words", `${word}.json`), WordDataSchema);

export const getPerplexity = (): Perplexity => readJson("perplexity.json", PerplexitySchema);

export const getPredictions = (): Predictions => readJson("predictions.json", PredictionsSchema);

/** The mix/rate decomposition of every pair's fall; absent until `drift decompose` has run. */
export const getFalls = (): Falls =>
  existsSync(path.join(DATA_DIR, "falls.json"))
    ? readJson("falls.json", FallsSchema)
    : { rows: [] };
