import { describe, expect, it } from "vitest";

import {
  defaultFields,
  defaultLayer,
  divergence,
  fieldsInLayer,
  frequency,
  neighbourColumns,
  sharedWords,
} from "@/lib/explorer";

import type { WordData } from "@/lib/schema";

const within = (field: string, decadeA: number, candidate: boolean) => ({
  layer: "openalex_full" as const,
  field,
  decadeA,
  decadeB: decadeA + 10,
  countA: 200,
  countB: 300,
  cosineDrift: 0.3,
  neighbourShift: 0.5,
  cosineRank: candidate ? 98 : 40,
  shiftRank: candidate ? 97 : 50,
  candidate,
});

const DATA: WordData = {
  word: "embodiment",
  counts: [
    { layer: "openalex_full", field: "hci", decade: 2010, count: 1018, perMillion: 60 },
    { layer: "openalex_full", field: "hci", decade: 2020, count: 2809, perMillion: 200 },
    { layer: "openalex_full", field: "philosophy", decade: 2020, count: 2067, perMillion: 66 },
    { layer: "openalex", field: "hci", decade: 2020, count: 180, perMillion: 190 },
  ],
  neighbours: [
    {
      layer: "openalex_full",
      field: "hci",
      decade: 2010,
      method: "twec",
      words: ["ownership", "body"],
    },
    {
      layer: "openalex_full",
      field: "hci",
      decade: 2020,
      method: "twec",
      words: ["presence", "body"],
    },
    {
      layer: "openalex_full",
      field: "philosophy",
      decade: 2020,
      method: "twec",
      words: ["corporeality", "body"],
    },
    { layer: "openalex", field: "hci", decade: 2020, method: "twec", words: ["presence"] },
    { layer: "openalex_full", field: "hci", decade: 2020, method: "ppmi", words: ["illusion"] },
  ],
  within: [
    within("hci", 2000, true),
    within("hci", 2010, false),
    within("philosophy", 2010, false),
  ],
  cross: [
    {
      fieldA: "hci",
      fieldB: "philosophy",
      decade: 2020,
      countA: 180,
      countB: 160,
      cosineRank: 90,
      shiftRank: 80,
      candidate: false,
    },
  ],
};

describe("explorer helpers", () => {
  it("picks the layer with the most cells and the fields with the most candidates", () => {
    expect(defaultLayer(DATA)).toBe("openalex_full");
    expect(defaultFields(DATA, "openalex_full")).toEqual(["hci", "philosophy"]);
    expect(fieldsInLayer(DATA, "openalex_full", "ppmi")).toEqual(["hci"]);
  });

  it("opens a research layer even when the books have more lists, and books only as a fallback", () => {
    const book = (decade: number) => ({
      layer: "books" as const,
      field: "books",
      decade,
      method: "twec" as const,
      words: ["mind"],
    });
    const books = [1850, 1860, 1870, 1880, 1890].map(book);

    expect(defaultLayer({ ...DATA, neighbours: [...DATA.neighbours, ...books] })).toBe(
      "openalex_full",
    );
    expect(defaultLayer({ ...DATA, neighbours: books })).toBe("books");
  });

  it("chooses default fields among those the method has", () => {
    expect(defaultFields(DATA, "openalex_full", "ppmi")).toEqual(["hci"]);
  });

  it("builds columns per field in order, with counts, and finds the shared words", () => {
    const columns = neighbourColumns(DATA, "openalex_full", "twec", ["philosophy", "hci"]);

    expect(columns.map((column) => [column.field, column.decade, column.count])).toEqual([
      ["philosophy", 2020, 2067],
      ["hci", 2010, 1018],
      ["hci", 2020, 2809],
    ]);
    expect(sharedWords(columns, 12)).toEqual(new Set(["body"]));
    expect(sharedWords(columns, 1)).toEqual(new Set());
  });

  it("lays out frequency by decade and the divergence matrix symmetrically", () => {
    expect(frequency(DATA, "openalex_full", ["hci", "philosophy"])).toEqual([
      { decade: 2010, perMillion: { hci: 60 } },
      { decade: 2020, perMillion: { hci: 200, philosophy: 66 } },
    ]);
    const matrix = divergence(DATA, 2020);
    expect(matrix.fields).toEqual(["hci", "philosophy"]);
    expect(matrix.cell("philosophy", "hci")?.cosineRank).toBe(90);
    expect(matrix.cell("hci", "hci")).toBeUndefined();
  });
});
