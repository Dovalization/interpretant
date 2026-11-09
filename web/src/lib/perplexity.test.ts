import { describe, expect, it } from "vitest";

import { pairSeries, perplexityMatrix } from "@/lib/perplexity";

import type { Perplexity } from "@/lib/schema";

const TABLE: Perplexity = {
  fields: ["hci", "philosophy"],
  decades: [1990, 2020],
  rows: [
    { decade: 1990, model: "philosophy", text: "hci", excess: [1.7, 1.72] },
    { decade: 1990, model: "hci", text: "philosophy", excess: [1.3, 1.26] },
    { decade: 1990, model: "hci", text: "hci", excess: [0, 0] },
    { decade: 2020, model: "philosophy", text: "hci", excess: [1.6, 1.64] },
  ],
};

describe("perplexity helpers", () => {
  it("builds a matrix of the decade's qualifying fields without the diagonal", () => {
    const matrix = perplexityMatrix(TABLE, 1990);

    expect(matrix.fields).toEqual(["hci", "philosophy"]);
    expect(matrix.cell("philosophy", "hci")?.excess).toEqual([1.7, 1.72]);
    expect(matrix.cell("hci", "hci")).toBeUndefined();
    expect(matrix.max).toBeCloseTo(1.71);
  });

  it("pairs both directions per decade with their means", () => {
    const series = pairSeries(TABLE, "philosophy", "hci");

    expect(series.map((point) => point.decade)).toEqual([1990, 2020]);
    expect(series[0].forward?.mean).toBeCloseTo(1.71);
    expect(series[0].backward?.mean).toBeCloseTo(1.28);
    expect(series[1].backward).toBeUndefined();
  });
});
