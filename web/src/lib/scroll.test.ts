import { describe, expect, it } from "vitest";

import { scrollEdges, scrollMask } from "@/lib/scroll";

describe("scroll hints", () => {
  it("reports which sides hide content", () => {
    expect(scrollEdges(0, 300, 300)).toEqual({ hiddenStart: false, hiddenEnd: false });
    expect(scrollEdges(0, 300, 900)).toEqual({ hiddenStart: false, hiddenEnd: true });
    expect(scrollEdges(300, 300, 900)).toEqual({ hiddenStart: true, hiddenEnd: true });
    expect(scrollEdges(600, 300, 900)).toEqual({ hiddenStart: true, hiddenEnd: false });
  });

  it("fades only the sides that hide content", () => {
    expect(scrollMask({ hiddenStart: false, hiddenEnd: false })).toBeUndefined();
    expect(scrollMask({ hiddenStart: false, hiddenEnd: true })).toMatch(
      /^linear-gradient\(to right, black,.*transparent\)$/,
    );
    expect(scrollMask({ hiddenStart: true, hiddenEnd: false })).toMatch(
      /^linear-gradient\(to right, transparent,.*black\)$/,
    );
  });
});
