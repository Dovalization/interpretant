"use client";

import { fieldLabel, fieldShortLabel } from "@/lib/fields";
import { cn } from "@/lib/utils";

import type { FallRow } from "@/lib/schema";

interface FallBarsProps {
  rows: FallRow[];
  highlight?: string;
}

const format = (value: number) => (value >= 0 ? "+" : "−") + Math.abs(value).toFixed(3);

/**
 * How much better each field's model predicts one text between two decades, split into the
 * part due to the text's word mix moving (slot 1) and the part due to the model's per-word
 * skill changing (slot 2). Bars are the means over seeds; the seed range is printed.
 */
export function FallBars({ rows, highlight }: FallBarsProps) {
  const sorted = [...rows].sort((a, b) => b.fall - a.fall);
  const scale = Math.max(
    ...sorted.flatMap((row) => [Math.abs(row.mix), Math.abs(row.rate), Math.abs(row.fall)]),
    0.001,
  );
  const width = (value: number) => `${(Math.abs(value) / scale) * 50}%`;
  const offset = (value: number) =>
    value >= 0 ? "50%" : `${50 - (Math.abs(value) / scale) * 50}%`;
  const text = sorted[0] ? fieldLabel(sorted[0].text) : "the text";

  return (
    <div className="flex flex-col gap-2">
      <div className="mb-1 flex items-center gap-4 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1.5">
          <span className="size-2.5 rounded-sm" style={{ backgroundColor: "var(--chart-1)" }} />
          word mix moved
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="size-2.5 rounded-sm" style={{ backgroundColor: "var(--chart-2)" }} />
          model&apos;s per-word skill changed
        </span>
      </div>
      {/* Axis: what each side of the centre line means. Same columns as the rows below. */}
      <div className="grid grid-cols-[1fr_auto] gap-x-3 text-xs text-muted-foreground sm:grid-cols-[9rem_1fr_5.5rem]">
        <span className="hidden sm:block" />
        <div className="col-span-2 grid grid-cols-2 sm:col-span-1">
          <span className="pr-2 text-right">← predicts {text} worse</span>
          <span className="pl-2">predicts {text} better →</span>
        </div>
        <span className="hidden text-right sm:block">total</span>
      </div>
      {sorted.map((row) => {
        const seeds = [...row.seeds].sort((a, b) => a - b);
        return (
          <div
            key={row.model}
            className={cn(
              "grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1 text-sm sm:grid-cols-[9rem_1fr_5.5rem] sm:gap-y-0",
              highlight === row.model && "font-semibold",
            )}
          >
            <span
              className="truncate text-muted-foreground sm:text-right"
              title={fieldLabel(row.model)}
            >
              {fieldShortLabel(row.model)}
            </span>
            {/* On phones the bar takes its own full-width row under the label and total. */}
            <div className="relative order-last col-span-2 h-7 sm:order-none sm:col-span-1">
              <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
              <div
                className="absolute top-0.5 h-2.5 rounded-sm"
                style={{
                  left: offset(row.mix),
                  width: width(row.mix),
                  backgroundColor: "var(--chart-1)",
                }}
                title={`word mix: ${format(row.mix)} bits per token`}
              />
              <div
                className="absolute bottom-0.5 h-2.5 rounded-sm"
                style={{
                  left: offset(row.rate),
                  width: width(row.rate),
                  backgroundColor: "var(--chart-2)",
                }}
                title={`per-word skill: ${format(row.rate)} bits per token`}
              />
            </div>
            <span
              className="text-right tabular-nums"
              title={`seeds: ${seeds.map(format).join(", ")}`}
            >
              {format(row.fall)}
              <span className="block text-[11px] font-normal text-muted-foreground">
                {format(seeds[0])} to {format(seeds[seeds.length - 1])}
              </span>
            </span>
          </div>
        );
      })}
      <p className="mt-1 text-xs text-muted-foreground">
        Each bar is one part of the change from the {sorted[0]?.early ?? 1990}s to the{" "}
        {sorted[0]?.late ?? 2020}s; the two parts add up to the total at the right, with the range
        over the five seeds beneath it. Bars pointing opposite ways mean the parts pulled against
        each other. Hover a bar for its value.
      </p>
    </div>
  );
}
