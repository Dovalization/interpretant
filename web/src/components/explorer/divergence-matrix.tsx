"use client";

import { HorizontalScroll } from "@/components/horizontal-scroll";
import { fieldLabel, fieldShortLabel } from "@/lib/fields";
import { cn } from "@/lib/utils";

import type { DivergenceMatrix as Matrix } from "@/lib/explorer";

interface DivergenceMatrixProps {
  matrix: Matrix;
  highlight: string[];
}

const RAMP = [
  "var(--ramp-1)",
  "var(--ramp-2)",
  "var(--ramp-3)",
  "var(--ramp-4)",
  "var(--ramp-5)",
  "var(--ramp-6)",
  "var(--ramp-7)",
];

/**
 * Field × field: how differently two fields use the word in one decade, as the percentile
 * rank of its cosine distance against the pair's noise floor (one blue ramp). A ring marks a
 * candidate: top 5% on both measures.
 */
export function DivergenceMatrix({ matrix, highlight }: DivergenceMatrixProps) {
  const { fields, cell } = matrix;
  if (fields.length === 0) {
    return (
      <p className="py-6 text-sm text-muted-foreground">
        No pair measures this word in this decade.
      </p>
    );
  }
  return (
    <HorizontalScroll>
      <table className="border-separate border-spacing-0.5 text-xs">
        <thead>
          <tr>
            <th scope="col" className="sr-only">
              Field
            </th>
            {fields.map((field) => (
              <th
                key={field}
                scope="col"
                className="h-28 align-bottom font-semibold text-muted-foreground"
              >
                <span className="inline-block w-8 origin-bottom-left -rotate-45 whitespace-nowrap">
                  {fieldShortLabel(field)}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {fields.map((rowField) => (
            <tr key={rowField}>
              <th
                scope="row"
                className={cn(
                  "pr-2 text-right font-semibold whitespace-nowrap",
                  highlight.includes(rowField) ? "text-foreground" : "text-muted-foreground",
                )}
              >
                {fieldLabel(rowField)}
              </th>
              {fields.map((columnField) => {
                const row = rowField === columnField ? undefined : cell(rowField, columnField);
                const step = row ? Math.min(6, Math.floor((row.cosineRank / 100) * 7)) : -1;
                return (
                  <td
                    key={columnField}
                    title={
                      row
                        ? `${fieldLabel(rowField)} × ${fieldLabel(columnField)}: cosine rank ${Math.round(row.cosineRank)}, shift rank ${Math.round(row.shiftRank)}${row.candidate ? ", candidate" : ""}`
                        : undefined
                    }
                    className={cn(
                      "size-8 rounded-sm text-center tabular-nums",
                      step < 0 && "bg-muted/40",
                      step >= 4 ? "text-white" : "text-neutral-950",
                      row?.candidate && "ring-2 ring-foreground ring-inset",
                    )}
                    style={step >= 0 ? { backgroundColor: RAMP[step] } : undefined}
                  >
                    {row ? Math.round(row.cosineRank) : ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </HorizontalScroll>
  );
}
