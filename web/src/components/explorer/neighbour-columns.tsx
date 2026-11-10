"use client";

import { useState } from "react";

import { HorizontalScroll } from "@/components/horizontal-scroll";
import { decadeLabel, fieldColor, fieldLabel } from "@/lib/fields";
import { cn } from "@/lib/utils";

import type { NeighbourColumn } from "@/lib/explorer";

interface NeighbourColumnsProps {
  columns: NeighbourColumn[];
  shared: Set<string>;
  k: number;
}

const number = new Intl.NumberFormat("en-US");

/**
 * Ranked neighbour lists, one column per cell, grouped by field. Words shared across fields
 * are muted; hovering a word highlights every list it appears in.
 */
export function NeighbourColumns({ columns, shared, k }: NeighbourColumnsProps) {
  const [hovered, setHovered] = useState<string | null>(null);
  const groups = columns.reduce<{ field: string; columns: NeighbourColumn[] }[]>((acc, column) => {
    const last = acc[acc.length - 1];
    return last && last.field === column.field
      ? [...acc.slice(0, -1), { field: last.field, columns: [...last.columns, column] }]
      : [...acc, { field: column.field, columns: [column] }];
  }, []);

  if (columns.length === 0) {
    return (
      <p className="py-10 text-center text-muted-foreground">
        No cell measures this word for the chosen fields and layer (it needs 150 occurrences).
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-8" onMouseLeave={() => setHovered(null)}>
      {groups.map((group) => (
        <section key={group.field}>
          <h3 className="mb-2 text-base font-bold" style={{ color: fieldColor(group.field) }}>
            {fieldLabel(group.field)}
          </h3>
          <HorizontalScroll className="flex gap-6 pb-2">
            {group.columns.map((column) => (
              <div key={column.decade} className="w-32 shrink-0">
                <div
                  className="mb-2 border-b-2 pb-1"
                  style={{ borderColor: fieldColor(group.field) }}
                >
                  <p className="font-semibold text-foreground">{decadeLabel(column.decade)}</p>
                  <p className="text-xs text-muted-foreground tabular-nums">
                    {number.format(column.count)} occurrences
                  </p>
                </div>
                <ol className="space-y-0.5">
                  {column.words.slice(0, k).map((word) => {
                    const isShared = shared.has(word);
                    const isHovered = hovered === word;
                    return (
                      <li
                        key={word}
                        onMouseEnter={() => setHovered(word)}
                        onFocus={() => setHovered(word)}
                        tabIndex={0}
                        className={cn(
                          "-mx-1 rounded px-1 text-[15px] leading-6 transition-colors",
                          isShared ? "text-muted-foreground" : "text-foreground",
                          isHovered && "bg-accent text-accent-foreground",
                        )}
                      >
                        {word.replaceAll("_", " ")}
                      </li>
                    );
                  })}
                </ol>
              </div>
            ))}
          </HorizontalScroll>
        </section>
      ))}
    </div>
  );
}
