"use client";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

import type { ReactNode } from "react";

interface Option<T extends string> {
  value: T;
  label: ReactNode;
  disabled?: boolean;
}

interface SegmentedProps<T extends string> {
  label: string;
  value: T;
  options: readonly Option<T>[];
  onChange: (value: T) => void;
}

/** One choice among a few: a labelled toggle group that always keeps one item pressed. */
export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: SegmentedProps<T>) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
        {label}
      </span>
      <ToggleGroup
        value={[value]}
        onValueChange={(next) => next.length > 0 && onChange(next[next.length - 1] as T)}
        variant="outline"
        size="sm"
        className="flex-wrap"
        aria-label={label}
      >
        {options.map((option) => (
          <ToggleGroupItem key={option.value} value={option.value} disabled={option.disabled}>
            {option.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  );
}

interface ChipsProps {
  label: string;
  value: string[];
  options: readonly Option<string>[];
  max?: number;
  onChange: (value: string[]) => void;
  colorFor?: (value: string) => string;
}

/** Several choices, at least one kept: toggle chips, each with its entity's colour as a dot. */
export function Chips({ label, value, options, max, onChange, colorFor }: ChipsProps) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
        {label}
      </span>
      <ToggleGroup
        multiple
        value={value}
        onValueChange={(next) => next.length > 0 && onChange(max ? next.slice(-max) : next)}
        variant="outline"
        size="sm"
        className="flex-wrap"
        aria-label={label}
      >
        {options.map((option) => (
          <ToggleGroupItem key={option.value} value={option.value} disabled={option.disabled}>
            {colorFor && (
              <span
                aria-hidden
                className="size-2 rounded-full"
                style={{ backgroundColor: colorFor(option.value) }}
              />
            )}
            {option.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  );
}
