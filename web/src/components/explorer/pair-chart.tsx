"use client";

import { CartesianGrid, ComposedChart, Line, Scatter, Tooltip, XAxis, YAxis } from "recharts";

import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { decadeLabel, fieldColor, fieldLabel } from "@/lib/fields";

import type { PairPoint } from "@/lib/perplexity";

interface PairChartProps {
  a: string;
  b: string;
  points: PairPoint[];
}

/**
 * Excess cross-entropy over the decades for both directions of a pair: the five seeds as
 * points, their mean as a line. Each direction takes the colour of the field whose model does
 * the predicting.
 */
export function PairChart({ a, b, points }: PairChartProps) {
  const forwardKey = "forward";
  const backwardKey = "backward";
  const data = points.map((point) => ({
    decade: decadeLabel(point.decade),
    [forwardKey]: point.forward?.mean,
    [backwardKey]: point.backward?.mean,
  }));
  const seeds = (direction: "forward" | "backward") =>
    points.flatMap((point) =>
      (point[direction]?.seeds ?? []).map((value) => ({
        decade: decadeLabel(point.decade),
        value,
      })),
    );
  const config = {
    [forwardKey]: { label: `${fieldLabel(a)} → ${fieldLabel(b)}`, color: fieldColor(a) },
    [backwardKey]: { label: `${fieldLabel(b)} → ${fieldLabel(a)}`, color: fieldColor(b) },
  };

  return (
    <ChartContainer config={config} className="aspect-[2/1] w-full">
      <ComposedChart data={data} margin={{ top: 12, right: 16, bottom: 0, left: -8 }}>
        <CartesianGrid vertical={false} />
        <XAxis
          dataKey="decade"
          tickLine={false}
          axisLine={false}
          fontSize={11}
          allowDuplicatedCategory={false}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          width={44}
          domain={["auto", "auto"]}
          tickFormatter={(value: number) => value.toFixed(2)}
        />
        <Tooltip content={<ChartTooltipContent />} cursor={{ stroke: "var(--color-border)" }} />
        <ChartLegend content={<ChartLegendContent />} />
        <Scatter
          data={seeds("forward")}
          dataKey="value"
          fill={fieldColor(a)}
          opacity={0.45}
          legendType="none"
          isAnimationActive={false}
        />
        <Scatter
          data={seeds("backward")}
          dataKey="value"
          fill={fieldColor(b)}
          opacity={0.45}
          legendType="none"
          isAnimationActive={false}
        />
        <Line
          type="linear"
          dataKey={forwardKey}
          stroke={fieldColor(a)}
          strokeWidth={2}
          dot={false}
          connectNulls
          isAnimationActive={false}
        />
        <Line
          type="linear"
          dataKey={backwardKey}
          stroke={fieldColor(b)}
          strokeWidth={2}
          dot={false}
          connectNulls
          isAnimationActive={false}
        />
      </ComposedChart>
    </ChartContainer>
  );
}
