import type { CSSProperties } from "react";

// Deterministic, theme-agnostic label color. We return an HSL hue and use it as
// a translucent background + solid underline in CSS (var --tagc), so it reads on
// both light and dark backgrounds without a fixed palette.
const HUES = [210, 145, 275, 25, 340, 190, 95, 300, 55, 165, 240, 15, 120, 320];

export function labelHue(label: string): number {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0;
  return HUES[h % HUES.length];
}

// A solid, readable color for the given label (used for underline/dot/chip border).
export function labelColor(label: string): string {
  return `hsl(${labelHue(label)} 70% 45%)`;
}

// Inline style setting the --tagc CSS variable consumed by .tag/.chip/.dot.
export function tagStyle(label: string): CSSProperties {
  return { ["--tagc" as any]: labelColor(label) };
}

export function fmtConf(c?: number): string {
  return c == null ? "" : c.toFixed(2);
}
