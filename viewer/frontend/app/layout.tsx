import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GLiNER2 Viewer",
  description:
    "Visualize GLiNER2 extractions: entities, relations, events, classifications, and structures.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
