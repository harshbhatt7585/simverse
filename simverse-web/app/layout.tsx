import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Simverse",
  description: "Simulation-first UI for replay and live environment visualization.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
