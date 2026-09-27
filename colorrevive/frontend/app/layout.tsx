import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-toggle";

export const metadata: Metadata = {
  title: "ColorRevive — AI photo colorization",
  description:
    "Upload a black-and-white image and generate a natural-looking color version with an AI colorization model.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-canvas font-sans text-ink antialiased">
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
