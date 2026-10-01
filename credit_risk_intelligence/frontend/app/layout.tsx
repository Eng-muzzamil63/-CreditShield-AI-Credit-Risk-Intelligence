import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = { title: 'CreditShield AI', description: 'Credit risk intelligence dashboard' };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
