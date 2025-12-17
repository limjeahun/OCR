import { OCRScanner } from '@/components/OCRScanner';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center py-12 px-4 bg-muted/10">
      <OCRScanner />
    </main>
  );
}
