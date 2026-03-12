import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <h1 className="text-7xl font-bold text-primary">404</h1>
      <p className="mt-4 text-xl text-text-muted">Page not found</p>
      <p className="mt-2 text-sm text-text-muted">
        The page you&apos;re looking for doesn&apos;t exist or has been moved.
      </p>
      <Link
        href="/"
        className="mt-8 rounded-lg bg-primary px-6 py-2.5 text-sm font-medium text-white transition-colors hover:bg-primary/90"
      >
        Back to Home
      </Link>
    </div>
  );
}
