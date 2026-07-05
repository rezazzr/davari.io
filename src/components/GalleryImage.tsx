"use client";

import { useEffect, useRef, useState } from "react";
import Image from "next/image";

interface GalleryImageProps {
  src: string;
  alt: string;
  /** optional tiny base64 preview for a true blur-up (see notes in baby-fund.ts) */
  blurDataURL?: string;
  /** "cover" crops to fill the 4:3 tile; "contain" shows the whole scan */
  fit?: "cover" | "contain";
}

export default function GalleryImage({
  src,
  alt,
  blurDataURL,
  fit = "cover",
}: GalleryImageProps) {
  const [loaded, setLoaded] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    // If the image is already cached (e.g. prefetched from the home page),
    // onLoad may never fire, so reveal it immediately.
    if (imgRef.current?.complete) setLoaded(true);
  }, []);

  return (
    <>
      {!loaded && (
        <div
          aria-hidden
          className="absolute inset-0 animate-pulse bg-linear-to-br from-primary/10 via-secondary/10 to-black/5 dark:to-white/10"
        />
      )}
      <Image
        ref={imgRef}
        src={src}
        alt={alt}
        fill
        sizes="(max-width: 640px) 100vw, 33vw"
        placeholder={blurDataURL ? "blur" : "empty"}
        blurDataURL={blurDataURL}
        onLoad={() => setLoaded(true)}
        className={`${fit === "contain" ? "object-contain" : "object-cover"} transition-opacity duration-700 ${
          loaded ? "opacity-100" : "opacity-0"
        }`}
      />
    </>
  );
}
