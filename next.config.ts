import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,

  // Exclude onnxruntime-web from Turbopack bundling (fixes Worker compatibility)
  serverExternalPackages: ['onnxruntime-web'],

  // Empty turbopack config to silence warning when webpack config is present
  turbopack: {},

  // Webpack configuration for client-side
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Prevent webpack from bundling onnxruntime-web's Worker code
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
    }
    return config;
  },

  // Enable Cross-Origin Isolation for SharedArrayBuffer (WASM multi-threading)
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          {
            key: "Cross-Origin-Opener-Policy",
            value: "same-origin",
          },
          {
            key: "Cross-Origin-Embedder-Policy",
            value: "require-corp",
          },
        ],
      },
    ];
  },
};

export default nextConfig;

