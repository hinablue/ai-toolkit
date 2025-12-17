import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  devIndicators: {
    buildActivity: false,
  },
  typescript: {
    // Remove this. Build fails because of route types
    ignoreBuildErrors: true,
  },
  experimental: {
    serverActions: {
      bodySizeLimit: '100mb',
    },
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      // Externalize native modules to prevent webpack from trying to bundle them
      config.externals = config.externals || [];
      config.externals.push('osx-temperature-sensor');
    }
    return config;
  },
};

export default nextConfig;
