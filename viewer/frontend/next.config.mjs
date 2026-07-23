/** @type {import('next').NextConfig} */

// Same-origin API proxy. The browser calls "/api/*" on the page's own origin
// (:3000) and Next forwards each request to the co-located backend. This makes
// the viewer work however it is reached -- localhost, SSH tunnel, box IP, or a
// root-mounted (subdomain) reverse proxy -- with no backend host in the client
// and no CORS, because the browser only ever talks to the origin it loaded from.
// (A path-prefix reverse proxy, e.g. host/gliner/, would additionally need Next
// basePath + a prefix-aware API base; not configured here.)
//
// Backend target: GLINER2_BACKEND_ORIGIN, default http://127.0.0.1:8000 (the
// co-located backend). NOTE the timing: `next dev` reads this env at startup,
// but `next build` bakes the destination into .next/routes-manifest.json, so
// `next start` (the Docker/production path) uses the value present AT BUILD --
// set it via --build-arg to retarget a split-container deployment. Trailing
// slashes are stripped so "http://host:8000/" does not become a 404-everything
// double slash in the destination.
const BACKEND_ORIGIN = (process.env.GLINER2_BACKEND_ORIGIN || "http://127.0.0.1:8000").replace(/\/+$/, "");

const nextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${BACKEND_ORIGIN}/:path*` }];
  },
};

export default nextConfig;
