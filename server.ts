import index from "./index.html";

const wasm = (path: string) =>
    new Response(Bun.file(path), {
        headers: { "Content-Type": "application/wasm" },
    });

const server = Bun.serve({
    port: Number(process.env.PORT ?? 3001),
    development: true,
    routes: {
        "/scichart2d.wasm": () => wasm("./public/scichart2d.wasm"),
        "/scichart2d-nosimd.wasm": () => wasm("./public/scichart2d-nosimd.wasm"),
        "/": index,
    },
});

console.log(`dev server: http://localhost:${server.port}`);
