import type { StreamClient } from "../data/ws";
import { addHeader } from "./_panel";

export const mountStatus = async (root: HTMLDivElement, _stream: StreamClient) => {
    addHeader(root, "status");
    const placeholder = document.createElement("div");
    placeholder.style.cssText = "flex:1;display:flex;align-items:center;justify-content:center;color:var(--soft);";
    placeholder.textContent = "waiting for orchestration";
    root.appendChild(placeholder);
};
