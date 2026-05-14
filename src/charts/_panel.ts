export const addHeader = (root: HTMLDivElement, title: string): HTMLDivElement => {
    const header = document.createElement("div");
    header.className = "dash-panel-header";
    header.textContent = title;
    root.appendChild(header);
    return header;
};

export const addChartHost = (root: HTMLDivElement): HTMLDivElement => {
    const host = document.createElement("div");
    host.className = "dash-panel-chart";
    root.appendChild(host);
    return host;
};
