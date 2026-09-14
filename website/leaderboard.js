const results = [["Gemini 3.1 Pro", 76.5, 72.7, 35.9, 2.9, 4.0, 28.3], ["GPT-5.5", 74.1, 77.3, 31.1, 1.3, 1.5, 27.3], ["Claude Opus 4.8", 77.9, 73.5, 21.0, 1.6, 2.5, 25.6], ["MiniMax M3", 77.0, 68.7, 13.3, 1.9, 2.5, 21.8], ["Qwen 3.6", 67.4, 64.8, 15.2, 1.0, 1.5, 16.4], ["Mistral Large 3", 65.2, 68.5, 11.0, 0.0, 0.5, 15.8], ["Llama 4 Maverick", 66.8, 66.0, 8.7, 0.0, 0.0, 14.9], ["Kimi K2.6", 60.2, 57.8, 11.7, 0.7, 0.5, 9.7], ["Grok 4.3 Fast", 52.3, 50.0, 2.6, 0.0, 0.0, 1.4]];
const body = document.getElementById("leaderboard");
for (const row of results) {
  const tr = document.createElement("tr");
  row.forEach((value, index) => {
    const cell = document.createElement(index === 0 ? "th" : "td");
    if (index === 0) cell.scope = "row";
    cell.textContent = typeof value === "number" ? value.toFixed(1) : value;
    tr.appendChild(cell);
  });
  body.appendChild(tr);
}
