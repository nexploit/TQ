# TurboQuant / llama.cpp GGUF Manager

Ein Python-Tool zum Verwalten, Starten, Benchmarken und Quantisieren von GGUF-Modellen mit einem lokalen `llama.cpp`-Build.

## Funktionen

- findet GGUF-Modelle in einem Ordner
- startet Modelle mit `llama-cli`
- benchmarked KV-Cache-Typen mit `llama-bench`
- quantisiert Modelle mit `llama-quantize`
- erkennt unterstützte KV- und Quantisierungstypen aus der llama.cpp-Hilfe
- schreibt Logs und Benchmark-Ergebnisse als CSV

## Voraussetzungen

- Python 3.10 oder neuer
- ein lokaler llama.cpp-Build mit:
  - `llama-cli.exe`
  - `llama-bench.exe`
  - `llama-quantize.exe`
- Google's Quellcode angepasst 
  https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant
  muss auf dem jeweiligen System kompiliert werden da 
  es die Basis des Python Skriptes stellt (llama*.exe)

## Beispiele

Interaktiver Modus:

```powershell
python .\turboquant_manager.py --bin-dir "C:\pfad\zu\llama.cpp\build\bin" interactive --models "D:\Models"
```

Modell starten:

```powershell
python .\turboquant_manager.py --bin-dir "C:\pfad\zu\llama.cpp\build\bin" run "D:\Models\model.gguf" --kv turbo3 --ctx 32768
```

Benchmark:

```powershell
python .\turboquant_manager.py --bin-dir "C:\pfad\zu\llama.cpp\build\bin" benchmark "D:\Models\model.gguf" --kv f16 turbo4 turbo3 turbo2 --ctx 32768 --repetitions 3 --csv .\benchmark-results\results.csv
```

Quantisierung:

```powershell
python .\turboquant_manager.py --bin-dir "C:\pfad\zu\llama.cpp\build\bin" quantize "D:\Models\input.gguf" "D:\Models\output-Q4_K_M.gguf" Q4_K_M
```

Dry-Run:

```powershell
python .\turboquant_manager.py --bin-dir "C:\pfad\zu\llama.cpp\build\bin" dry-run "D:\Models\input.gguf" Q4_K_M
```

## Hinweise

- Der Standardwert fuer GPU-Layer ist `99`, da llama.cpp bei `-ngl` normalerweise eine Zahl erwartet.
- `llama-bench` bekommt fuer Flash Attention aus Kompatibilitaetsgruenden `1` oder `0`; `llama-cli` nutzt weiter `on`, `off` oder `auto`.
- Logdateien landen standardmaessig in `logs/`.
- Benchmark-CSV-Dateien koennen in `benchmark-results/` abgelegt werden.
