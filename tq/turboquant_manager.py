from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import subprocess
import sys
import time

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


# =====================================================================
# TurboQuant / llama.cpp GGUF Manager
# erstellt von @xpmen99 PAN
# =====================================================================

PROGRAM_NAME = "TurboQuant / llama.cpp GGUF Manager"
PROGRAM_AUTHOR = "@xpmen99 PAN"
PROGRAM_VERSION = "1.2.0"

DEFAULT_CONTEXT = 32768
DEFAULT_GPU_LAYERS = "99"
DEFAULT_FLASH_ATTN = "on"
DEFAULT_REPETITIONS = 3
DEFAULT_LOG_LEVEL = "INFO"

DEFAULT_KV_TESTS = [
    "f16",
    "turbo4",
    "turbo3",
    "turbo2",
]

LOGGER_NAME = "TurboQuantManager"

logger = logging.getLogger(LOGGER_NAME)


# =====================================================================
# Prozessstatus
# =====================================================================

class ProcessStatus(Enum):
    EXIT_OK = "EXIT_OK"
    USER_ABORT = "USER_ABORT"
    TOOL_ERROR = "TOOL_ERROR"
    HELP_OK_NONZERO = "HELP_OK_NONZERO"


# =====================================================================
# Datenstrukturen
# =====================================================================

@dataclass
class Executables:
    cli: Path
    quantize: Path
    bench: Path
    perplexity: Optional[Path] = None
    imatrix: Optional[Path] = None
    server: Optional[Path] = None


@dataclass
class BuildCapabilities:
    kv_types: list[str]
    quant_types: list[str]
    cli_help: str
    quantize_help: str


@dataclass
class ProcessResult:
    return_code: int
    output: str
    duration_seconds: float
    status: ProcessStatus


@dataclass
class BenchResult:
    model: str
    kv_type: str
    context: int
    return_code: int
    duration_seconds: float
    raw_output: str
    tokens_per_second: Optional[float] = None


# =====================================================================
# Logging
# =====================================================================

def setup_logging(
    log_dir: Path,
    log_level: str = DEFAULT_LOG_LEVEL,
) -> Path:

    log_dir = log_dir.resolve()
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    log_file = (
        log_dir
        / f"turboquant_{timestamp}.log"
    )

    console_level = getattr(
        logging,
        log_level.upper(),
        logging.INFO,
    )

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    # ---------------------------------------------------------
    # Datei
    # ---------------------------------------------------------

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    file_handler.setLevel(
        logging.DEBUG
    )

    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(funcName)-24s | "
            "%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(
        file_handler
    )

    # ---------------------------------------------------------
    # Konsole
    # ---------------------------------------------------------

    console_handler = logging.StreamHandler(
        sys.stdout
    )

    console_handler.setLevel(
        console_level
    )

    console_handler.setFormatter(
        logging.Formatter(
            "%(levelname)-8s | %(message)s"
        )
    )

    logger.addHandler(
        console_handler
    )

    logger.info(
        "%s gestartet",
        PROGRAM_NAME,
    )

    logger.info(
        "Erstellt von %s",
        PROGRAM_AUTHOR,
    )

    logger.info(
        "Version: %s",
        PROGRAM_VERSION,
    )

    logger.info(
        "Logdatei: %s",
        log_file,
    )

    logger.debug(
        "Python: %s",
        sys.version.replace(
            "\n",
            " ",
        ),
    )

    logger.debug(
        "Python Executable: %s",
        sys.executable,
    )

    logger.debug(
        "Arbeitsverzeichnis: %s",
        Path.cwd(),
    )

    return log_file


# =====================================================================
# Hilfsfunktionen
# =====================================================================

def human_size(
    size: int,
) -> str:

    value = float(size)

    for unit in [
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
        "PiB",
    ]:

        if value < 1024:
            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} EiB"


def unique_preserve_order(
    values: list[str],
) -> list[str]:

    return list(
        dict.fromkeys(
            values
        )
    )


def normalize_choice(
    value: str,
    allowed_values: list[str],
) -> Optional[str]:

    value_lower = (
        value.strip()
        .lower()
    )

    for allowed in allowed_values:

        if allowed.lower() == value_lower:
            return allowed

    return None


def bench_flash_attn_value(
    flash_attn: str,
) -> str:

    mapping = {
        "on": "1",
        "off": "0",
        "auto": "auto",
    }

    return mapping.get(
        flash_attn.lower(),
        flash_attn,
    )


def positive_int(
    value: str,
) -> int:

    number = int(
        value
    )

    if number < 1:
        raise argparse.ArgumentTypeError(
            "Wert muss größer als 0 sein."
        )

    return number


def quote_command(
    command: list[str],
) -> str:

    return subprocess.list2cmdline(
        command
    )


def print_separator(
    title: Optional[str] = None,
) -> None:

    print()
    print("=" * 80)

    if title:
        print(title)
        print("=" * 80)


def print_program_header() -> None:

    print()
    print("=" * 80)
    print(
        "         TurboQuant / llama.cpp GGUF Manager"
    )
    print(
        "              erstellt von @xpmen99 PAN"
    )
    print(
        f"                    Version {PROGRAM_VERSION}"
    )
    print("=" * 80)


# =====================================================================
# Prozessausführung
# =====================================================================

def classify_process_status(
    return_code: int,
    user_abort: bool = False,
) -> ProcessStatus:

    if user_abort:
        return ProcessStatus.USER_ABORT

    if return_code == 0:
        return ProcessStatus.EXIT_OK

    return ProcessStatus.TOOL_ERROR


def run_capture(
    command: list[str],
    cwd: Optional[Path] = None,
    log_full_output: bool = False,
) -> ProcessResult:

    command_text = quote_command(
        command
    )

    logger.info(
        "Starte Prozess: %s",
        command_text,
    )

    start = time.perf_counter()

    try:

        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        duration = (
            time.perf_counter()
            - start
        )

        status = classify_process_status(
            result.returncode
        )

        logger.info(
            "Prozess beendet: "
            "Status=%s ExitCode=%d Dauer=%.2fs",
            status.value,
            result.returncode,
            duration,
        )

        if (
            log_full_output
            and result.stdout
        ):

            logger.debug(
                "Prozessausgabe:\n%s",
                result.stdout.rstrip(),
            )

        return ProcessResult(
            return_code=result.returncode,
            output=result.stdout,
            duration_seconds=duration,
            status=status,
        )

    except FileNotFoundError:

        duration = (
            time.perf_counter()
            - start
        )

        message = (
            f"Programm nicht gefunden: "
            f"{command[0]}"
        )

        logger.error(
            message
        )

        return ProcessResult(
            return_code=1,
            output=message,
            duration_seconds=duration,
            status=ProcessStatus.TOOL_ERROR,
        )

    except KeyboardInterrupt:

        duration = (
            time.perf_counter()
            - start
        )

        logger.warning(
            "Prozess durch Benutzer abgebrochen"
        )

        return ProcessResult(
            return_code=130,
            output="Abbruch durch Benutzer.",
            duration_seconds=duration,
            status=ProcessStatus.USER_ABORT,
        )

    except Exception as exc:

        duration = (
            time.perf_counter()
            - start
        )

        logger.exception(
            "Fehler bei Prozessausführung"
        )

        return ProcessResult(
            return_code=1,
            output=str(exc),
            duration_seconds=duration,
            status=ProcessStatus.TOOL_ERROR,
        )


def run_live(
    command: list[str],
    cwd: Optional[Path] = None,
) -> ProcessResult:

    command_text = quote_command(
        command
    )

    logger.info(
        "Starte Prozess: %s",
        command_text,
    )

    print_separator(
        "Auszuführender Befehl"
    )

    print(
        command_text
    )

    print("=" * 80)
    print()

    start = time.perf_counter()

    process: Optional[
        subprocess.Popen
    ] = None

    output_lines: list[str] = []

    try:

        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        if process.stdout:

            for line in process.stdout:

                print(
                    line,
                    end="",
                )

                output_lines.append(
                    line
                )

                # Prozessoutput nicht nochmals
                # als riesige DEBUG-Blöcke schreiben.
                logger.debug(
                    "[TOOL] %s",
                    line.rstrip(),
                )

        return_code = (
            process.wait()
        )

        duration = (
            time.perf_counter()
            - start
        )

        status = (
            classify_process_status(
                return_code
            )
        )

        logger.info(
            "Prozess beendet: "
            "Status=%s ExitCode=%d Dauer=%.2fs",
            status.value,
            return_code,
            duration,
        )

        return ProcessResult(
            return_code=return_code,
            output="".join(
                output_lines
            ),
            duration_seconds=duration,
            status=status,
        )

    except KeyboardInterrupt:

        logger.warning(
            "Prozess durch Benutzer abgebrochen"
        )

        if process:

            try:
                process.terminate()

            except Exception:
                logger.debug(
                    "Prozess konnte "
                    "nicht beendet werden",
                    exc_info=True,
                )

        duration = (
            time.perf_counter()
            - start
        )

        return ProcessResult(
            return_code=130,
            output="".join(
                output_lines
            ),
            duration_seconds=duration,
            status=ProcessStatus.USER_ABORT,
        )

    except Exception as exc:

        duration = (
            time.perf_counter()
            - start
        )

        logger.exception(
            "Fehler beim Starten "
            "des Prozesses"
        )

        return ProcessResult(
            return_code=1,
            output=str(exc),
            duration_seconds=duration,
            status=ProcessStatus.TOOL_ERROR,
        )


# =====================================================================
# Executables
# =====================================================================

def locate_executables(
    bin_dir: Path,
) -> Executables:

    bin_dir = bin_dir.resolve()

    logger.info(
        "Prüfe Build-Verzeichnis: %s",
        bin_dir,
    )

    if not bin_dir.exists():

        raise FileNotFoundError(
            f"Build-Verzeichnis "
            f"existiert nicht:\n"
            f"{bin_dir}"
        )

    required = {
        "cli": "llama-cli.exe",
        "quantize": "llama-quantize.exe",
        "bench": "llama-bench.exe",
    }

    found: dict[
        str,
        Path
    ] = {}

    for key, filename in required.items():

        path = (
            bin_dir
            / filename
        )

        if not path.exists():

            raise FileNotFoundError(
                f"{filename} "
                f"wurde nicht gefunden:\n"
                f"{path}"
            )

        found[key] = path

        logger.debug(
            "Gefunden: %s",
            path,
        )

    def optional(
        name: str,
    ) -> Optional[Path]:

        path = (
            bin_dir
            / name
        )

        return (
            path
            if path.exists()
            else None
        )

    return Executables(
        cli=found["cli"],
        quantize=found["quantize"],
        bench=found["bench"],
        perplexity=optional(
            "llama-perplexity.exe"
        ),
        imatrix=optional(
            "llama-imatrix.exe"
        ),
        server=optional(
            "llama-server.exe"
        ),
    )


# =====================================================================
# Help einmalig lesen
# =====================================================================

def read_help_output(
    exe: Path,
) -> str:

    result = run_capture(
        [
            str(exe),
            "--help",
        ],
        log_full_output=False,
    )

    if not result.output.strip():

        raise RuntimeError(
            f"{exe.name} --help "
            f"lieferte keine Ausgabe."
        )

    output_lower = (
        result.output.lower()
    )

    help_markers = (
        "usage:",
        "--help",
        "common params",
        "allowed quantization types",
    )

    valid_help = any(
        marker in output_lower
        for marker in help_markers
    )

    if not valid_help:

        raise RuntimeError(
            f"{exe.name} lieferte "
            f"keine erkennbare Help-Ausgabe."
        )

    if result.return_code != 0:

        logger.info(
            "%s --help: "
            "Status=%s ExitCode=%d "
            "(gültige Help-Ausgabe)",
            exe.name,
            ProcessStatus.HELP_OK_NONZERO.value,
            result.return_code,
        )

    return result.output


# =====================================================================
# KV Typen
# =====================================================================

def detect_kv_types_from_text(
    text: str,
) -> list[str]:

    lines = (
        text.splitlines()
    )

    for index, line in enumerate(
        lines
    ):

        if "--cache-type-k" not in line:
            continue

        for offset in range(
            index + 1,
            min(
                index + 10,
                len(lines),
            ),
        ):

            current = (
                lines[offset]
                .strip()
            )

            if (
                "allowed values:"
                not in current.lower()
            ):
                continue

            current = re.sub(
                r"^.*?allowed values:\s*",
                "",
                current,
                flags=re.IGNORECASE,
            )

            values = [
                item.strip()
                for item
                in current.split(",")
                if item.strip()
            ]

            # Folgezeilen
            for follow_index in range(
                offset + 1,
                min(
                    offset + 6,
                    len(lines),
                ),
            ):

                follow = (
                    lines[
                        follow_index
                    ]
                    .strip()
                )

                if not follow:
                    break

                if follow.startswith(
                    "(default:"
                ):
                    break

                if follow.startswith(
                    "(env:"
                ):
                    break

                if "--cache-type-v" in follow:
                    break

                if re.fullmatch(
                    r"[A-Za-z0-9_,\s]+",
                    follow,
                ):

                    values.extend(
                        item.strip()
                        for item
                        in follow.split(",")
                        if item.strip()
                    )

                else:
                    break

            # Duplikate entfernen
            values = unique_preserve_order(
                values
            )

            logger.info(
                "KV-Typen erkannt: %s",
                ", ".join(
                    values
                ),
            )

            return values

    logger.warning(
        "Keine KV-Typen erkannt"
    )

    return []


# =====================================================================
# Quant Typen
# =====================================================================

def detect_quant_types_from_text(
    text: str,
) -> list[str]:

    marker = (
        "allowed quantization types"
    )

    marker_index = (
        text.lower()
        .find(marker)
    )

    if marker_index < 0:

        logger.warning(
            "Abschnitt "
            "'allowed quantization types' "
            "nicht gefunden"
        )

        return []

    quant_section = (
        text[
            marker_index:
        ]
    )

    quant_types: list[str] = []

    patterns = [
        re.compile(
            r"^\s*"
            r"(?:\d+\s+or\s+)?"
            r"([A-Z][A-Z0-9_]+)"
            r"\s*:",
            re.MULTILINE,
        ),
        re.compile(
            r"\b([A-Z][A-Z0-9_]{2,})\b"
        ),
    ]

    for pattern in patterns:

        for match in pattern.finditer(
            quant_section
        ):

            quant_type = (
                match.group(1)
            )

            # Defensive Filter
            if quant_type in {
                "WARNING",
                "NOTE",
                "ERROR",
                "USAGE",
                "ALLOWED",
                "QUANTIZATION",
                "TYPES",
            }:
                continue

            if (
                quant_type
                not in quant_types
            ):

                quant_types.append(
                    quant_type
                )

    if (
        "COPY" in quant_section
        and "COPY" not in quant_types
    ):

        quant_types.append(
            "COPY"
        )

    logger.info(
        "%d Weight-Quant-Typen erkannt",
        len(
            quant_types
        ),
    )

    logger.debug(
        "Weight-Quant-Typen: %s",
        ", ".join(
            quant_types
        ),
    )

    return quant_types


# =====================================================================
# Build Fähigkeiten einmalig erkennen
# =====================================================================

def detect_build_capabilities(
    executables: Executables,
) -> BuildCapabilities:

    logger.info(
        "Ermittle Build-Fähigkeiten..."
    )

    cli_help = (
        read_help_output(
            executables.cli
        )
    )

    quantize_help = (
        read_help_output(
            executables.quantize
        )
    )

    kv_types = (
        detect_kv_types_from_text(
            cli_help
        )
    )

    quant_types = (
        detect_quant_types_from_text(
            quantize_help
        )
    )

    logger.info(
        "Build-Fähigkeiten "
        "erfolgreich ermittelt"
    )

    return BuildCapabilities(
        kv_types=kv_types,
        quant_types=quant_types,
        cli_help=cli_help,
        quantize_help=quantize_help,
    )


# =====================================================================
# Modelle
# =====================================================================

def find_models(
    path: Path,
) -> list[Path]:

    path = (
        path.resolve()
    )

    logger.info(
        "Suche GGUF-Modelle unter: %s",
        path,
    )

    if not path.exists():

        logger.error(
            "Modellpfad existiert nicht: %s",
            path,
        )

        return []

    # direkte Datei
    if path.is_file():

        if (
            path.suffix.lower()
            == ".gguf"
        ):

            logger.info(
                "Einzelnes GGUF-Modell: %s",
                path,
            )

            return [
                path
            ]

        logger.error(
            "Keine GGUF-Datei: %s",
            path,
        )

        return []

    models = sorted(
        path.rglob(
            "*.gguf"
        ),
        key=lambda item:
            item.name.lower(),
    )

    logger.info(
        "%d GGUF-Modelle gefunden",
        len(
            models
        ),
    )

    return models


def print_models(
    models: list[Path],
) -> None:

    print_separator(
        "Gefundene GGUF-Modelle"
    )

    if not models:

        print(
            "Keine Modelle gefunden."
        )

        return

    for index, model in enumerate(
        models,
        start=1,
    ):

        try:

            size = human_size(
                model.stat().st_size
            )

        except OSError:

            size = "?"

        print(
            f"[{index:3}] "
            f"{size:>12}  "
            f"{model.name}"
        )


def validate_model_file(
    model: Path,
) -> Path:

    model = model.resolve()

    if not model.exists():
        raise FileNotFoundError(
            f"Modell existiert nicht:\n{model}"
        )

    if not model.is_file():
        raise ValueError(
            f"Modellpfad ist keine Datei:\n{model}"
        )

    if model.suffix.lower() != ".gguf":
        raise ValueError(
            f"Modell ist keine GGUF-Datei:\n{model}"
        )

    return model


def validate_input_file(
    path: Optional[Path],
    label: str,
) -> Optional[Path]:

    if path is None:
        return None

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"{label} existiert nicht:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{label} ist keine Datei:\n{path}"
        )

    return path


def prepare_output_file(
    path: Path,
    overwrite: bool = False,
) -> Path:

    path = path.resolve()
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"Zieldatei existiert bereits:\n{path}\n"
            "Bitte anderen Namen wählen oder Datei bewusst löschen."
        )

    return path


def prompt_positive_int(
    prompt: str,
    default: int,
) -> Optional[int]:

    answer = input(
        prompt
    ).strip()

    if not answer:
        return default

    try:
        return positive_int(
            answer
        )

    except (
        ValueError,
        argparse.ArgumentTypeError,
    ):
        print(
            "Ungültiger Wert. Bitte eine Zahl größer 0 eingeben."
        )

        return None


def select_model(
    path: Path,
) -> Optional[Path]:

    models = (
        find_models(
            path
        )
    )

    if not models:

        return None

    if (
        len(models) == 1
        and path.resolve().is_file()
    ):

        logger.info(
            "Direktmodell verwendet: %s",
            models[0],
        )

        return models[0]

    print_models(
        models
    )

    while True:

        answer = input(
            "\nModellnummer auswählen "
            "[Enter = Abbruch]: "
        ).strip()

        if not answer:

            logger.info(
                "Modellauswahl abgebrochen"
            )

            return None

        try:

            number = int(
                answer
            )

            if (
                1
                <= number
                <= len(models)
            ):

                selected = (
                    models[
                        number - 1
                    ]
                )

                logger.info(
                    "Modell ausgewählt: %s",
                    selected,
                )

                return selected

        except ValueError:

            pass

        print(
            "Ungültige Auswahl."
        )


# =====================================================================
# CLI
# =====================================================================

def build_cli_command(
    exe: Path,
    model: Path,
    kv_type: str,
    context: int,
    gpu_layers: str,
    flash_attn: str,
    fit: bool,
    device: Optional[str],
    temperature: Optional[float],
) -> list[str]:

    command = [
        str(exe),

        "-m",
        str(
            model.resolve()
        ),

        "-c",
        str(
            context
        ),

        "-ngl",
        gpu_layers,

        "-fa",
        flash_attn,

        "-ctk",
        kv_type,

        "-ctv",
        kv_type,
    ]

    if fit:

        command.extend(
            [
                "--fit",
                "on",
            ]
        )

    if device:

        command.extend(
            [
                "--device",
                device,
            ]
        )

    if temperature is not None:

        command.extend(
            [
                "--temp",
                str(
                    temperature
                ),
            ]
        )

    return command


def start_model(
    executables: Executables,
    model: Path,
    kv_type: str,
    context: int,
    gpu_layers: str,
    flash_attn: str,
    fit: bool,
    device: Optional[str],
    temperature: Optional[float],
) -> ProcessResult:

    model = validate_model_file(
        model
    )

    logger.info(
        "Starte Modell: "
        "%s | KV=%s | Context=%d",
        model,
        kv_type,
        context,
    )

    command = (
        build_cli_command(
            executables.cli,
            model,
            kv_type,
            context,
            gpu_layers,
            flash_attn,
            fit,
            device,
            temperature,
        )
    )

    return run_live(
        command
    )


# =====================================================================
# Benchmark
# =====================================================================

def build_bench_command(
    bench_exe: Path,
    model: Path,
    kv_type: str,
    context: int,
    gpu_layers: str,
    flash_attn: str,
    repetitions: int,
) -> list[str]:

    return [
        str(
            bench_exe
        ),

        "-m",
        str(
            model.resolve()
        ),

        "-c",
        str(
            context
        ),

        "-ngl",
        gpu_layers,

        "-fa",
        bench_flash_attn_value(
            flash_attn
        ),

        "-ctk",
        kv_type,

        "-ctv",
        kv_type,

        "-r",
        str(
            repetitions
        ),
    ]


def extract_tokens_per_second(
    output: str,
) -> Optional[float]:

    patterns = [
        r"([\d.,]+)\s*"
        r"(?:±\s*[\d.]+)?"
        r"\s*t/s",

        r"([\d.,]+)\s*"
        r"tokens/s",

        r"([\d.,]+)\s*"
        r"tokens\s+per\s+second",

        r"([\d.,]+)\s*"
        r"tok/s",
    ]

    values: list[
        float
    ] = []

    for pattern in patterns:

        for value in re.findall(
            pattern,
            output,
            re.IGNORECASE,
        ):

            try:
                values.append(
                    float(
                        value.replace(
                            ",",
                            ".",
                        )
                    )
                )
            except ValueError:
                pass

    if not values:
        return None

    return values[-1]


def benchmark_one(
    executables: Executables,
    model: Path,
    kv_type: str,
    context: int,
    gpu_layers: str,
    flash_attn: str,
    repetitions: int,
) -> BenchResult:

    model = validate_model_file(
        model
    )

    logger.info(
        "Benchmark gestartet: "
        "KV=%s Context=%d Wiederholungen=%d",
        kv_type,
        context,
        repetitions,
    )

    command = (
        build_bench_command(
            executables.bench,
            model,
            kv_type,
            context,
            gpu_layers,
            flash_attn,
            repetitions,
        )
    )

    print_separator(
        f"Benchmark: {kv_type}"
    )

    print(
        quote_command(
            command
        )
    )

    result = run_capture(
        command,
        log_full_output=False,
    )

    print(
        result.output
    )

    tps = (
        extract_tokens_per_second(
            result.output
        )
    )

    logger.info(
        "Benchmark Ergebnis: "
        "KV=%s Status=%s "
        "Exit=%d Dauer=%.2fs TPS=%s",
        kv_type,
        result.status.value,
        result.return_code,
        result.duration_seconds,
        (
            f"{tps:.2f}"
            if tps is not None
            else "?"
        ),
    )

    return BenchResult(
        model=str(
            model
        ),
        kv_type=kv_type,
        context=context,
        return_code=(
            result.return_code
        ),
        duration_seconds=(
            result.duration_seconds
        ),
        raw_output=(
            result.output
        ),
        tokens_per_second=tps,
    )


def benchmark_all(
    executables: Executables,
    model: Path,
    available_kv_types: list[str],
    requested_types: list[str],
    context: int,
    gpu_layers: str,
    flash_attn: str,
    repetitions: int,
) -> list[BenchResult]:

    results: list[
        BenchResult
    ] = []

    for kv_type in requested_types:

        if (
            available_kv_types
            and kv_type
            not in available_kv_types
        ):

            normalized_kv_type = normalize_choice(
                kv_type,
                available_kv_types,
            )

            if normalized_kv_type is None:

                logger.warning(
                    "KV-Typ '%s' "
                    "wird nicht unterstützt",
                    kv_type,
                )

                continue

            kv_type = normalized_kv_type

        result = benchmark_one(
            executables,
            model,
            kv_type,
            context,
            gpu_layers,
            flash_attn,
            repetitions,
        )

        results.append(
            result
        )

    return results


def print_benchmark_summary(
    results: list[BenchResult],
) -> None:

    print_separator(
        "TurboQuant Benchmark-Zusammenfassung"
    )

    print(
        f"{'KV-Typ':<12}"
        f"{'Exit':>8}"
        f"{'Laufzeit':>14}"
        f"{'Token/s':>14}"
    )

    print(
        "-" * 48
    )

    for result in results:

        tps = (
            f"{result.tokens_per_second:.2f}"
            if result.tokens_per_second
            is not None
            else "?"
        )

        print(
            f"{result.kv_type:<12}"
            f"{result.return_code:>8}"
            f"{result.duration_seconds:>12.2f}s"
            f"{tps:>14}"
        )


def save_benchmark_csv(
    results: list[BenchResult],
    output_file: Path,
) -> None:

    output_file = (
        output_file.resolve()
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "model",
                "kv_type",
                "context",
                "return_code",
                "duration_seconds",
                "tokens_per_second",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result.model,
                    result.kv_type,
                    result.context,
                    result.return_code,
                    f"{result.duration_seconds:.4f}",
                    result.tokens_per_second,
                ]
            )

    logger.info(
        "Benchmark CSV gespeichert: %s",
        output_file,
    )


# =====================================================================
# Quantisierung
# =====================================================================

def quantize_dry_run(
    executables: Executables,
    source: Path,
    quant_type: str,
    imatrix: Optional[Path] = None,
) -> ProcessResult:

    source = validate_model_file(
        source
    )

    imatrix = validate_input_file(
        imatrix,
        "iMatrix-Datei",
    )

    command = [
        str(
            executables.quantize
        ),
        "--dry-run",
    ]

    if imatrix:

        command.extend(
            [
                "--imatrix",
                str(
                    imatrix.resolve()
                ),
            ]
        )

    command.extend(
        [
            str(
                source.resolve()
            ),
            quant_type,
        ]
    )

    return run_live(
        command
    )


def quantize_model(
    executables: Executables,
    source: Path,
    destination: Path,
    quant_type: str,
    threads: int,
    imatrix: Optional[Path] = None,
    allow_requantize: bool = False,
    leave_output_tensor: bool = False,
    pure: bool = False,
) -> ProcessResult:

    source = validate_model_file(
        source
    )

    destination = prepare_output_file(
        destination
    )

    imatrix = validate_input_file(
        imatrix,
        "iMatrix-Datei",
    )

    command = [
        str(
            executables.quantize
        )
    ]

    if allow_requantize:

        command.append(
            "--allow-requantize"
        )

    if leave_output_tensor:

        command.append(
            "--leave-output-tensor"
        )

    if pure:

        command.append(
            "--pure"
        )

    if imatrix:

        command.extend(
            [
                "--imatrix",
                str(
                    imatrix.resolve()
                ),
            ]
        )

    command.extend(
        [
            str(
                source.resolve()
            ),

            str(
                destination.resolve()
            ),

            quant_type,

            str(
                threads
            ),
        ]
    )

    result = run_live(
        command
    )

    if (
        result.status
        == ProcessStatus.EXIT_OK
        and destination.exists()
    ):

        logger.info(
            "Quantisierung erfolgreich: "
            "%s (%s)",
            destination,
            human_size(
                destination.stat().st_size
            ),
        )

    return result


# =====================================================================
# Build/GPU
# =====================================================================

def show_build_info(
    executables: Executables,
) -> None:

    print_separator(
        "TurboQuant / llama.cpp Build"
    )

    result = run_capture(
        [
            str(
                executables.cli
            ),
            "--version",
        ]
    )

    print(
        result.output
    )

    print_separator(
        "GPU / Vulkan Geräte"
    )

    result = run_capture(
        [
            str(
                executables.cli
            ),
            "--list-devices",
        ]
    )

    print(
        result.output
    )


# =====================================================================
# Auswahl
# =====================================================================

def select_kv_type(
    kv_types: list[str],
) -> Optional[str]:

    print_separator(
        "Verfügbare KV-Cache-Typen"
    )

    for index, value in enumerate(
        kv_types,
        start=1,
    ):

        marker = (
            "  <- empfohlen"
            if value == "turbo3"
            else ""
        )

        print(
            f"[{index:3}] "
            f"{value}"
            f"{marker}"
        )

    default = (
        "turbo3"
        if "turbo3" in kv_types
        else (
            kv_types[0]
            if kv_types
            else None
        )
    )

    if default is None:
        return None

    answer = input(
        f"\nKV-Typ "
        f"[Default: {default}]: "
    ).strip()

    if not answer:
        return default

    try:

        number = int(
            answer
        )

        if (
            1
            <= number
            <= len(kv_types)
        ):

            return kv_types[
                number - 1
            ]

    except ValueError:
        selected = normalize_choice(
            answer,
            kv_types,
        )

        if selected:
            return selected

    return None


def select_quant_type(
    quant_types: list[str],
) -> Optional[str]:

    print_separator(
        "Weight-Quantisierung"
    )

    for index, value in enumerate(
        quant_types,
        start=1,
    ):

        marker = ""

        if value in {
            "TQ3_1S",
            "TQ4_1S",
        }:

            marker = (
                "  <- TurboQuant"
            )

        print(
            f"[{index:3}] "
            f"{value}"
            f"{marker}"
        )

    answer = input(
        "\nQuant-Typ "
        "[Enter = Abbruch]: "
    ).strip()

    if not answer:
        return None

    try:

        number = int(
            answer
        )

        if (
            1
            <= number
            <= len(quant_types)
        ):

            return quant_types[
                number - 1
            ]

    except ValueError:
        selected = normalize_choice(
            answer,
            quant_types,
        )

        if selected:
            return selected

    return None


# =====================================================================
# Interaktiv
# =====================================================================

def interactive_mode(
    executables: Executables,
    capabilities: BuildCapabilities,
    model_path: Path,
) -> None:

    kv_types = (
        capabilities.kv_types
    )

    quant_types = (
        capabilities.quant_types
    )

    while True:

        print_program_header()

        print()
        print(
            "KV-Typen:"
        )
        print(
            "  "
            + ", ".join(
                kv_types
            )
        )

        print()

        print(
            f"Weight-Quant-Typen: "
            f"{len(quant_types)}"
        )

        print()

        print(
            "[1] Modelle anzeigen"
        )

        print(
            "[2] Modell starten"
        )

        print(
            "[3] TurboQuant KV Benchmark"
        )

        print(
            "[4] Modell quantisieren"
        )

        print(
            "[5] Quantisierung Dry-Run"
        )

        print(
            "[6] Build / GPU Informationen"
        )

        print(
            "[7] Unterstützte Quant-Typen"
        )

        print(
            "[8] Logdatei anzeigen"
        )

        print(
            "[0] Beenden"
        )

        selection = input(
            "\nAuswahl: "
        ).strip()

        # ---------------------------------------------------------
        # Modelle
        # ---------------------------------------------------------

        if selection == "1":

            print_models(
                find_models(
                    model_path
                )
            )

        # ---------------------------------------------------------
        # Modell starten
        # ---------------------------------------------------------

        elif selection == "2":

            model = select_model(
                model_path
            )

            if not model:
                continue

            kv_type = select_kv_type(
                kv_types
            )

            if not kv_type:
                continue

            context = prompt_positive_int(
                f"Context "
                f"[{DEFAULT_CONTEXT}]: ",
                DEFAULT_CONTEXT,
            )

            if context is None:
                continue

            result = start_model(
                executables=executables,
                model=model,
                kv_type=kv_type,
                context=context,
                gpu_layers=(
                    DEFAULT_GPU_LAYERS
                ),
                flash_attn=(
                    DEFAULT_FLASH_ATTN
                ),
                fit=True,
                device=None,
                temperature=None,
            )

            if (
                result.status
                == ProcessStatus.USER_ABORT
            ):

                print(
                    "\nModell wurde "
                    "vom Benutzer beendet."
                )

            elif (
                result.status
                == ProcessStatus.TOOL_ERROR
            ):

                print(
                    "\nllama-cli meldete "
                    f"einen Fehler "
                    f"(ExitCode "
                    f"{result.return_code})."
                )

        # ---------------------------------------------------------
        # Benchmark
        # ---------------------------------------------------------

        elif selection == "3":

            model = select_model(
                model_path
            )

            if not model:
                continue

            context = prompt_positive_int(
                f"Context "
                f"[{DEFAULT_CONTEXT}]: ",
                DEFAULT_CONTEXT,
            )

            if context is None:
                continue

            repetitions = prompt_positive_int(
                f"Wiederholungen "
                f"[{DEFAULT_REPETITIONS}]: ",
                DEFAULT_REPETITIONS,
            )

            if repetitions is None:
                continue

            results = benchmark_all(
                executables=executables,
                model=model,
                available_kv_types=(
                    kv_types
                ),
                requested_types=(
                    DEFAULT_KV_TESTS
                ),
                context=context,
                gpu_layers=(
                    DEFAULT_GPU_LAYERS
                ),
                flash_attn=(
                    DEFAULT_FLASH_ATTN
                ),
                repetitions=repetitions,
            )

            print_benchmark_summary(
                results
            )

            timestamp = (
                datetime.now()
                .strftime(
                    "%Y-%m-%d_%H-%M-%S"
                )
            )

            save_benchmark_csv(
                results,
                (
                    Path(
                        "benchmark-results"
                    )
                    / (
                        f"turboquant_"
                        f"{timestamp}.csv"
                    )
                ),
            )

        # ---------------------------------------------------------
        # Quantisierung
        # ---------------------------------------------------------

        elif selection == "4":

            model = select_model(
                model_path
            )

            if not model:
                continue

            quant_type = (
                select_quant_type(
                    quant_types
                )
            )

            if not quant_type:
                continue

            destination = (
                model.parent
                / (
                    f"{model.stem}"
                    f"-{quant_type}"
                    f".gguf"
                )
            )

            print_separator(
                "Quantisierung"
            )

            print(
                f"Quelle: {model}"
            )

            print(
                f"Ziel:   {destination}"
            )

            print(
                f"Typ:    {quant_type}"
            )

            dry_result = (
                quantize_dry_run(
                    executables,
                    model,
                    quant_type,
                )
            )

            if (
                dry_result.status
                != ProcessStatus.EXIT_OK
            ):

                print(
                    "\nDry-Run fehlgeschlagen."
                )

                continue

            answer = input(
                "\nQuantisierung starten? "
                "[j/N]: "
            ).strip().lower()

            if answer not in {
                "j",
                "ja",
                "y",
                "yes",
            }:

                continue

            quantize_model(
                executables=executables,
                source=model,
                destination=destination,
                quant_type=quant_type,
                threads=(
                    os.cpu_count()
                    or 1
                ),
            )

        # ---------------------------------------------------------
        # Dry Run
        # ---------------------------------------------------------

        elif selection == "5":

            model = select_model(
                model_path
            )

            if not model:
                continue

            quant_type = (
                select_quant_type(
                    quant_types
                )
            )

            if not quant_type:
                continue

            quantize_dry_run(
                executables,
                model,
                quant_type,
            )

        # ---------------------------------------------------------
        # Info
        # ---------------------------------------------------------

        elif selection == "6":

            show_build_info(
                executables
            )

        # ---------------------------------------------------------
        # Typen
        # ---------------------------------------------------------

        elif selection == "7":

            print_separator(
                "KV-Typen"
            )

            for value in kv_types:

                print(
                    f"  {value}"
                )

            print_separator(
                "Weight-Quant-Typen"
            )

            for value in quant_types:

                print(
                    f"  {value}"
                )

        # ---------------------------------------------------------
        # Log
        # ---------------------------------------------------------

        elif selection == "8":

            for handler in (
                logger.handlers
            ):

                if isinstance(
                    handler,
                    logging.FileHandler,
                ):

                    print(
                        "\nLogdatei:"
                    )

                    print(
                        handler.baseFilename
                    )

        # ---------------------------------------------------------
        # Exit
        # ---------------------------------------------------------

        elif selection == "0":

            logger.info(
                "Benutzer wählte Beenden"
            )

            return


# =====================================================================
# Argument Parser
# =====================================================================

def create_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            f"{PROGRAM_NAME} "
            f"- erstellt von "
            f"{PROGRAM_AUTHOR}"
        )
    )

    parser.add_argument(
        "--bin-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(
            "logs"
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ],
        default=(
            DEFAULT_LOG_LEVEL
        ),
    )

    subparsers = (
        parser.add_subparsers(
            dest="command"
        )
    )

    interactive = (
        subparsers.add_parser(
            "interactive"
        )
    )

    interactive.add_argument(
        "--models",
        type=Path,
        required=True,
    )

    run = (
        subparsers.add_parser(
            "run"
        )
    )

    run.add_argument(
        "model",
        type=Path,
    )

    run.add_argument(
        "--kv",
        default="turbo3",
    )

    run.add_argument(
        "--ctx",
        type=positive_int,
        default=(
            DEFAULT_CONTEXT
        ),
    )

    run.add_argument(
        "--gpu-layers",
        default=(
            DEFAULT_GPU_LAYERS
        ),
    )

    run.add_argument(
        "--flash-attn",
        choices=[
            "on",
            "off",
            "auto",
        ],
        default=(
            DEFAULT_FLASH_ATTN
        ),
    )

    run.add_argument(
        "--device",
        default=None,
    )

    run.add_argument(
        "--temperature",
        type=float,
        default=None,
    )

    bench = (
        subparsers.add_parser(
            "benchmark"
        )
    )

    bench.add_argument(
        "model",
        type=Path,
    )

    bench.add_argument(
        "--ctx",
        type=positive_int,
        default=(
            DEFAULT_CONTEXT
        ),
    )

    bench.add_argument(
        "--kv",
        nargs="+",
        default=(
            DEFAULT_KV_TESTS
        ),
    )

    bench.add_argument(
        "--repetitions",
        type=positive_int,
        default=(
            DEFAULT_REPETITIONS
        ),
    )

    bench.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "turboquant_results.csv"
        ),
    )

    quant = (
        subparsers.add_parser(
            "quantize"
        )
    )

    quant.add_argument(
        "input",
        type=Path,
    )

    quant.add_argument(
        "output",
        type=Path,
    )

    quant.add_argument(
        "type",
    )

    quant.add_argument(
        "--threads",
        type=positive_int,
        default=(
            os.cpu_count()
            or 1
        ),
    )

    quant.add_argument(
        "--imatrix",
        type=Path,
        default=None,
    )

    quant.add_argument(
        "--allow-requantize",
        action="store_true",
    )

    quant.add_argument(
        "--leave-output-tensor",
        action="store_true",
    )

    quant.add_argument(
        "--pure",
        action="store_true",
    )

    dry = (
        subparsers.add_parser(
            "dry-run"
        )
    )

    dry.add_argument(
        "input",
        type=Path,
    )

    dry.add_argument(
        "type",
    )

    dry.add_argument(
        "--imatrix",
        type=Path,
        default=None,
    )

    subparsers.add_parser(
        "info"
    )

    return parser


# =====================================================================
# Main
# =====================================================================

def main() -> int:

    parser = (
        create_parser()
    )

    args = (
        parser.parse_args()
    )

    log_file = (
        setup_logging(
            args.log_dir,
            args.log_level,
        )
    )

    print_program_header()

    try:

        executables = (
            locate_executables(
                args.bin_dir
            )
        )

        # ---------------------------------------------------------
        # EINMALIGE Erkennung
        # ---------------------------------------------------------

        capabilities = (
            detect_build_capabilities(
                executables
            )
        )

        logger.info(
            "TurboQuant Build "
            "erfolgreich erkannt"
        )

        # ---------------------------------------------------------
        # Interactive
        # ---------------------------------------------------------

        if (
            args.command
            == "interactive"
        ):

            interactive_mode(
                executables,
                capabilities,
                args.models.resolve(),
            )

            return 0

        # ---------------------------------------------------------
        # Run
        # ---------------------------------------------------------

        if args.command == "run":

            if (
                capabilities.kv_types
                and args.kv
                not in capabilities.kv_types
            ):

                normalized_kv = normalize_choice(
                    args.kv,
                    capabilities.kv_types,
                )

                if normalized_kv is None:

                    logger.error(
                        "Unbekannter KV-Typ: %s",
                        args.kv,
                    )

                    return 1

                args.kv = normalized_kv

            result = start_model(
                executables=executables,
                model=args.model,
                kv_type=args.kv,
                context=args.ctx,
                gpu_layers=(
                    args.gpu_layers
                ),
                flash_attn=(
                    args.flash_attn
                ),
                fit=True,
                device=args.device,
                temperature=(
                    args.temperature
                ),
            )

            return (
                0
                if (
                    result.status
                    in {
                        ProcessStatus.EXIT_OK,
                        ProcessStatus.USER_ABORT,
                    }
                )
                else result.return_code
            )

        # ---------------------------------------------------------
        # Benchmark
        # ---------------------------------------------------------

        if (
            args.command
            == "benchmark"
        ):

            results = (
                benchmark_all(
                    executables=executables,
                    model=args.model,
                    available_kv_types=(
                        capabilities.kv_types
                    ),
                    requested_types=(
                        args.kv
                    ),
                    context=args.ctx,
                    gpu_layers=(
                        DEFAULT_GPU_LAYERS
                    ),
                    flash_attn=(
                        DEFAULT_FLASH_ATTN
                    ),
                    repetitions=(
                        args.repetitions
                    ),
                )
            )

            print_benchmark_summary(
                results
            )

            save_benchmark_csv(
                results,
                args.csv,
            )

            return 0

        # ---------------------------------------------------------
        # Quantize
        # ---------------------------------------------------------

        if (
            args.command
            == "quantize"
        ):

            if (
                capabilities.quant_types
                and args.type
                not in capabilities.quant_types
            ):

                normalized_type = normalize_choice(
                    args.type,
                    capabilities.quant_types,
                )

                if normalized_type is None:

                    logger.error(
                        "Unbekannter Quant-Typ: %s",
                        args.type,
                    )

                    return 1

                args.type = normalized_type

            result = quantize_model(
                executables=executables,
                source=args.input,
                destination=args.output,
                quant_type=args.type,
                threads=args.threads,
                imatrix=args.imatrix,
                allow_requantize=(
                    args.allow_requantize
                ),
                leave_output_tensor=(
                    args.leave_output_tensor
                ),
                pure=args.pure,
            )

            return result.return_code

        # ---------------------------------------------------------
        # Dry Run
        # ---------------------------------------------------------

        if (
            args.command
            == "dry-run"
        ):

            if (
                capabilities.quant_types
                and args.type
                not in capabilities.quant_types
            ):

                normalized_type = normalize_choice(
                    args.type,
                    capabilities.quant_types,
                )

                if normalized_type is None:

                    logger.error(
                        "Unbekannter Quant-Typ: %s",
                        args.type,
                    )

                    return 1

                args.type = normalized_type

            result = (
                quantize_dry_run(
                    executables,
                    args.input,
                    args.type,
                    args.imatrix,
                )
            )

            return (
                result.return_code
            )

        # ---------------------------------------------------------
        # Info
        # ---------------------------------------------------------

        if (
            args.command
            == "info"
        ):

            show_build_info(
                executables
            )

            print_separator(
                "KV-Typen"
            )

            for value in (
                capabilities.kv_types
            ):

                print(
                    f"  {value}"
                )

            print_separator(
                "Weight-Quant-Typen"
            )

            for value in (
                capabilities.quant_types
            ):

                print(
                    f"  {value}"
                )

            return 0

        parser.print_help()

        return 0

    except KeyboardInterrupt:

        logger.warning(
            "Programm durch "
            "Benutzer beendet"
        )

        return 130

    except Exception:

        logger.exception(
            "Unbehandelter "
            "Programmfehler"
        )

        print(
            "\nFehler. Details:"
        )

        print(
            log_file
        )

        return 1

    finally:

        logger.info(
            "%s beendet",
            PROGRAM_NAME,
        )


# =====================================================================
# Entry Point
# =====================================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )
