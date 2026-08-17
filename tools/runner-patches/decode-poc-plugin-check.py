"""Post-install check for the decode-PoC plugin swap (decode-poc-plugin fragment).

Fails the image build loudly if the decode branch did not actually land —
e.g. pip resolved the pre-installed base package, or the tarball pin points at
a commit that predates the decode port.
"""
import importlib
import sys

MODS = [
    "gonka_poc",
    "gonka_poc.poc.decode_runner",      # decode loop — decode branch only
    "gonka_poc.poc.decode_chain",       # mismatch rule + prev_k chaining
    "gonka_poc.worker.extension",       # collective_rpc entry
    "gonka_poc.entrypoint.api_router",  # composed server entrypoint
    "gonka_poc.models.minimax_m2_poc",  # ModelRegistry subclass
]


def main() -> int:
    for mod in MODS:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001 — any import failure fails the build
            sys.stderr.write(
                f"decode-poc-plugin: cannot import {mod}: {exc!r} — the installed "
                "gonka-poc is not the decode branch\n")
            return 1

    ext = importlib.import_module("gonka_poc.worker.extension")
    if not hasattr(ext.PoCWorkerExtension, "execute_poc_decode"):
        sys.stderr.write(
            "decode-poc-plugin: PoCWorkerExtension.execute_poc_decode missing — "
            "the installed package is not the decode branch\n")
        return 1

    # The decode branch intentionally DROPPED the prefill scheme; if the old
    # prefill runner is still importable, the swap did not happen.
    try:
        importlib.import_module("gonka_poc.poc.validation")
    except ImportError:
        pass  # expected on the decode branch
    else:
        sys.stderr.write(
            "decode-poc-plugin: legacy prefill module gonka_poc.poc.validation "
            "still present — pip kept the base package, swap failed\n")
        return 1

    print("decode-poc-plugin: decode branch installed, all entry points import")
    return 0


if __name__ == "__main__":
    sys.exit(main())
