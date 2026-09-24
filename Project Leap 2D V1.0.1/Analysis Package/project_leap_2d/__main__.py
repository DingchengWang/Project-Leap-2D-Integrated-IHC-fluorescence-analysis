from __future__ import annotations

import sys

from .workspace_launcher import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nANALYSIS CANCELLED\n"
            "The source images were retained and no cleanup was performed.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(
            "\nANALYSIS STOPPED\n"
            f"{type(exc).__name__}: {exc}\n"
            "The source images were retained.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
