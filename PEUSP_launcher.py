from __future__ import annotations

import os
import sys
import shutil
import tempfile
import traceback
import webbrowser
import threading
from pathlib import Path


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def _exe_dir() -> Path:
    # Folder where the .exe sits (or this .py when running from source)
    return Path(sys.executable).resolve().parent if _is_frozen() else Path(__file__).resolve().parent


def _bundle_dir() -> Path:
    # PyInstaller temp extraction dir (or this .py folder when running from source)
    return Path(getattr(sys, "_MEIPASS")).resolve() if _is_frozen() else Path(__file__).resolve().parent


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_log(exe_dir: Path, msg: str) -> None:
    try:
        log_path = exe_dir / "PEUSP_launcher.log"
        log_path.write_text(msg, encoding="utf-8", errors="ignore")
    except Exception:
        # Last resort: swallow logging errors
        pass


def main() -> int:
    exe_dir = _exe_dir()
    bundle_dir = _bundle_dir()

    # Log everything, because the packaged exe is built with console=False
    try:
        work_dir = Path(tempfile.mkdtemp(prefix="PEUSP_beta_")).resolve()

        # ---- Copy bundled code to a writable temp folder
        bundled_app = bundle_dir / "app.py"
        if not bundled_app.exists():
            raise FileNotFoundError(f"Bundled app.py not found at: {bundled_app}")
        shutil.copy2(bundled_app, work_dir / "app.py")

        # Optional: if you later decide to bundle the runners too, they will be available in bundle_dir
        # but app.py imports them, so PyInstaller must include them in hiddenimports (spec file).

        # ---- Copy user-editable data/assets from exe folder into the temp working folder
        # (so the app reads them regardless of where Streamlit runs from)
        _copy_if_exists(exe_dir / "metadata.csv", work_dir / "metadata.csv")
        _copy_if_exists(exe_dir / "PEUSP.png", work_dir / "PEUSP.png")
        _copy_if_exists(exe_dir / "1D.png", work_dir / "1D.png")
        _copy_if_exists(exe_dir / "2D.png", work_dir / "2D.png")

        # ---- Create an isolated Streamlit config so we DO NOT depend on the user's ~/.streamlit/config.toml
        # This avoids the "server.port does not work when global.developmentMode is true" error on other PCs.
        cfg_dir = work_dir / ".streamlit"
        cfg_dir.mkdir(parents=True, exist_ok=True)

        port = int(os.environ.get("PEUSP_PORT", "8502"))
        address = os.environ.get("PEUSP_ADDRESS", "127.0.0.1")
        headless = os.environ.get("PEUSP_HEADLESS", "true").lower() in ("1", "true", "yes", "y")

        (cfg_dir / "config.toml").write_text(
            f"""
[global]
developmentMode = false

[server]
headless = {str(headless).lower()}
address = "{address}"
port = {port}

[browser]
gatherUsageStats = false
""".lstrip(),
            encoding="utf-8",
        )

        # Force Streamlit to use OUR config dir (ignores the user's home config)
        os.environ["STREAMLIT_CONFIG_DIR"] = str(cfg_dir)

        # Streamlit uses cwd for relative paths in your code; make it the temp work dir
        os.chdir(work_dir)

        # Open the browser shortly after launch (Streamlit won't auto-open in headless mode)
        url = f"http://{address}:{port}"
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

        from streamlit.web import cli as stcli  # type: ignore

        # Minimal argv; config.toml controls address/port/headless/devMode robustly
        sys.argv = [
            "streamlit",
            "run",
            str(work_dir / "app.py"),
        ]
        return stcli.main()

    except Exception:
        _write_log(exe_dir, traceback.format_exc())
        # If something fails, at least show the log in Notepad
        try:
            webbrowser.open(str((exe_dir / "PEUSP_launcher.log").resolve()))
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
