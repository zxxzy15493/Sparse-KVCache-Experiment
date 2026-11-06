"""VRAM-only entry point: peak GPU allocation and reservation."""

from pred import main


if __name__ == "__main__":
    main(measurement="vram")
