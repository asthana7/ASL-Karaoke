from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent / "src"/ "templates"

def main():
    if not TEMPLATE_DIR.exists():
        print("Template dir does not exist:", TEMPLATE_DIR)
        return

    deleted = 0
    for p in TEMPLATE_DIR.glob("*.npy"):
        print("Deleting", p.name)
        p.unlink()
        deleted += 1

    print(f"Done. Deleted {deleted} template file(s).")

if __name__ == "__main__":
    main()
