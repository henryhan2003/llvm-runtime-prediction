from pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(["--no-run", *(__import__("sys").argv[1:])]))
