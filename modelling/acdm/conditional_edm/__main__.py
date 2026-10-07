"""Entry points for the shared-POD training and evaluation workflow."""
import sys


def main():
    commands = ("train", "evaluate", "rollout", "sanity")
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("python -m modelling.acdm.conditional_edm {train,evaluate,rollout,sanity} --help")
        return
    command, arguments = sys.argv[1], sys.argv[2:]
    if command not in commands:
        raise SystemExit(f"Unknown command {command!r}; choose from {commands}.")
    if command == "train":
        from .train import main as run
        run(arguments)
    elif command in ("evaluate", "rollout"):
        from .evaluate import main as run
        run(arguments, metrics=command == "evaluate")
    else:
        from .sanity import main as run
        run(arguments)


if __name__ == "__main__":
    main()
