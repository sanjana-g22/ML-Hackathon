r"""API configuration, read from environment variables.

EVAL_DIR  folder whose sub-folders are trip folders (<id>/<id>.track, ...).
          Development default: evaluator_test if available, otherwise student_test.
          Demo day: dynamically recognizes all ten evaluator trips or student_test,
          or can be overridden via EVAL_DIR, e.g.:
              export EVAL_DIR=data/evaluator_test
              export EVAL_DIR=student_test
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_eval_dir() -> str:
    """Dynamically resolves the evaluation folder."""
    env_dir = os.environ.get("EVAL_DIR")
    if env_dir:
        if os.path.isabs(env_dir) and os.path.isdir(env_dir):
            return env_dir
        rel = os.path.join(REPO_ROOT, env_dir)
        if os.path.isdir(rel):
            return os.path.abspath(rel)
        if os.path.isdir(env_dir):
            return os.path.abspath(env_dir)

    for candidate in [
        os.path.join(REPO_ROOT, "data", "evaluator_test"),
        os.path.join(REPO_ROOT, "evaluator_test"),
        os.path.join(REPO_ROOT, "student_test"),
        os.path.join(REPO_ROOT, "data", "student_test"),
    ]:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)

    return os.path.abspath(os.path.join(REPO_ROOT, "student_test"))


EVAL_DIR = resolve_eval_dir()
