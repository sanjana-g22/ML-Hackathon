r"""API configuration, read from environment variables.

EVAL_DIR  folder whose sub-folders are trip folders (<id>/<id>.track, ...).
          Development default: the student_test partition.
          Demo day: point it at the evaluator trips, e.g.
              set EVAL_DIR=D:\evaluator_test      (Windows CMD)
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.abspath(os.environ.get("EVAL_DIR", os.path.join(REPO_ROOT, "student_test")))
