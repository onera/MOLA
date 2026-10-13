"""Command line interface for the examples runner."""

from __future__ import annotations

import argparse

from .cleanup import clear_examples
from .postprocess import postprocess_examples
from .repatriate import repatriate_examples
from .status import check_status, validate_examples
from .submit import submit_examples


BEHAVIOR_TO_EXIT_CODE = {"continue": 0, "fail": 1, "retry": 2}


def build_parser():
    parser = argparse.ArgumentParser(description="MOLA CI/CD Examples Automation Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    submit = subparsers.add_parser("submit", help="Submit all examples")
    submit.add_argument("--source-user", help="User owning the remote example paths")
    submit.add_argument("-f", "--force", action="store_true")
    submit.add_argument("-n", "--dry-run", action="store_true")
    submit.add_argument("--failed-only", action="store_true")

    status = subparsers.add_parser("status", help="Check status of examples")
    status.add_argument("--source-user", help="User owning the remote example paths")
    repatriate = subparsers.add_parser(
        "repatriate", help="Repatriate completed examples"
    )
    repatriate_mode = repatriate.add_mutually_exclusive_group()
    repatriate_mode.add_argument(
        "--full", dest="repatriate_mode", action="store_const", const="full"
    )
    repatriate_mode.add_argument(
        "--light", dest="repatriate_mode", action="store_const", const="light"
    )
    repatriate.add_argument("--tag", help="Release tag used by the result spaces")
    repatriate.add_argument("--root", help="Custom root path for repatriation destination (e.g. mola/v2.0.1-rc12 or lbernard/MOLA/Dev)")
    repatriate.add_argument("--source-user", help="User owning the remote example paths")
    repatriate.set_defaults(repatriate_mode="full")
    postprocess = subparsers.add_parser(
        "postprocess", help="Run comparison and visualization scripts"
    )
    postprocess.add_argument("--tag", help="Release tag used by the result spaces")
    postprocess.add_argument("--root", help="Custom root path for repatriation destination (e.g. mola/v2.0.1-rc12 or lbernard/MOLA/Dev)")
    postprocess.add_argument("--source-user", help="User owning the remote example paths")
    validate = subparsers.add_parser("validate", help="Validate results")
    validate.add_argument("--source-user", help="User owning the remote example paths")

    clear = subparsers.add_parser("clear", help="Clean local artifacts and remote cases")
    group = clear.add_mutually_exclusive_group(required=True)
    group.add_argument("--local-only", action="store_true")
    group.add_argument("--failed-only", action="store_true")
    group.add_argument("--all", dest="all_cases", action="store_true")
    clear.add_argument("--source-user", help="User owning the remote example paths")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "submit":
        submit_examples(
            force=args.force,
            dry_run=args.dry_run,
            failed_only=args.failed_only,
            source_user=args.source_user,
        )
    elif args.command == "status":
        behavior = check_status(source_user=args.source_user)
        raise SystemExit(BEHAVIOR_TO_EXIT_CODE[behavior])
    elif args.command == "repatriate":
        repatriate_examples(
            mode=args.repatriate_mode,
            tag=args.tag,
            root=args.root,
            source_user=args.source_user,
        )
    elif args.command == "postprocess":
        postprocess_examples(
            tag=args.tag,
            root=args.root,
            source_user=args.source_user,
        )
    elif args.command == "validate":
        validate_examples(source_user=args.source_user)
    elif args.command == "clear":
        clear_kwargs = dict(
            local_only=args.local_only,
            failed_only=args.failed_only,
            all_cases=args.all_cases,
        )
        if args.source_user is not None:
            clear_kwargs["source_user"] = args.source_user
        if not clear_examples(**clear_kwargs):
            raise SystemExit(1)
    else:
        build_parser().print_help()
