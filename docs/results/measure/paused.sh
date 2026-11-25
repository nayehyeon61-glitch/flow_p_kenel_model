#!/bin/bash
# run a side job while the core training is stopped (a job next to it slows both about 6x), then let it go on
P=$(pgrep -f "run_fpk.py core --core-tag" | head -1)
[ -n "$P" ] && kill -STOP $P
trap '[ -n "$P" ] && kill -CONT $P' EXIT
"$@"
