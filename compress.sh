#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: ./compress.sh <Student ID>"
    exit 1
fi

student_id="$1"
archive="${student_id}.tar.gz"

if [ ! -d output ]; then
    echo "Missing output directory."
    exit 1
fi

if [ ! -d student ]; then
    echo "Missing student directory."
    exit 1
fi

tar -czf "${archive}" output student
echo "Created ${archive}"
