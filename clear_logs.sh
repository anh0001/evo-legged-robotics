#!/usr/bin/env bash
LOG_DIR="logs"

if [ ! -d "$LOG_DIR" ]; then
  echo "Directory '$LOG_DIR' not found."
  exit 1
fi

read -p "This will delete everything under '$LOG_DIR/'. Continue? [y/N] " ans
if [[ $ans =~ ^[Yy]$ ]]; then
  rm -rf "$LOG_DIR"/* && echo "Logs cleared."
else
  echo "Aborted."
fi