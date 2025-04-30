#!/usr/bin/env bash

# Script to remove all contents in the models directory with confirmation
MODELS_DIR="models"

echo "This will delete all files in the '$MODELS_DIR' directory."
read -p "Are you sure you want to continue? [y/N]: " confirm
if [[ $confirm =~ ^[Yy]$ ]]; then
  rm -rf "$MODELS_DIR"/*
  echo "All files in '$MODELS_DIR' have been deleted."
else
  echo "Operation canceled."
fi
