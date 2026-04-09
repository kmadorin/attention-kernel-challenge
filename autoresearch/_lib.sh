#!/usr/bin/env bash
# _lib.sh — shared helpers sourced by tier scripts.

# check_exit <result_line>
# Exits with the same code that _run_eval.sh would: 0 if valid=true, 1 otherwise.
check_exit() {
  local line="$1"
  if echo "${line}" | grep -qP '\bvalid=true\b'; then
    return 0
  else
    return 1
  fi
}

# extract_field <field_name> <result_line>
# Prints the value of field=value from the result line, or "NA" if absent.
extract_field() {
  local field="$1" line="$2"
  echo "${line}" | grep -oP "(?<=\b${field}=)[^\t]+" || echo "NA"
}
