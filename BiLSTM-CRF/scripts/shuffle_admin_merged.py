#!/usr/bin/env python3
import os
import random
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))
# path relative to project root
in_path = os.path.join(base_dir, '..', 'data', 'my', 'admin_merged.json')
backup_path = in_path + '.bak'

if not os.path.exists(in_path):
    print(f"File not found: {in_path}")
    raise SystemExit(1)

# create backup
shutil.copy(in_path, backup_path)
print(f"Backup saved to: {backup_path}")

# read non-empty lines
with open(in_path, 'r', encoding='utf-8') as f:
    lines = [line for line in f if line.strip()]

if not lines:
    print("No lines to shuffle.")
    raise SystemExit(1)

random.shuffle(lines)

with open(in_path, 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(line.rstrip('\n') + '\n')

print(f"Shuffled {len(lines)} lines and wrote back to {in_path}")
