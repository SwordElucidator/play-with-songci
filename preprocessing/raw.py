import os
import re


def read_raw():
    filenames = os.listdir('data/songci/')
    authors = []
    part = 0
    current_author = None
    current_title = None
    current_lines = ""
    out = []
    space_lines = 0
    for filename in filenames:
        if filename.endswith(".txt"):
            with open(os.path.join('data/songci/', filename)) as f:
                for line in f:
                    if "全宋词" in line:
                        part += 1
                        continue
                    line = line.strip()
                    if part == 1 and line:
                        authors.append(line)
                    elif part == 2:
                        if not line:
                            space_lines += 1
                            continue

                        if line in authors and space_lines >= 2:
                            space_lines = 0
                            if current_title and current_lines:
                                out.append({
                                    "title": current_title,
                                    "text": current_lines,
                                    "author": current_author
                                })
                            current_title, current_lines = None, ""
                            current_author = line
                            continue

                        line = line.split("（")[0]
                        if len(line) <= 8:
                            if current_title and current_lines:
                                out.append({
                                    "title": current_title,
                                    "text": current_lines,
                                    "author": current_author
                                })
                            current_title, current_lines = line, ""
                        else:
                            current_lines += line

    return out
