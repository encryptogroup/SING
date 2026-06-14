import os.path as osp

def read_hashes(path):
    result = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("#"):
                continue
            if line == "":
                continue

            result.append(line[0:64])

    return result

def read_failed_log(path):
    failed = []
    duplicates = dict()

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if "duplicate of" in line:
                fields = line.split()

                assert len(fields) == 4
                duplicate, _, _, original = fields

                assert len(duplicate) == 64
                assert len(original) == 64
                duplicates[duplicate] = original
            else:
                failed.append(line[0:64])

    return failed, duplicates

def load_metadata(shahash, assignment_hash, db_con):
    res = db_con.execute("SELECT key, value FROM metadata WHERE shahash = ? AND assignment_hash = ?", [shahash, assignment_hash])
    metadata = dict(res.fetchall())

    return metadata

def check_assignment_filter(metadata, assignment_filter):
    if assignment_filter is None:
        return True

    return assignment_filter(metadata)
