"""User-run TSV bootstrap for a fresh team package; no EDA database required."""
import argparse
import hashlib
import os
from pathlib import Path

from b01_common import ROOT, CODEX, atomic_json, environment, identity
from b01_match_common import read_json, stage_lock

SOURCE_COLUMNS = ['entity_id', 'business_name', 'business_address', 'country']
TRUTH_COLUMNS = ['source1_entity_id', 'matched_entity_ids']


def signature(paths):
    return {'inputs': [identity(p) for p in paths],
            'bootstrap_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def reusable(destination, stamp, expected):
    import pyarrow.parquet as pq
    if not stamp.exists():
        return False
    saved = read_json(stamp)
    if saved['signature'] != expected:
        raise RuntimeError(f'Bootstrap inputs/code changed: {stamp}. Use a fresh extracted package.')
    if not destination.is_file() or identity(destination) != saved['output']:
        raise RuntimeError(f'Completed bootstrap output changed or is missing: {destination}')
    if pq.ParquetFile(destination).metadata.num_rows != saved['rows']:
        raise RuntimeError(f'Invalid bootstrap row count: {destination}')
    print(f'Reuse {destination.name}: {saved["rows"]:,} rows.', flush=True)
    return True


def convert_tsv(source, destination, batch_rows=25000):
    """Literal tab fields, UTF-8, no CSV quoting, empty strings, original row order."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm
    destination.parent.mkdir(parents=True, exist_ok=True)
    stamp = destination.with_suffix('.complete.json')
    expected = signature([source])
    if reusable(destination, stamp, expected):
        return
    columns = TRUTH_COLUMNS if source.stem == 'train_ground_truth' else SOURCE_COLUMNS
    schema = pa.schema([(name, pa.large_string()) for name in columns])
    partial = destination.with_suffix('.parquet.part')
    checksum = hashlib.sha256()
    rows, buffered_bytes = 0, 0
    values = {name: [] for name in columns}
    with source.open('rb') as stream, pq.ParquetWriter(partial, schema, compression='zstd') as writer, \
            tqdm(total=source.stat().st_size, desc=source.stem, unit='B', unit_scale=True, dynamic_ncols=True) as bar:
        header = stream.readline()
        checksum.update(header)
        bar.update(len(header))
        if header.decode('utf-8').rstrip('\r\n').split('\t') != columns:
            raise ValueError(f'Unexpected TSV header: {source}')
        for line_number, line in enumerate(stream, 2):
            checksum.update(line)
            buffered_bytes += len(line)
            fields = line.decode('utf-8').rstrip('\r\n').split('\t')
            if len(fields) != len(columns) or not fields[0]:
                raise ValueError(f'Invalid TSV fields/ID in {source.name}, line {line_number}')
            for name, value in zip(columns, fields):
                values[name].append(value)
            rows += 1
            if len(values[columns[0]]) == batch_rows:
                writer.write_table(pa.Table.from_pydict(values, schema=schema))
                values = {name: [] for name in columns}
                bar.update(buffered_bytes)
                buffered_bytes = 0
                bar.set_postfix(rows=rows, refresh=False)
        if values[columns[0]]:
            writer.write_table(pa.Table.from_pydict(values, schema=schema))
        bar.update(buffered_bytes)
    if identity(source) != expected['inputs'][0]:
        raise RuntimeError(f'Input changed while reading: {source}')
    partial.replace(destination)
    atomic_json(stamp, {'signature': expected, 'output': identity(destination),
                        'rows': rows, 'raw_sha256': checksum.hexdigest()})


def sql_path(path):
    return "'" + str(path).replace("'", "''") + "'"


def language_pairs(parquet_dir, destination, memory_gib=2, threads=4):
    """Rebuild the original positive-pair selection directly from train and truth."""
    import duckdb
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm
    sources = [parquet_dir / f'train_source{s}.parquet' for s in (1, 2, 3)]
    truth = parquet_dir / 'train_ground_truth.parquet'
    expected = signature([*sources, truth])
    destination.parent.mkdir(parents=True, exist_ok=True)
    stamp = destination.with_suffix('.complete.json')
    if reusable(destination, stamp, expected):
        return
    spill = CODEX / 'tmp/raw_bootstrap_duckdb'
    spill.mkdir(parents=True, exist_ok=True)
    parts_dir = CODEX / 'data/raw_bootstrap/language_parts'
    parts_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(':memory:')
    con.execute(f"SET memory_limit='{memory_gib}GB'")
    con.execute(f'SET threads={threads}')
    con.execute(f'SET temp_directory={sql_path(spill)}')
    con.execute('SET preserve_insertion_order=false')
    parts = []
    try:
        for source in (2, 3):
            part = parts_dir / f'source{source}.parquet'
            part_stamp = part.with_suffix('.complete.json')
            part_signature = {**expected, 'source': source}
            parts.append(part)
            if reusable(part, part_stamp, part_signature):
                continue
            print(f'Join Source {source} to training ownership and S1. DuckDB builds the join before rows stream; setup has no ETA.', flush=True)
            # Same predicate as audit_baseline_revision.py. No test rows or labels enter this join.
            query = f"""
                WITH edges AS (
                  SELECT source1_entity_id AS s1_id,
                         unnest(string_split(coalesce(matched_entity_ids, ''), ',')) AS target_id
                  FROM read_parquet({sql_path(truth)})
                  WHERE coalesce(matched_entity_ids, '') <> ''
                )
                SELECT e.s1_id, e.target_id, {source}::INTEGER AS source,
                       coalesce(a.business_name, '') AS n1, coalesce(a.business_address, '') AS a1,
                       coalesce(b.business_name, '') AS n2, coalesce(b.business_address, '') AS a2
                FROM edges e
                JOIN read_parquet({sql_path(sources[0])}) a ON e.s1_id = a.entity_id
                JOIN read_parquet({sql_path(sources[source - 1])}) b ON e.target_id = b.entity_id
                WHERE a.country = 'India'
                  AND (regexp_matches(coalesce(b.business_name, ''), '[ऀ-ൿ]')
                       OR regexp_matches(coalesce(b.business_address, ''), '[ऀ-ൿ]'))
            """
            reader = con.execute(query).fetch_record_batch(rows_per_batch=25000)
            partial = part.with_suffix('.parquet.part')
            rows = 0
            with pq.ParquetWriter(partial, reader.schema, compression='zstd') as writer, \
                    tqdm(desc=f'Write native matched pairs S{source}', unit='pairs', dynamic_ncols=True) as bar:
                for batch in reader:
                    writer.write_batch(batch)
                    rows += batch.num_rows
                    bar.update(batch.num_rows)
            partial.replace(part)
            atomic_json(part_stamp, {'signature': part_signature, 'output': identity(part), 'rows': rows})
    finally:
        con.close()
    count = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    partial = destination.with_suffix('.parquet.part')
    with pq.ParquetWriter(partial, pq.ParquetFile(parts[0]).schema_arrow, compression='zstd') as writer, \
            tqdm(total=count, desc='Combine training language pairs', unit='pairs', dynamic_ncols=True) as bar:
        for part in parts:
            for batch in pq.ParquetFile(part).iter_batches(batch_size=25000):
                writer.write_batch(batch)
                bar.update(batch.num_rows)
    if signature([*sources, truth]) != expected:
        raise RuntimeError('Parquet inputs changed during the language join.')
    partial.replace(destination)
    atomic_json(stamp, {'signature': expected, 'output': identity(destination), 'rows': count})


def bootstrap():
    environment()
    if not (ROOT / 'TEAM_PACKAGE.json').is_file():
        raise RuntimeError('Run bootstrap only inside a fresh extracted team package. Existing experiments are protected.')
    _lock = stage_lock(CODEX / 'data/raw_bootstrap')
    parquet_dir = CODEX / 'data/raw_bootstrap/parquet'
    legacy = ROOT / 'matcher/eda/pq'
    if legacy.exists() or legacy.is_symlink():
        if not legacy.is_symlink() or legacy.resolve() != parquet_dir.resolve():
            raise RuntimeError('Existing matcher/eda/pq is not this bootstrap link. Use a clean extraction; nothing was overwritten.')
    files = [(split, f'{split}_source{s}') for split in ('train', 'test') for s in (1, 2, 3)]
    files.insert(3, ('train', 'train_ground_truth'))
    sources = [(ROOT / 'student_resource/dataset' / split / f'{name}.tsv', parquet_dir / f'{name}.parquet') for split, name in files]
    for source, _ in sources:
        if not source.is_file():
            raise FileNotFoundError(f'Place the supplied base TSV here: {source}')
    print('Build raw Parquet files and supervised language pairs only. Each completed file is resumable.', flush=True)
    for source, destination in sources:
        convert_tsv(source, destination)
    language_pairs(parquet_dir, CODEX / 'data/language/baseline_review_pairs.parquet')
    if not legacy.is_symlink():
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.symlink_to(os.path.relpath(parquet_dir, legacy.parent), target_is_directory=True)
    print('Bootstrap complete. matcher/eda/pq is a local compatibility link; no Claude code or old cache is needed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    bootstrap()
