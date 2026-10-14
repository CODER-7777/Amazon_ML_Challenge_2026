import csv
import hashlib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

from rapidfuzz import fuzz

VERSION = 1
LEGAL = set('inc incorporated corp corporation llc llp ltd limited pvt private company co plc sa sas sarl eurl gmbh'.split())
ADDRESS_WORDS = dict(street='st', road='rd', avenue='ave', boulevard='blvd', lane='ln', drive='dr', highway='hwy', apartment='apt', building='bldg', floor='fl', suite='ste', number='no', nagar='ngr', near='nr', opposite='opp')
COMMON_ADDRESS = set('the of and nr near opp opposite behind beside floor fl bldg building'.split())


def stable_hash(value):
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), 'little')


def fold(entity_id, seed=42):
    # IDs partition entities; they are never predictive features.
    return 'valid' if stable_hash(f'{seed}:{entity_id}') % 5 == 0 else 'train'


def rows(path):
    with open(path, encoding='utf-8', newline='') as stream:
        yield from csv.DictReader(stream, delimiter='\t')


def normalize(text, address=False):
    text = unicodedata.normalize('NFKD', text.casefold())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r'https?://|www\.', ' ', text)
    text = re.sub(r'\.(com|net|org|in|fr|co)\b', ' ', text)
    words = re.findall(r'[^\W_]+', text.replace('&', ' and '), re.UNICODE)
    return ' '.join(ADDRESS_WORDS.get(w, w) if address else w for w in words)


def core_name(name):
    return ' '.join(w for w in name.split() if w not in LEGAL and w not in {'the', 'and'}) or name


def grams(name):
    s = ''.join(c for c in name if c.isalnum())
    return {s[i:i+3] for i in range(max(0, len(s)-2))}


def country_key(country):
    return normalize(country)


def index_name(country):
    return hashlib.sha256(country_key(country).encode()).hexdigest()[:16] + '.sqlite'


def prepare(row):
    n = normalize(row['business_name'])
    return dict(id=row['entity_id'], country=country_key(row['country']), name=n,
                core=core_name(n), address=normalize(row.get('business_address', ''), True))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, version=VERSION)


def build_indexes(source, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    signature = fingerprint(source)
    marker = dest / 'complete.json'
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if manifest.get('source') == signature and all((dest/index_name(c)).is_file() for c in manifest['countries']):
            print(f'Reusing completed index: {dest}', flush=True)
            return
    conns, counts = {}, {}
    try:
        for i, raw in enumerate(rows(source), 1):
            r = prepare(raw)
            country = r['country']
            if country not in conns:
                path = dest / (index_name(country) + '.building')
                path.unlink(missing_ok=True)
                c = sqlite3.connect(path)
                c.execute('PRAGMA journal_mode=OFF')
                c.execute('PRAGMA synchronous=OFF')
                c.execute('PRAGMA cache_size=-16384')
                c.execute('CREATE TABLE records(entity_id TEXT PRIMARY KEY, name TEXT, address TEXT, n TEXT, g TEXT, a TEXT)')
                c.execute("CREATE VIRTUAL TABLE search USING fts5(n,g,a,content='records',content_rowid='rowid',tokenize='unicode61 remove_diacritics 2')")
                c.execute("CREATE VIRTUAL TABLE vocab USING fts5vocab(search,'col')")
                conns[country], counts[country] = c, 0
            c = conns[country]
            g = ' '.join(sorted(grams(r['core'])))
            cursor = c.execute('INSERT INTO records VALUES (?,?,?,?,?,?)', (r['id'], r['name'], r['address'], r['core'], g, r['address']))
            c.execute('INSERT INTO search(rowid,n,g,a) VALUES (?,?,?,?)', (cursor.lastrowid, r['core'], g, r['address']))
            counts[country] += 1
            if i % 100000 == 0:
                for db in conns.values():
                    db.commit()
                print(f'Indexed {i:,} anchors', flush=True)
        for country, c in conns.items():
            c.commit()
            c.execute("INSERT INTO search(search) VALUES ('optimize')")
            c.commit()
            c.close()
            (dest / (index_name(country) + '.building')).replace(dest / index_name(country))
        marker.write_text(json.dumps(dict(source=signature, countries=counts), indent=2))
    finally:
        for c in conns.values():
            c.close()


class Retriever:
    def __init__(self, directory, max_candidates=3, retrieval_limit=12, max_postings=3000, lexical_floor=.32):
        self.directory = Path(directory)
        self.max_candidates, self.retrieval_limit = max_candidates, retrieval_limit
        self.max_postings, self.lexical_floor = max_postings, lexical_floor
        self.connections, self.frequencies = {}, {}
        self.countries = set(json.loads((self.directory/'complete.json').read_text())['countries'])

    def connect(self, country):
        if country not in self.countries:
            return None
        if country in self.connections:
            return self.connections[country]
        path = self.directory / index_name(country)
        if not path.exists():
            raise FileNotFoundError(f'Incomplete anchor index: {path}. Rebuild the index before inference.')
        c = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        c.execute('PRAGMA cache_size=-32768')
        c.execute('PRAGMA temp_store=MEMORY')
        self.connections[country] = c
        self.frequencies[country] = {(col, term): count for term, col, count in c.execute('SELECT term,col,doc FROM vocab WHERE doc<=?', (self.max_postings,))}
        return c

    @staticmethod
    def record(row, country):
        return dict(id=row[0], name=row[1], address=row[2], core=row[3], country=country)

    def get(self, country, entity_id):
        c = self.connect(country)
        if c is None:
            return None
        row = c.execute('SELECT entity_id,name,address,n FROM records WHERE entity_id=?', (entity_id,)).fetchone()
        return self.record(row, country) if row else None

    def retrieve(self, target):
        c = self.connect(target['country'])
        if c is None:
            return []
        freq = self.frequencies[target['country']]
        fields = [('n', set(target['core'].split()), 3), ('g', grams(target['core']), 4), ('a', set(target['address'].split()) - COMMON_ADDRESS, 4)]
        pool = {}
        for field, tokens, count in fields:
            selected = sorted((freq[(field, t)], t) for t in tokens if (field, t) in freq)[:count]
            if not selected:
                continue
            query = ' OR '.join(f'{field}:"{token}"' for _, token in selected)
            found = c.execute('SELECT rowid FROM search WHERE search MATCH ? ORDER BY bm25(search,3.0,1.0,2.0) LIMIT ?', (query, self.retrieval_limit)).fetchall()
            if found:
                ids = [r[0] for r in found]
                placeholders = ','.join('?' for _ in ids)
                for row in c.execute(f'SELECT entity_id,name,address,n FROM records WHERE rowid IN ({placeholders})', ids):
                    pool[row[0]] = self.record(row, target['country'])
        ranked = []
        for a in pool.values():
            n = fuzz.WRatio(a['core'], target['core']) / 100
            ad = fuzz.WRatio(a['address'], target['address']) / 100 if a['address'] and target['address'] else 0.
            ranked.append((max(n*.82 + ad*.18, ad*.80 + n*.20), a['id'], a))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        cutoff = max(self.lexical_floor, ranked[0][0] - .16) if ranked else 1.
        return [a for score, _, a in ranked[:self.max_candidates] if score >= cutoff]

    def close(self):
        for c in self.connections.values():
            c.close()
        self.connections.clear()
        self.frequencies.clear()


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 0.


def overlap(a, b):
    return len(a & b) / min(len(a), len(b)) if a and b else 0.


def length_ratio(a, b):
    return min(len(a), len(b)) / max(len(a), len(b)) if a or b else 0.


FEATURE_NAMES = 'name_ratio name_token_sort name_token_set name_wratio name_compact_ratio name_token_jaccard name_token_overlap name_trigram_jaccard name_equal name_length_ratio address_ratio address_token_sort address_token_set address_wratio address_token_jaccard address_token_overlap address_equal address_length_ratio address_missing number_jaccard number_overlap number_conflict postcode_overlap postcode_conflict first_number_equal first_number_conflict both_fields_agree name_address_min name_word_count target_name_word_count address_word_count target_address_word_count source3'.split()


def features(anchor, target):
    n, m, a, b = anchor['core'], target['core'], anchor['address'], target['address']
    nt, mt, at, bt = set(n.split()), set(m.split()), set(a.split()), set(b.split())
    nums_a, nums_b = re.findall(r'\b\d+\b', a), re.findall(r'\b\d+\b', b)
    na, nb = set(nums_a), set(nums_b)
    pa, pb = {x for x in na if len(x) in (5, 6)}, {x for x in nb if len(x) in (5, 6)}
    nr = fuzz.ratio(n, m)/100
    ar = fuzz.ratio(a, b)/100 if a and b else 0.
    hn = bool(na and nb)
    return [nr, fuzz.token_sort_ratio(n,m)/100, fuzz.token_set_ratio(n,m)/100, fuzz.WRatio(n,m)/100,
            fuzz.ratio(n.replace(' ',''),m.replace(' ',''))/100, jaccard(nt,mt), overlap(nt,mt), jaccard(grams(n),grams(m)), float(bool(n) and n==m), length_ratio(n,m), ar,
            fuzz.token_sort_ratio(a,b)/100 if a and b else 0., fuzz.token_set_ratio(a,b)/100 if a and b else 0., fuzz.WRatio(a,b)/100 if a and b else 0.,
            jaccard(at,bt), overlap(at,bt), float(bool(a) and a==b), length_ratio(a,b), float(not a or not b), jaccard(na,nb), overlap(na,nb),
            float(hn and not na & nb), overlap(pa,pb), float(bool(pa and pb) and not pa & pb), float(hn and nums_a[0]==nums_b[0]), float(hn and nums_a[0]!=nums_b[0]),
            nr*ar, min(nr,ar), len(nt), len(mt), len(at), len(bt), float(target['id'].startswith('S3-'))]


def macro_score(truth, predictions):
    total = 0.
    for anchor, expected in truth.items():
        predicted = predictions.get(anchor, set())
        if not expected:
            total += float(not predicted)
        else:
            tp = len(expected & predicted)
            total += 1.25*tp / (1.25*tp + len(predicted-expected) + .25*len(expected-predicted))
    return total/len(truth) if truth else 0.
