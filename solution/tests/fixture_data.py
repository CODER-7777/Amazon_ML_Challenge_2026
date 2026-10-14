"""Small deterministic data for exercising the complete pipeline, not scoring it."""
import csv
from pathlib import Path

FIELDS = ['entity_id','business_name','business_address','country']


def write_tsv(path, fields, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(fields)
        writer.writerows(records)


def make_dataset(root):
    root = Path(root)
    all_truth = {}
    for split, count, offset in [('train',240,0), ('test',60,10000)]:
        sources = {1:[], 2:[], 3:[]}
        truth = []
        countries = ['US','India'] if split=='train' else ['US','India','France']
        for i in range(count):
            group, branch = divmod(i,3)
            country = countries[group%len(countries)]
            aid = f'S1-{offset+i}'
            brand = f'Meridian Quartz{offset+group} Robotics'
            if country=='France':
                brand = f'École Quartz{offset+group} Mécanique'
            house = 10+branch*30
            address = f'{house} Willow Street, District {group}, City {group}, {10000+group}'
            sources[1].append([aid,brand+' Ltd',address,country])
            matched = []
            if i%11:
                for source, variant in [(2,0),(3,1)]+([(2,2)] if i%4==0 else []):
                    tid = f'S{source}-{offset*10+i*10+variant}'
                    name = brand if variant!=1 else brand.replace('Robotics','Robotix').replace('École','Ecole')
                    addr = address.replace('Street','St')
                    if i%17==0 and variant==1:
                        addr = ''
                    sources[source].append([tid,name,addr,country])
                    matched.append(tid)
            truth.append([aid,','.join(matched)])
            # Close-looking but genuinely different business: useful hard negative.
            if branch==0:
                sources[2].append([f'S2-neg-{offset+i}',brand,'9999 Desert Road, Remote Town 99999',country])
        # A target country absent from the reference must safely yield no candidates.
        sources[3].append([f'S3-unlinked-{offset}','Completely Unrelated','404 Nowhere Road','UnknownCountry'])
        for source, records in sources.items():
            write_tsv(root/split/f'{split}_source{source}.tsv',FIELDS,records)
        all_truth[split] = {anchor:set(filter(None,ids.split(','))) for anchor,ids in truth}
        if split=='train':
            write_tsv(root/split/'train_ground_truth.tsv',['source1_entity_id','matched_entity_ids'],truth)
    return all_truth
