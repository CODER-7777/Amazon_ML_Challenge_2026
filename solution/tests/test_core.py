import csv
import tempfile
import unittest
from pathlib import Path
from ber.core import FEATURE_NAMES, Retriever, build_indexes, features, macro_score, normalize, prepare, index_name


class CoreTests(unittest.TestCase):
    def test_official_metric_and_singletons(self):
        truth = {'a': {'b','c'}, 's': set()}
        self.assertAlmostEqual(macro_score(truth, {'a':{'b','c','d'}}),(5/7+1)/2)
        self.assertAlmostEqual(macro_score({'a':{'b','c'}},{'a':{'b'}}),5/6)
        self.assertEqual(macro_score({'s':set()},{'s':{'wrong'}}),0.)
        self.assertEqual(macro_score({'a':{'b'}},{}),0.)

    def test_unicode_numbers_and_absent_address(self):
        self.assertEqual(normalize('École & Fils SARL'),'ecole and fils sarl')
        left = prepare(dict(entity_id='S1-1',business_name='Acme Ltd',business_address='12 Lake Road 560001',country='India'))
        right = prepare(dict(entity_id='S2-2',business_name='ACME',business_address='81 Lake Rd 560099',country='India'))
        values = dict(zip(FEATURE_NAMES,features(left,right)))
        self.assertEqual(len(values),len(FEATURE_NAMES))
        self.assertEqual(values['first_number_conflict'],1.)
        self.assertEqual(values['postcode_conflict'],1.)
        right['address']=''
        values = dict(zip(FEATURE_NAMES,features(left,right)))
        self.assertEqual(values['address_missing'],1.)
        self.assertEqual(values['first_number_conflict'],0.)

    def test_index_unknown_country_and_one_to_many(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/'source.tsv'
            fields=['entity_id','business_name','business_address','country']
            records=[['S1-1','École Atlas SARL','12 Rue Bleue Paris','France'],['S1-2','Atlas Bakery','88 Pine Street Austin','US']]
            with source.open('w',newline='') as f:
                writer=csv.writer(f,delimiter='\t');writer.writerow(fields);writer.writerows(records)
            build_indexes(source,root/'index')
            retriever=Retriever(root/'index')
            for tid in ['S2-1','S2-2','S3-3']:
                q=prepare(dict(entity_id=tid,business_name='Ecole Atlas',business_address='12 rue bleue Paris',country='France'))
                self.assertEqual(retriever.retrieve(q)[0]['id'],'S1-1')
            q['country']='unseen-without-reference'
            self.assertEqual(retriever.retrieve(q),[])
            retriever.close()
            build_indexes(source,root/'index')
            missing=root/'index'/index_name('France')
            missing.unlink()
            broken=Retriever(root/'index')
            with self.assertRaises(FileNotFoundError):
                broken.connect('france')
            build_indexes(source,root/'index')
            self.assertTrue(missing.exists())


if __name__=='__main__':
    unittest.main()
