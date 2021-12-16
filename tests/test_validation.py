import unittest
from netcoloc import validation


class MyTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_load_plain_MPO(self):
        mpo = validation.load_MPO()
        self.assertIsNotNone(mpo)

    def test_use_genes_assertion(self):
        self.assertRaises(AssertionError, validation.load_MPO, use_genes=True)

    def test_use_genes(self):
        pass

    def test_restrict_to(self):
        pass


if __name__ == '__main__':
    unittest.main()
