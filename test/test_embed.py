import unittest

from app.embed import Embed

class Test_Embed(unittest.TestCase):

    embed = None

    @classmethod
    def setUpClass(self):
        self.embed = Embed()

    # def test_get_vectors(self):        
    #     string = "I went to a show"
    #     results = self.embed.get_vectors(string)
    #     self.assertIsNotNone(results)
    #     print(results["duration"])

    # def test_add_document_vectors(self):        
    #     string = "I went to a show"
    #     metadata = {"source":"test"}
    #     results = self.embed.get_vectors(string)
    #     self.assertIsNotNone(results)
    #     document = {"document":string,
    #                 "metadata":metadata,
    #                 "vectors":results['results']['embedding'],
    #                 "id":"1"}
    #     self.embed.add_document_vectors(document)

    def test_add_document(self):        
        string = "I went to a play"
        metadata = {"source":"test"}
        document = {"document":string,
                    "metadata":metadata,
                    "id":"1"}
        self.embed.add_document(document)        
        results = self.embed.quick_query("I went to a play")
        self.assertIsNotNone(results)

    # def test_quick_query(self):
    #     string = "I went to a show"
    #     metadata = {"source":"test"}
    #     results = self.embed.get_vectors(string)
    #     self.assertIsNotNone(results)
    #     document = {"document":string,
    #                 "metadata":metadata,
    #                 "vectors":results['results']['embedding'],
    #                 "id":"1"}
    #     self.embed.add_document(document)        
    #     # now search
    #     results = self.embed.quick_query("I went to a play")
    #     self.assertIsNotNone(results)

    # def test_collection_count(self):
    #     results = self.embed.collection_count()
    #     print("result count")
    #     print(results)
