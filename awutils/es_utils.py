import requests
import json

from elasticsearch import helpers
from elasticsearch import Elasticsearch


class ES:
    def __init__(self, index_name):
        self.index_name = index_name
        # self.es = Elasticsearch(hosts="127.0.0.1", port=9200)
        self.es = Elasticsearch(hosts="114.212.85.73", port=9200, maxsize=20)
        if not self.es.indices.exists(self.index_name):
            self.build_index()
        pass

    def build_index(self):
        # index_name = self.index_name
        settings = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 1,

                    # configure our default similarity algorithm explicitly to use bm25,
                    # this allows it to use it for all the fields
                    'similarity': {
                        'default': {
                            'type': 'BM25'
                        }
                    }
                }
            },
            # we will be indexing our documents in the title field using the English analyzer,
            # which removes stop words for us, the default standard analyzer doesn't have
            # this preprocessing step
            # https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis.html
            'mappings': {
                # this key is the "type", which will be explained in the next code chunk
                'properties': {
                    'content': {
                        'type': 'text',
                        # 'analyzer': 'english'
                        'analyzer': 'whitespace'
                        # 'analyzer': 'hanlp'
                    }
                }

            }
        }
        # headers = {'Content-Type': 'application/json'}
        print(json.dumps(settings))
        # response = requests.put(f'http://localhost:9200/{index_name}', data=json.dumps(settings), headers=headers)
        self.es.indices.create(index=self.index_name, body=settings)
        # print(response)

    def index_corpus(self, corpus):
        # index_name = self.index_name
        # url = f'http://localhost:9200/{index_name}/_doc'
        # for document in corpus:
        #     # we insert the document into the 'title' field
        #     data = {'content': document}
        #     # response = requests.post(url, data=json.dumps(data), headers=headers)
        #     response = requests.post(url, data=json.dumps(data))
        print(f"indexing {len(corpus)} for {self.index_name}")
        actions = [
            {
                "_index": self.index_name,
                # "_type": "tickets",
                "_id": i,
                "_source": {
                    "content": t
                }
            }
            for i, t in enumerate(corpus)
        ]

        helpers.bulk(self.es, actions)
        self.es.indices.refresh(index=self.index_name)
        print("index success")
        # print(response)

    def search(self, query, topk=20):
        query = {
            "query": {
                "match": {
                    "content": query
                }
            }
        }
        res = self.es.search(index=self.index_name, size=topk, body=query)
        # res = self.es.search(index=self.index_name, body=query)
        # print(res)
        res = [(t['_id'], t['_score']) for t in res['hits']['hits']]
        # print(res)
        return res

    def delete(self):
        # response = requests.delete(f'http://localhost:9200/{self.index_name}')
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
        # print(response)


def test_esbm25():
    model = ES(index_name="test")
    model.delete()
    model.build_index()
    corpus = ["中国 牛逼", "日本 傻逼， 中国 牛逼"]
    model.index_corpus(corpus)
    print(model.search(query="中国 牛逼"))


if __name__ == '__main__':
    test_esbm25()
