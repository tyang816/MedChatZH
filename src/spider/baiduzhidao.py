from baiduspider import BaiduSpider
from pprint import pprint

spider = BaiduSpider()

pprint(spider.search_zhidao(query="重庆").plain)