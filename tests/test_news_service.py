import unittest


from web_api.news_service import build_trending, classify_sentiment, extract_tags, gdelt_article_to_news_item


class TestNewsService(unittest.TestCase):
    def test_gdelt_article_to_news_item_basic(self) -> None:
        art = {
            "url": "https://finance.ifeng.com/c/8p3DIOLve2l",
            "title": "央行降息利好：流动性宽松提振市场",
            "seendate": "20251213T073000Z",
            "domain": "finance.ifeng.com",
        }
        item = gdelt_article_to_news_item(art)

        self.assertEqual(item["title"], art["title"])
        self.assertEqual(item["source"], "finance.ifeng.com")
        self.assertEqual(item["sentiment"], "Bullish")
        self.assertIn("宏观", item["tags"])
        self.assertEqual(item["url"], art["url"])
        self.assertEqual(item["time"], "15:30")  # UTC+8
        self.assertEqual(len(item["id"]), 12)

    def test_sentiment_rules(self) -> None:
        self.assertEqual(classify_sentiment("利好消息：公司回购增持"), "Bullish")
        self.assertEqual(classify_sentiment("业绩预警，股价大跌承压"), "Bearish")
        self.assertEqual(classify_sentiment("公司公告：召开股东大会"), "Neutral")

    def test_tag_extraction(self) -> None:
        tags = extract_tags("北向资金净流出，半导体板块承压")
        self.assertIn("资金流向", tags)
        self.assertIn("半导体", tags)

        tags2 = extract_tags("")
        self.assertEqual(tags2, ["综合"])

    def test_build_trending(self) -> None:
        items = [
            {"tags": ["宏观", "货币政策"]},
            {"tags": ["宏观", "利率"]},
            {"tags": ["半导体"]},
        ]
        trending = build_trending(items, limit=3)
        self.assertEqual(trending[0], "#宏观")
        self.assertIn("#半导体", trending)


if __name__ == "__main__":
    unittest.main()

