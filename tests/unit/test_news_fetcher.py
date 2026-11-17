import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from bot import NewsFetcher


class TestNewsFetcher(unittest.TestCase):
	"""Validates RSS parsing resilience, HTML sanitization and date handling."""

	def test_clean_html_strips_tags_and_unescapes(self):
		raw = "<p>Breaking: <strong>Bitcoin</strong> surges past &dollar;100k &amp; holds!</p>"
		cleaned = NewsFetcher._clean_html(raw)
		self.assertEqual(cleaned, "Breaking: Bitcoin surges past $100k & holds!")

	def test_parse_published_date_with_tuple(self):
		entry = {"published_parsed": (2026, 11, 17, 12, 0, 0, 0, 0, 0)}
		parsed = NewsFetcher._parse_published_date(entry)
		self.assertEqual(parsed, datetime(2026, 11, 17, 12, 0, 0, tzinfo=timezone.utc))

	def test_parse_published_date_missing(self):
		entry = {}
		parsed = NewsFetcher._parse_published_date(entry)
		self.assertEqual(parsed, datetime.min.replace(tzinfo=timezone.utc))

	@patch("feedparser.parse")
	def test_fetch_single_feed_resilience_on_exception(self, mock_parse):
		mock_parse.side_effect = ConnectionError("Network unreachable")
		articles = NewsFetcher._fetch_single_feed({"name": "Test", "url": "https://test.local/rss"})
		self.assertEqual(articles, [])

	@patch.object(NewsFetcher, "_fetch_single_feed")
	def test_get_latest_news_fallback_when_empty(self, mock_fetch):
		mock_fetch.return_value = []
		output = NewsFetcher.get_latest_news(limit=5)
		self.assertEqual(output, "News data currently unavailable.")


if __name__ == "__main__":
	unittest.main()
