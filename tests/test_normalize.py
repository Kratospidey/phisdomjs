import unittest
from phisdom.data.normalize import normalize_dom, extract_scripts


class TestNormalize(unittest.TestCase):
    def test_normalize_and_extract(self):
        html = """
        <html><head>
        <script>var x=1;\nconsole.log(x);</script>
        <script src="https://cdn.example.com/app.js" async></script>
        </head>
        <body><div  id='A'  class="c  b">  Hello   World </div></body></html>
        """
        norm = normalize_dom(html)
        self.assertIn("Hello World", norm)
        scripts = extract_scripts(html)
        self.assertEqual(len(scripts), 2)
        self.assertTrue(scripts[0]["inline"])  # first is inline
        self.assertFalse(scripts[1]["inline"])  # second is external
        self.assertIsNone(scripts[1]["text"])  # external text not in HTML


if __name__ == "__main__":
    unittest.main()
