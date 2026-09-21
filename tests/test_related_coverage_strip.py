"""AP embeds unrelated headline rails INSIDE the story body, and they poison association.

They are flattened into the text with NO separator:

    "...paired at pro-am event in Scotland More than 180 people have been killed from
     Hurricane Helene..."

so the boundary cannot be recovered downstream. Measured on the Helene feed, that caused
both of the spatial anchor's errors -- its ONLY miss and its ONLY false positive:

    dozens  keyed `tennessee`  -- a Taiwan typhoon, from the next headline in the rail
    180     keyed `scotland`   -- a GENUINE Helene figure, from a golf headline

The fix removes the rail at HTML->text time, where it is a DOM node.
"""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "build_helene_feed", Path("tools/ekf_showcase/build_helene_feed.py"))
bhf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bhf)

PAGE = """<html><body>
  <h1>Helene death toll rises</h1>
  <div class="RichTextStoryBody">
    <p>Rescue crews searched <span class="LinkEnhancement"><a class="Link"
       href="/x">western North Carolina</a></span> for survivors.</p>
    <div class="PageListEnhancementGeneric Enhancement">
      <bsp-list-loadmore class="PageListStandardB" data-gtm-region="RELATED COVERAGE">
        PGA Tour commissioner paired at pro-am event in Scotland
        Typhoon headed to Taiwan injures dozens
      </bsp-list-loadmore>
    </div>
    <p>More than 180 people have been killed from Hurricane Helene.</p>
  </div>
</body></html>"""


def test_the_rail_is_removed():
    text = bhf.plain(PAGE)
    assert "Scotland" not in text
    assert "Taiwan" not in text
    assert "PGA Tour" not in text


def test_the_body_survives_intact():
    text = bhf.plain(PAGE)
    assert "More than 180 people have been killed" in text
    assert "Rescue crews searched" in text
    assert "Helene death toll rises" in text, "the headline is part of the body"


def test_inline_link_enhancements_are_PRESERVED():
    """THE NON-REGRESSION THAT MATTERS. `contains(@class,"Enhancement")` also matches
    `LinkEnhancement`, an inline link inside the prose -- stripping those deletes real
    article words. Measured: the broad selector matches 26-48 nodes per article, the
    block selector 1-2."""
    text = bhf.plain(PAGE)
    assert "western North Carolina" in text, "an inline link is body text, not a rail"


def test_a_rail_adjacent_figure_is_no_longer_next_to_a_foreign_place():
    """The exact failure: `180` sat beside `Scotland` with no separator between them."""
    text = bhf.plain(PAGE)
    i = text.find("180 people have been killed")
    assert i > 0
    assert "Scotland" not in text[max(0, i - 300):i]


def test_missing_lxml_raises_rather_than_returning_page_furniture(monkeypatch):
    """A silent fallback returned nav, rails and footer as article text -- a 26,598-char
    median document against 5,100 -- so articles about a four-day workweek 'named' Florida
    and Georgia. Failing loudly is the point."""
    import builtins
    real = builtins.__import__

    def no_lxml(name, *a, **k):
        if name.startswith("lxml"):
            raise ImportError("no lxml")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_lxml)
    with pytest.raises(RuntimeError, match="lxml is required"):
        bhf.plain(PAGE)
