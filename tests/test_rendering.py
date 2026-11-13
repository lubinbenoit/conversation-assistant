import assistant.rendering as rendering


def test_converts_inline_and_display_math_to_unicode():
    source = (
        r"Inline $x^2 + \alpha$ and \(y_1\). "
        "Display $$\\frac{x}{y}$$ and \\[z^3\\]."
    )

    rendered = rendering.render_assistant_markdown(source)

    assert "𝑥² + α" in rendered
    assert "𝑦₁" in rendered
    assert "𝑥/𝑦" in rendered
    assert "𝑧³" in rendered
    assert "$" not in rendered


def test_converted_output_is_not_processed_as_another_formula():
    source = (
        r"$$L(h(\mathbf{x}_i), y_i) = -[y_i \log(\hat{y}_i) "
        r"+ (1-y_i) \log(1-\hat{y}_i)]$$"
    )

    rendered = rendering.render_assistant_markdown(source)

    assert r"\[𝑦ᵢ log(" in rendered
    assert "log(1 - 𝑦̂ᵢ)\\]" in rendered
    assert "𝑙𝑜𝑔" not in rendered


def test_preserves_currency_code_and_escaped_dollars():
    source = (
        "It costs $5 today and $10 tomorrow. `math = '$x$'` and \\$x$.\n\n"
        "```latex\n$x^2$\n```"
    )

    assert rendering.render_assistant_markdown(source) == source


def test_unclosed_or_failed_formula_is_left_unchanged(monkeypatch):
    assert rendering.render_assistant_markdown("unfinished $x + 1") == (
        "unfinished $x + 1"
    )

    def fail(_source):
        raise ValueError("unsupported formula")

    monkeypatch.setattr(rendering._CONVERTER, "latex_to_text", fail)
    assert rendering.render_assistant_markdown("formula $x$") == "formula $x$"
