"""Entity extraction for pre-retrieval filtering (offline)."""

from framework.rag.router import extract_entities

VOCAB = {
    "punjab": "Punjab", "sindh": "Sindh", "balochistan": "Balochistan",
    "khyber pakhtunkhwa": "Khyber Pakhtunkhwa", "kp": "Khyber Pakhtunkhwa",
    "islamabad capital territory": "Islamabad Capital Territory", "ict": "Islamabad Capital Territory",
    "hrcp stands": "HRCP stands", "hrcp stance": "HRCP stands",
}


def test_year_and_chapter():
    e = extract_entities("how many prisoners on death row in Punjab 2024?", VOCAB)
    assert e["years"] == {"2024"}
    assert e["chapters"] == {"Punjab"}


def test_multiple_years():
    e = extract_entities("compare 2023 and 2024 death sentences", VOCAB)
    assert e["years"] == {"2023", "2024"}
    assert e["chapters"] == set()


def test_alias_and_possessive():
    assert extract_entities("death row in KP", VOCAB)["chapters"] == {"Khyber Pakhtunkhwa"}
    assert extract_entities("what are HRCP's stands?", VOCAB)["chapters"] == {"HRCP stands"}  # possessive
    assert extract_entities("ICT prison data", VOCAB)["chapters"] == {"Islamabad Capital Territory"}


def test_no_partial_word_match():
    # 'kp' must not fire on an unrelated word; no chapter/year here.
    e = extract_entities("the capital of France", VOCAB)
    assert e["years"] == set() and e["chapters"] == set()


def test_no_vocab_means_only_years():
    assert extract_entities("death row in Punjab 2024?")["chapters"] == set()
