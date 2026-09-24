import pytest
from framework.context.notes import NoteStore, note_tools


def test_write_read_update_list_roundtrip(tmp_path):
    store = NoteStore(tmp_path / "mem")
    assert "wrote" in store.write("PLAN.md", "goal: ship it")
    assert store.read("PLAN.md") == "goal: ship it"
    store.append("PLAN.md", "\nstep 1 done")
    assert "step 1 done" in store.read("PLAN.md")
    assert "PLAN.md" in store.list()


def test_read_missing_note_is_graceful(tmp_path):
    store = NoteStore(tmp_path / "mem")
    assert "not found" in store.read("nope.md")


@pytest.mark.parametrize("evil", ["../escape.txt", "../../etc/passwd", "/abs/path", "a/../../b"])
def test_path_traversal_and_absolute_are_rejected(tmp_path, evil):
    store = NoteStore(tmp_path / "mem")
    with pytest.raises(ValueError):
        store.write(evil, "malicious")


def test_tools_expose_the_four_operations_and_swallow_bad_paths(tmp_path):
    tools = note_tools(tmp_path / "mem")
    names = {t.name for t in tools}
    assert names == {"write_note", "read_note", "update_note", "list_notes"}
    write = next(t for t in tools if t.name == "write_note")
    # a traversal attempt returns a safe error string, not a raised exception
    assert "error" in write.invoke({"name": "../evil.txt", "content": "x"}).lower()
