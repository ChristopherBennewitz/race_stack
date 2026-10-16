from id_controller import common


def test_default_bag_directory_uses_repository_root(tmp_path):
    repository = tmp_path / "race_stack"
    nested = repository / "system_identification" / "id_controller"
    nested.mkdir(parents=True)
    (repository / ".git").mkdir()
    assert common.default_bag_directory(nested) == str(
        repository / "sysid_bags")


def test_default_bag_directory_honours_container_root(monkeypatch):
    monkeypatch.setenv("RACE_STACK_ROOT", "/container/ws/src/race_stack")
    assert common.default_bag_directory() == (
        "/container/ws/src/race_stack/sysid_bags")
