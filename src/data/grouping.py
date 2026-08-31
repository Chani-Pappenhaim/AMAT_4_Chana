"""Group-by-source-unit - the one implementation every layer must use
instead of grouping by filename or tile. EMPS groups by DOI, RODARE by
field (never by tile or detector channel - two channels of one field are
siblings, not two samples), NIST by intensity set. Used by L1's manifest,
the hidden-test guard, and L7's copying diagnostic / bootstrap-over-groups.

Pure functions only: no dependency on any loader, so this stays usable by
every dataset and every layer without a risk of circular imports.
"""


def group_id_for_emps(doi):
    """EMPS group id is its DOI - one group per source paper."""
    return doi


def group_id_for_rodare(field_id):
    """RODARE group id is the field (material/instrument/session)."""
    return field_id


def group_id_for_nist(set_id):
    """NIST group id is the intensity set - every noise/contrast variant
    drawn from one clean reference shares that reference's group.
    """
    return set_id


def build_group_of(rows):
    """rows: iterable of SourceManifestRow (or any object/dict with
    `source_id` and `group_id`). Returns {source_id: group_id}.
    """
    def get(row, key):
        return row[key] if isinstance(row, dict) else getattr(row, key)

    return {get(row, "source_id"): get(row, "group_id") for row in rows}


def assert_no_group_split_across_sets(group_of, **id_sets_by_name):
    """Raise if any group id appears in more than one named id set (e.g.
    train=..., validation=..., test=...) - the DOI/field-safe split
    invariant every later layer relies on.
    """
    owner_of_group = {}
    for set_name, ids in id_sets_by_name.items():
        for source_id in ids:
            group = group_of.get(source_id, source_id)
            prior_owner = owner_of_group.get(group)
            if prior_owner is not None and prior_owner != set_name:
                raise AssertionError(
                    f"group {group!r} (from source {source_id!r}) appears in "
                    f"both {prior_owner!r} and {set_name!r} - groups must "
                    "never be split across sets"
                )
            owner_of_group[group] = set_name


if __name__ == "__main__":
    group_of = {"a1": "doiA", "a2": "doiA", "b1": "doiB"}
    assert_no_group_split_across_sets(group_of, train=["a1", "a2"], test=["b1"])
    print("clean split (no group crosses a set boundary): passed")

    try:
        assert_no_group_split_across_sets(group_of, train=["a1"], test=["a2", "b1"])
        raise SystemExit("ERROR: a group split across train/test was NOT rejected!")
    except AssertionError as e:
        print(f"correctly rejected a group split across sets: {e}")
