from splbert.data import TAC2019Dataset
from splbert.tests.constants import (
    TEST_LABEL_TEXT_SEQUENCES,
    TEST_LABEL_MENTION_SEQUENCES,
    TEST_LABEL_RELATION_SEQUENCES,
)


def test_read_xml_files():
    (
        text_sequences,
        mention_sequences,
        relation_sequences,
    ) = TAC2019Dataset.read_xml_files("splbert/tests/test_fixtures")

    # See constants.py for the expected output from this function
    assert text_sequences == TEST_LABEL_TEXT_SEQUENCES
    assert mention_sequences == TEST_LABEL_MENTION_SEQUENCES
    assert relation_sequences == TEST_LABEL_RELATION_SEQUENCES
