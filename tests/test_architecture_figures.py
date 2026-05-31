from pathlib import Path


REQUIRED_DOT_FILES = {
    "fig1_overall_govtrust_fl_architecture.dot",
    "fig2_external_validation_protocol.dot",
    "fig3_federated_learning_workflow.dot",
    "fig4_trustworthiness_evaluation_framework.dot",
    "fig5_pedi_workflow.dot",
}


def test_architecture_dot_sources_exist_and_are_nontrivial():
    dot_dir = Path("paper_figures") / "architecture" / "dot"

    assert dot_dir.exists()
    actual = {path.name for path in dot_dir.glob("*.dot")}
    assert actual == REQUIRED_DOT_FILES

    for filename in REQUIRED_DOT_FILES:
        text = (dot_dir / filename).read_text(encoding="utf-8")
        assert "digraph" in text
        assert len(text) > 500
