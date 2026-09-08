from __future__ import annotations
'Canonical directory layout for the single paper post-freeze analysis tree.'
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class PaperOutputLayout:
    root: Path
    manifest: Path
    substrate: Path
    hypothesis_tests: Path
    contract_diagnostics: Path
    paper_assets: Path
    extensions: Path
    registry: Path
    verification: Path

    @property
    def b2_gap(self) -> Path:
        return self.substrate / 'b2_gap_decomposition'

    @property
    def test3(self) -> Path:
        return self.substrate / 'test3_counterfactual_fidelity'

    @property
    def structural_slice(self) -> Path:
        return self.substrate / 'structural_event_slice'

    @property
    def holm(self) -> Path:
        return self.hypothesis_tests / 'holm_h1_h3'

    @property
    def reference_quality_acceptance(self) -> Path:
        return self.hypothesis_tests / 'reference_quality_acceptance'

    @property
    def ablation(self) -> Path:
        return self.contract_diagnostics / 'ablation'

    @property
    def signflip(self) -> Path:
        return self.contract_diagnostics / 'signflip_mean_null'

    @property
    def n5_holm(self) -> Path:
        return self.contract_diagnostics / 'n5_budget_holm'

    @property
    def n5_budget_frontier_holm(self) -> Path:
        return self.contract_diagnostics / 'n5_generation_budget_frontier_holm'

    @property
    def n5_matched_budget_frontier_holm(self) -> Path:
        return self.contract_diagnostics / 'n5_matched_budget_frontier_holm'

    @property
    def n5m_posthoc(self) -> Path:
        return self.contract_diagnostics / 'n5m_posthoc'

    @property
    def n5m_adaptive_selection(self) -> Path:
        return self.contract_diagnostics / 'n5m_adaptive_selection'

    @property
    def main_harness_backend_decomposition(self) -> Path:
        return self.contract_diagnostics / 'main_harness_backend_decomposition'

    @property
    def frontier(self) -> Path:
        return self.contract_diagnostics / 'budget_frontier'

    @property
    def winrate(self) -> Path:
        return self.contract_diagnostics / 'winrate_heterogeneity'

    @property
    def icc_probe(self) -> Path:
        return self.contract_diagnostics / 'icc_probe'

    @property
    def tables(self) -> Path:
        return self.paper_assets / 'tables'

    @property
    def figures(self) -> Path:
        return self.paper_assets / 'figures'

    @property
    def visual_plot_data(self) -> Path:
        """Exact processed data used to render human-facing figures."""
        return self.paper_assets / 'plot_data'

    @property
    def visual_registry(self) -> Path:
        """Visual asset inventory, lineage, and claim-evidence mapping."""
        return self.registry / 'visual_assets'

    @property
    def visual_asset_manifests(self) -> Path:
        return self.visual_registry / 'asset_manifests'

    @property
    def e2(self) -> Path:
        return self.extensions / 'e2_c4r_matched'

    @property
    def e3(self) -> Path:
        return self.extensions / 'e3_c4r_journal'

    @property
    def e4(self) -> Path:
        return self.extensions / 'e4_haiku_characterization'

def build_layout(root: Path) -> PaperOutputLayout:
    root = Path(root).resolve()
    return PaperOutputLayout(root=root, manifest=root / '00_manifest', substrate=root / '01_substrate_validation', hypothesis_tests=root / '02_llm_hypothesis_tests', contract_diagnostics=root / '03_output_contract_diagnostics', paper_assets=root / '04_paper_assets', extensions=root / '05_extension_e3_e4', registry=root / '06_thesis_registry', verification=root / '99_verification')

def ensure_layout(layout: PaperOutputLayout) -> None:
    for p in (layout.root, layout.manifest, layout.substrate, layout.hypothesis_tests, layout.contract_diagnostics, layout.paper_assets, layout.extensions, layout.registry, layout.verification):
        p.mkdir(parents=True, exist_ok=True)
