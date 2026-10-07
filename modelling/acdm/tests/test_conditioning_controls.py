import pytest

from modelling.acdm.conditional_edm.config import EDMConfig
from modelling.acdm.conditional_edm.train import build_parser


def test_conditioning_mode_is_validated_and_exposed_by_cli():
    with pytest.raises(ValueError, match="conditioning_mode"):
        EDMConfig(reduced_dim=2, conditioning_mode="unknown")
    args = build_parser().parse_args(["--conditioning-mode", "clean",
                                      "--diffusion-formulation", "edm"])
    assert args.conditioning_mode == "clean" and args.diffusion_formulation == "edm"
    with pytest.raises(ValueError, match="Kohl DDPM"):
        EDMConfig(reduced_dim=2, diffusion_formulation="ddpm", conditioning_mode="clean")
