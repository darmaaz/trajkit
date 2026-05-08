"""Frozen parameters for the L3 ``process`` runner."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from trajkit.clean import CleanParams, StaleMergeParams
from trajkit.embed import EmbedParams
from trajkit.episode import EpisodeParams
from trajkit.segment import SegmentParams


class RunParams(BaseModel):
    """Bundle of per-stage parameters for ``process``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    clean: CleanParams = Field(default_factory=CleanParams)
    run_stale_merge: bool = Field(
        default=False,
        description=(
            "When True, the clean stage runs ``clean`` followed by "
            "``merge_stale_positions``. Default is off — stale-position "
            "merging is opt-in per-deployment."
        ),
    )
    stale_merge: StaleMergeParams = Field(default_factory=StaleMergeParams)
    segment: SegmentParams = Field(default_factory=SegmentParams)
    episode: EpisodeParams = Field(default_factory=EpisodeParams)
    embed: EmbedParams = Field(default_factory=EmbedParams)

    @classmethod
    def from_preset(cls, name: str) -> RunParams:
        """Return the named domain preset from ``trajkit.presets``.

        Lazy-imports ``trajkit.presets`` to avoid a circular import at
        module load time.
        """
        from trajkit.presets import get_preset

        return get_preset(name)
