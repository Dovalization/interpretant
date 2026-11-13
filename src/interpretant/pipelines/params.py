"""The pipelines' settings: one section of ``params.yaml`` each, which keyword arguments override.

The file is the record of a run's settings (decision agenda, step 9: DVC tracks it as parameters);
environment variables never change them.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PARAMS_FILE = Path(__file__).resolve().parents[3] / "params.yaml"


class ParamsSettings(BaseSettings):
    """A section of ``params.yaml``, named by each subclass's ``yaml_config_section``."""

    model_config = SettingsConfigDict(extra="forbid", frozen=True, yaml_file=PARAMS_FILE)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Keyword arguments first, then the parameters file; no environment variables."""
        return init_settings, YamlConfigSettingsSource(settings_cls)


class CorpusSelection(ParamsSettings):
    """The ``corpus`` section: the fields and decades whose raw cells make up the corpus."""

    model_config = SettingsConfigDict(yaml_config_section="corpus")

    fields: tuple[str, ...] = Field(min_length=1)
    decades: tuple[int, ...] = Field(min_length=1)
