from glob import escape as glob_escape
from pathlib import Path


class Workspace:
    """The user's RF project directory.

    Layout: artifacts/{captures,renders,pipelines,scenes}/ — created on demand.
    Artifacts are exchanged as paths into this directory, never as blobs.
    """

    #: Parent folder gathering every generated subdir under the workspace root.
    _ARTIFACTS = "artifacts"

    def __init__(self, root: Path | str = ".") -> None:
        self.root = Path(root)

    def _subdir(self, name: str, *, create: bool = True) -> Path:
        d = self.root / self._ARTIFACTS / name
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def scenes_dir(self) -> Path:
        """The scenes directory path, without creating it — for startup
        device reload, which must not materialize an empty tree on first run."""
        return self._subdir("scenes", create=False)

    @staticmethod
    def _dedupe(directory: Path, stem: str, probe_suffix: str, suffix: str) -> Path:
        """Return a collision-free path under *directory*.

        Two checks per candidate:
        - *probe* — ``glob(escaped_candidate + probe_suffix + "*")`` catches any
          sidecar files that share the same base name (e.g. ``.sigmf-meta`` /
          ``.sigmf-data`` for SigMF pairs, or a bare ``.yaml`` alongside others).
        - *exact* — ``(directory / (candidate + suffix)).exists()`` catches the
          primary output file itself when *probe_suffix == suffix*.

        ``glob_escape`` is applied to the candidate before globbing so that names
        containing glob metacharacters (e.g. ``ISM[2.4GHz]``) are treated as
        literals rather than patterns.
        """
        candidate = stem
        i = 0
        while (
            next(directory.glob(glob_escape(candidate) + probe_suffix + "*"), None)
            is not None
            or (directory / (candidate + suffix)).exists()
        ):
            i += 1
            candidate = f"{stem}-{i}"
        return directory / (candidate + suffix)

    def new_capture_path(self, name: str) -> Path:
        """Extension-less base path for a SigMF pair under captures/."""
        return self._dedupe(self._subdir("captures"), name, ".sigmf", "")

    def new_render_path(self, name: str) -> Path:
        return self._dedupe(self._subdir("renders"), name, ".png", ".png")

    def new_scene_path(self, name: str) -> Path:
        return self._dedupe(self._subdir("scenes"), name, ".yaml", ".yaml")

    def new_pipeline_path(self, name: str) -> Path:
        return self._dedupe(self._subdir("pipelines"), name, ".yaml", ".yaml")

    def new_grc_path(self, name: str) -> Path:
        """Collision-free .grc path under pipelines/ — never overwrites a
        flowgraph the user may have hand-tweaked in GNU Radio Companion."""
        return self._dedupe(self._subdir("pipelines"), name, ".grc", ".grc")

    def scene_file(self, device_id: str) -> Path:
        """The canonical (deterministic, not deduped) scene path for a device —
        scenes/<device_id>.yaml. Keyed on the id so re-registering overwrites
        the same file; this is the one place the convention lives."""
        return self._subdir("scenes") / f"{device_id}.yaml"
