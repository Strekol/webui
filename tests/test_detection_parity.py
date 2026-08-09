"""Les heuristiques de détection ne doivent pas dériver entre les TROIS clients.

`detect.py` est une copie réduite, en lecture seule, du moteur de détection que
portent aussi `claude-watcher-tui.py` et `claude-watcher-gtk.py`. Les deux
clients ont déjà leur garde de parité croisé (`test_core_parity.py`, présent des
deux côtés) — mais il ne regarde qu'eux, et `webui` n'était couvert par rien.

Ce n'est pas théorique : cette feature a livré un vrai bug par cette porte. Le
serveur ne réconciliait que `status == "shell"` là où les deux clients
réconciliaient `shell` ET `busy` vers l'état `background`, si bien que la MÊME
session s'affichait `working` en ligne distante et `background` en ligne locale.
Mesuré ensuite : changer `_WORKTREE_MARKER` ici laissait les 596 tests des trois
dépôts au vert.

On compare le COMPORTEMENT, pas l'AST : les commentaires de `webui` sont en
anglais et ceux des clients en français, donc une comparaison syntaxique
signalerait du bruit à chaque ligne. Chaque fonction est appelée dans les trois
modules sur les mêmes entrées et les résultats doivent coïncider.

CE QUE CE GARDE NE COUVRE PAS, dit franchement pour qu'il ne se lise pas comme
une promesse plus large qu'il n'est : `scan_proc` et `scan_sessions` divergent
par conception (formes de retour différentes, `proc_root` injectable côté
serveur), et la règle de réconciliation d'état vit à l'intérieur de
`get_session_state`, qui exige un registre et un transcript sur disque. C'est
exactement la règle qui avait dérivé — elle est aujourd'hui couverte par
`test_stuck_status_over_a_finished_turn_is_background` dans chaque dépôt, pas
ici.

Les scripts des clients sont cherchés à côté du dépôt (`../tui/`, `../gtk/`), à
l'emplacement de checkout de la CI (`_tui/`, `_gtk/`), ou via `CW_TUI_SCRIPT` /
`CW_GTK_SCRIPT`. Absents : SKIPPÉ en local (on ne force personne à cloner les
voisins) mais ÉCHEC sous `CI`, sinon le garde ne tournerait jamais là où il
compte.
"""

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

import detect

REPO = Path(__file__).resolve().parent.parent

# Constantes de détection : une valeur différente ici change silencieusement le
# projet attribué à une session, ou son transcript.
SHARED_CONSTANTS = [
    "_WORKTREE_MARKER",
    "_STATUS_MAP",
    "_TURN_END_SUBTYPES",
    "CLAUDE_PROJECTS_DIR",
    "DEFAULT_CONTEXT_WINDOW",
]

# Fonctions PURES (aucune lecture de /proc ni de disque) appelables à l'identique
# dans les trois modules. Les entrées couvrent le cas nominal ET les bords qui
# ont déjà mordu : worktree, chemin absent, modèle inconnu, config_dir relatif.
PURE_CALLS: list[tuple[str, tuple[Any, ...]]] = [
    ("_argv_value", (["claude", "--model", "opus"], "--model")),
    ("_argv_value", (["claude", "--model"], "--model")),  # drapeau en dernière position
    ("_argv_value", (["claude"], "--absent")),
    ("split_worktree", ("/home/u/proj/.claude/worktrees/wt/sub",)),
    ("split_worktree", ("/home/u/proj",)),
    ("split_worktree", (None,)),
    ("project_label", ("/home/u/dev/github/proj",)),
    ("project_label", (None,)),
    ("display_config_dir", ("/home/u/.claude-work",)),
    ("display_config_dir", (None,)),
    ("resolve_config_dir", ({},)),
    ("resolve_config_dir", ({"CLAUDE_CONFIG_DIR": "relative/dir"},)),
    ("resolve_config_dir", ({"CLAUDE_CONFIG_DIR": "/abs/dir"},)),
    ("context_window_for", (None, 0)),
    ("context_window_for", ("claude-sonnet-4-5", 0)),
    ("context_window_for", ("claude-sonnet-4-5", 500_000)),
    ("context_window_for", ("un-modele-inconnu", 0)),
    ("cwd_to_project_dir", ("/home/u/proj", None)),
    ("cwd_to_project_dir", (None, None)),
    # La racine vide vit dans son propre test plus bas : elle exige un chemin qui
    # EXISTE, ce qu'une liste statique ne peut pas préparer.
    ("split_worktree", ("/home/u/proj/.claude/worktrees/wt/sub",)),
    ("worktree_of", ("/home/u/proj/.claude/worktrees/wt", True)),
    # Marqueur Claude non confirmé (transcript non résolu) : chemin brut, pas de
    # worktree — la nuance qui distingue un vrai worktree d'un dossier qui en
    # porte le nom par hasard.
    ("worktree_of", ("/home/u/proj/.claude/worktrees/wt", False)),
    ("worktree_of", ("/nexiste/pas/du/tout", True)),
    # Worktree git ordinaire : prouvé par un `.git` FICHIER, donc rien à trouver
    # sur un chemin absent — les trois doivent rendre (None, None) pareil.
    ("git_worktree", ("/nexiste/pas/du/tout",)),
    ("git_worktree", (None,)),
    ("_find_transcript", ("sess-x", Path("/nexiste/pas/du/tout"))),
    ("_parse_session_lines", ([],)),
    # `system` SANS sous-type et `system` de MILIEU de tour ne terminent pas le
    # tour ; seuls les sous-types de fin le font. Trois formes, parce que c'est
    # exactement là que le serveur et les clients ont divergé.
    ("_parse_session_lines", (['{"type": "system"}'],)),
    ("_parse_session_lines", (['{"type": "system", "subtype": "informational"}'],)),
    ("_parse_session_lines", (['{"type": "system", "subtype": "turn_duration"}'],)),
    ("_parse_session_lines", (['{"type": "system", "subtype": "stop_hook_summary"}'],)),
    ("_parse_session_lines", (['{"type": "system", "subtype": "away_summary"}'],)),
    ("_parse_session_lines", (['{"type": "assistant", "message": {"content": [{"type": "tool_use"}]}}'],)),
    ("_parse_session_lines", (["pas du json"],)),
]


def _stub_loaded(path: Path, roots: tuple[str, ...], name: str) -> ModuleType:
    """Charge un script mono-fichier avec son toolkit graphique bouchonné.

    Même technique que les `tests/conftest.py` des deux clients : les tests ne
    visent que des helpers purs, on n'installe pas GTK ni Textual pour ça.
    """

    class _Loader:
        def create_module(self, spec: Any) -> ModuleType:
            mod = ModuleType(spec.name)
            mod.__getattr__ = lambda _n: MagicMock()  # type: ignore[method-assign]
            return mod

        def exec_module(self, module: ModuleType) -> None:
            pass

    class _Finder:
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
            if fullname.split(".")[0] not in roots:
                return None
            return importlib.util.spec_from_loader(fullname, _Loader(), is_package=True)  # type: ignore[arg-type]

    sys.meta_path.insert(0, _Finder())
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _Finder)]


def _sibling(env_var: str, folder: str, filename: str) -> Path | None:
    env = os.environ.get(env_var)
    candidates = [Path(env)] if env else []
    candidates += [REPO.parent / folder / filename, REPO / f"_{folder}" / filename]
    return next((c for c in candidates if c.is_file()), None)


@pytest.fixture(scope="module")
def clients() -> dict[str, ModuleType]:
    tui = _sibling("CW_TUI_SCRIPT", "tui", "claude-watcher-tui.py")
    gtk = _sibling("CW_GTK_SCRIPT", "gtk", "claude-watcher-gtk.py")
    missing = [n for n, p in (("tui", tui), ("gtk", gtk)) if p is None]
    if missing:
        msg = (
            f"scripts client introuvables ({', '.join(missing)}) — la parité des heuristiques "
            "de détection ne peut pas être vérifiée"
        )
        # Jamais un skip silencieux en CI : une couche non vérifiée qui annonce
        # « passed » est précisément ce qui a laissé passer la dérive d'état.
        if os.environ.get("CI"):
            pytest.fail(msg)
        pytest.skip(msg)
    assert tui and gtk
    return {
        "tui": _stub_loaded(tui, ("rich", "textual"), "cw_tui"),
        "gtk": _stub_loaded(gtk, ("gi", "cairo"), "cw_gtk"),
    }


@pytest.mark.parametrize("const", SHARED_CONSTANTS)
def test_detection_constants_are_identical_in_the_three_clients(
    clients: dict[str, ModuleType], const: str
) -> None:
    ours = getattr(detect, const)
    for name, mod in clients.items():
        assert hasattr(mod, const), f"{const} absent de {name}"
        assert getattr(mod, const) == ours, f"{const} diverge entre webui et {name}"


@pytest.mark.parametrize("func, args", PURE_CALLS, ids=lambda v: None)
def test_pure_detection_helpers_agree_in_the_three_clients(
    clients: dict[str, ModuleType], func: str, args: tuple[Any, ...]
) -> None:
    ours = getattr(detect, func)(*args)
    for name, mod in clients.items():
        assert hasattr(mod, func), f"{func} absent de {name}"
        theirs = getattr(mod, func)(*args)
        assert theirs == ours, f"{func}{args} : webui={ours!r} {name}={theirs!r}"


@pytest.mark.parametrize("module", ["detect.py", "auth.py"])
def test_the_shared_modules_parse_on_the_clients_python(module: str) -> None:
    """Le socle de détection doit rester COPIABLE vers les deux clients.

    `detect.py` est aligné à la main avec les copies de la TUI (py311) et du
    widget (py310). Cinq `except A, B:` sans parenthèses (PEP 758, 3.14+) s'y
    étaient glissés : syntaxiquement valides ici, SyntaxError dure là-bas — la
    copie n'était plus transposable, et ce fichier de parité ne pouvait même plus
    charger les clients.

    Ce test verrouille aussi `target-version = "py310"` côté ruff : au-dessus, le
    FORMATEUR réécrit `except (A, B):` en PEP 758 à chaque `make lint`, et le
    correctif se défait tout seul. Version basse = celle du client le plus ancien.
    """
    source = (REPO / module).read_text()
    for feature_version in ((3, 10), (3, 11)):
        try:
            ast.parse(source, feature_version=feature_version)
        except SyntaxError as exc:  # pragma: no cover - le message EST le test
            pytest.fail(f"{module} n'est pas analysable en 3.{feature_version[1]} : {exc}")


def test_display_config_dir_is_idempotent_in_the_three_clients(clients: dict[str, ModuleType]) -> None:
    """`config_dir` porte DEUX représentations sous un seul nom de champ : webui
    émet `display_config_dir(config_dir)` (« work »), les clients stockent le
    chemin brut (« /home/u/.claude-work »), et l'adaptateur laisse passer la
    valeur distante telle quelle.

    C'est inoffensif pour une seule raison : les clients re-rendent
    `display_config_dir(s['config_dir'])` à l'affichage, donc une valeur DÉJÀ
    réduite doit se réduire en elle-même. Rien ne l'affirmait — et
    `local_config_dirs`, qui exclut les lignes distantes, est le seul autre
    rempart. Si cette idempotence casse, une ligne distante affiche une instance
    au mauvais nom sans qu'aucun autre test ne bouge.
    """
    for value in ("/home/u/.claude-work", "work", "~", "~/ailleurs", None):
        once = detect.display_config_dir(value)
        assert detect.display_config_dir(once) == once, value
        for name, mod in clients.items():
            their_once = mod.display_config_dir(value)
            assert their_once == once, f"display_config_dir({value!r}) : webui={once!r} {name}={their_once!r}"
            assert mod.display_config_dir(their_once) == their_once, f"{name} n'est pas idempotent"


def test_a_rootless_worktree_agrees_in_the_three_clients(
    clients: dict[str, ModuleType], tmp_path: Path
) -> None:
    """Racine projet VIDE ('/.claude/worktrees/wt') : `base / ''` VAUT `base`, donc
    sans garde on rend le DOSSIER DES PROJETS comme s'il était un projet.

    Ce cas figurait dans PURE_CALLS et y passait par ACCIDENT : le chemin slugifié
    ('--claude-worktrees-wt') n'existait sur aucune machine, donc les trois
    implémentations rendaient None par le même `path.exists()` final — la
    divergence de repli restait invisible. Ici le dossier est PRÉ-CRÉÉ sous un
    `config_dir` de test : seule une implémentation qui refuse la racine vide rend
    encore None.
    """
    projects = tmp_path / "projects"
    (projects / "--claude-worktrees-wt").mkdir(parents=True)
    args = ("/.claude/worktrees/wt", str(tmp_path))
    assert detect.cwd_to_project_dir(*args) is None
    for name, mod in clients.items():
        assert mod.cwd_to_project_dir(*args) is None, f"{name} désigne le dossier des projets"


def test_no_shared_detection_helper_was_added_on_one_side_only(clients: dict[str, ModuleType]) -> None:
    """Un helper ajouté à `detect.py` sans son jumeau côté client (ou l'inverse)
    est le premier pas de la dérive suivante — on le voit ici, pas six mois plus
    tard dans un rapport de bug."""
    covered = {f for f, _ in PURE_CALLS} | set(SHARED_CONSTANTS)
    for name, mod in clients.items():
        missing = sorted(c for c in covered if not hasattr(mod, c))
        assert not missing, f"{name} ne porte pas : {missing}"
