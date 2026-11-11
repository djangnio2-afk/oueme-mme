# -*- coding: utf-8 -*-
"""Figure styling, applied at save time; importing the module is enough.

Titles are set in Times New Roman, and type sizes are normalized to the size
the reader actually sees. Each figure is placed at a fixed width in the
manuscript, so a 24-inch figure is reduced by a factor 0.35 and a 7-inch one
enlarged by 1.18: at equal font size in the script, the first would appear
three times smaller on the page. Each figure's own reduction factor is
computed here, and the fonts scaled so that the apparent size is the same
everywhere.
"""
import re

from matplotlib.figure import Figure

TIMES = "Times New Roman"   # présente sous Windows ; pas de repli bruyant
LARGEUR_DOC = 600.0         # points ; identique à la constante LARG de build.js
HAUTEUR_DOC = 636.0         # points ; hauteur utile d'une page, cf. build.js

# Tailles VOULUES SUR LA PAGE, en points. Le corps du texte est à 11 pt.
VU = {"suptitre": 13.0, "titre": 12.0, "axes": 10.5,
      "graduations": 9.5, "legende": 9.5, "annotations": 8.5}

# Toutes les manières dont la période d'évaluation s'est glissée dans les
# titres : « (test 2006-2016) », « , test period 2006–2016 », « (test) », etc.
_PER = re.compile(
    r"\s*[,(—-]?\s*\(?\s*test(?:\s*[- ]?period)?"
    r"(?:\s*\(?\s*2006\s*[-–]\s*2016\s*\)?)?\s*\)?\s*[,;]?\s*", re.I)
_VIDE = re.compile(r"\(\s*\)|\(\s*,|,\s*\)")

# La période reste écrite lorsque la figure oppose calage et validation.
_GARDE = ("calibration", "training", "cal.", "reversed", "over-fitting",
          "overfitting")


def _nettoie(txt, contexte=""):
    if not txt:
        return txt
    if any(g in (txt + " " + contexte).lower() for g in _GARDE):
        return txt
    t = txt.split("\n")[0]          # une 2e ligne ne porte que de l'explication
    t = _PER.sub(" ", t)
    t = t.replace("(, ", "(").replace(" ,", ",")
    t = _VIDE.sub(lambda m: "" if m.group(0) in ("()", "( )") else
                  ("(" if m.group(0).startswith("(") else ")"), t)
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t.rstrip(" ,;:—-").strip()


def _echelle(fig):
    """Facteur de réduction subi par la figure une fois posée dans le document."""
    try:
        l_po, h_po = fig.get_size_inches()[:2]
        # build.js réduit une figure trop haute pour qu'elle tienne dans la page
        pose_pt = min(LARGEUR_DOC, HAUTEUR_DOC * l_po / h_po)
        return max(0.15, min(2.0, pose_pt / (l_po * 72.0)))
    except Exception:
        return 1.0


def _plancher(obj, cible):
    try:
        if obj.get_fontsize() < cible:
            obj.set_fontsize(cible)
    except Exception:
        pass


def _exact(obj, cible):
    try:
        obj.set_fontsize(cible)
    except Exception:
        pass


def _lisibilite(ax, k):
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        _plancher(lab, VU["graduations"] / k)
    for lab in (ax.xaxis.label, ax.yaxis.label):
        if lab.get_text():
            _plancher(lab, VU["axes"] / k)
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            _plancher(t, VU["legende"] / k)
        if leg.get_title() is not None and leg.get_title().get_text():
            _plancher(leg.get_title(), VU["legende"] / k)
    # Les annotations d'une grille dense (valeurs de cases d'une carte de
    # chaleur, étiquettes de barres) ont été dimensionnées pour tenir dans la
    # place disponible : les remonter les ferait se chevaucher. On ne relève
    # que les annotations peu nombreuses, qui sont des commentaires libres.
    if len(ax.texts) <= 24:
        for t in ax.texts:
            _plancher(t, VU["annotations"] / k)


_orig = Figure.savefig


def savefig(self, *a, **kw):
    k = _echelle(self)

    for ax in self.axes:
        _lisibilite(ax, k)
    for leg in getattr(self, "legends", []):
        for t in leg.get_texts():
            _plancher(t, VU["legende"] / k)
        if leg.get_title() is not None and leg.get_title().get_text():
            _plancher(leg.get_title(), VU["legende"] / k)

    for ax in self.axes:
        # contexte de l'axe : ce que portent l'autre étiquette et le titre
        ctx = " ".join([ax.get_xlabel(), ax.get_ylabel(), ax.get_title(),
                        ax._left_title.get_text()])
        for obj in (ax.title, ax._left_title, ax._right_title):
            if not obj.get_text():
                continue
            obj.set_text(_nettoie(obj.get_text(), ctx))
            obj.set_fontfamily(TIMES)
            obj.set_fontweight("bold")
            _exact(obj, VU["titre"] / k)
        # un titre calé à gauche revient au centre. On recopie le texte dans
        # l'objet central plutôt que d'appeler set_title, qui remettrait à zéro
        # l'écart au cadre et ferait retomber le titre sur le graphique.
        if ax._left_title.get_text() and not ax.title.get_text():
            lt = ax._left_title
            ax.title.set_text(lt.get_text())
            ax.title.set_fontfamily(TIMES)
            ax.title.set_fontweight("bold")
            _exact(ax.title, VU["titre"] / k)
            lt.set_text("")
        for lab in (ax.xaxis.label, ax.yaxis.label):
            lab.set_text(_nettoie(lab.get_text(), ctx))

    sup = getattr(self, "_suptitle", None)
    if sup is not None and sup.get_text():
        sup.set_text(_nettoie(sup.get_text()))
        sup.set_fontfamily(TIMES)
        sup.set_fontweight("bold")
        sup.set_ha("center")
        sup.set_position((0.5, sup.get_position()[1]))
        _exact(sup, VU["suptitre"] / k)

    return _orig(self, *a, **kw)


Figure.savefig = savefig
