"""The caveats of BS Profiler 3.1 (design 11) — the only source of these texts.

Every text is kept here word for word; the page, the characterization, the MBTI tab, the PDF and the journal take them
from this module and never write their own. C1 and C2 are the disclaimers of report.py (one string for the whole
copy). C13 is a template: `c13(k, n, model)` fills it and chooses the Russian word forms, so the numbers and the
model name in the text always agree with the data. No caveat mentions a group of processed videos: levels and letters
use the absolute scale 0…1 (change of 2026-09-26). One model per analysis (3.1): C6 and C7 exist in one Russian
version, C7 says that the type is the type of the one model that ran; C18 (the second strip of 3.0) is gone.

Where each caveat goes (design 11):
- characterization: C11 (paragraph 4), C14, C15, C12, when needed C13, C19, C20;
- tab «Тип MBTI» and section 2 of the PDF: C3, C4, C5, C6, C7, C9, C16 (MBTI_READ); next to the letter strip C8,
  C19; the language-model note C17;
- «Как получены оценки»: C13; page footer: C1, C10, C3 (PAGE_FOOTER); tab «Данные»: C22; PDF «Как читать
  результаты»: C1, C10, C11, C14, C15 (PDF_HOW_TO_READ); no Big Five at all: C21 instead of the MBTI block;
- C2 (the label «собеседование» of AMLAI 1.0) goes only where that label is shown: under the score bars of the page
  and in the PDF «Как читать результаты» of a job that carries the label (3.1: a job of AMLAI 1.0).
"""
from __future__ import annotations

from . import MODEL_TITLES, PRODUCT
from .report import DISCLAIMER_RU, INTERVIEW_DISCLAIMER_RU
from .textfmt import plural_ru

C1 = DISCLAIMER_RU
C2 = INTERVIEW_DISCLAIMER_RU

C3 = ("Тип MBTI здесь не измерен опросником: он пересчитан из четырёх шкал Big Five (экстраверсия → E–I, открытость "
      "опыту → S–N, доброжелательность → T–F, добросовестность → J–P) по опубликованным соответствиям шкал (McCrae, "
      "Costa, 1989). Система не определяет тип личности: она оценивает пять непрерывных шкал по видео и записывает "
      "четыре из них в нотации MBTI. Обратного перевода нет: по четырём буквам нельзя восстановить пять шкал, поэтому "
      "основой отчёта остаются шкалы Big Five.")
C4 = ("Оси E–I и S–N связаны со шкалами Big Five сильно (r ≈ 0.74 и 0.72), T–F и J–P — умеренно (r ≈ 0.44 и 0.49), "
      "поэтому буквы T/F и J/P ошибаются заметно чаще. Эти корреляции получены на самоотчётах по опросникам MBTI и "
      "NEO-PI, а не на оценках по видео: это соответствие шкал, а не точность нашей оценки.")
C5 = ("Ось «на границе» — оценка от 0.36 до 0.64, то есть отстоит от середины шкалы 0.5 меньше чем на "
      "0.15: небольшое изменение оценки поменяло бы букву. В записи с учётом границ такая ось обозначена X, рядом показана буква, которая "
      "получилась бы при строгом делении.")
C6 = ("Граница между буквами — середина шкалы 0.5: оценка модели от 0 до 1 не ниже 0.5 даёт первую букву пары "
      "(E, N, F, J), ниже 0.5 — вторую (I, S, T, P). Буква E здесь означает, что оценка экстраверсии по первому "
      "впечатлению выше середины шкалы модели, а не принадлежность к экстравертам вообще.")
C7 = ("Тип посчитан по оценкам одной модели — той, что выбрана для анализа: OCEAN-AI (веса MuPTA) или AMLAI 1.0. "
      "Модели обучены на разных данных и работают на разных шкалах, поэтому типы, полученные разными моделями по одному "
      "ролику, — два разных описания, и напрямую их сравнивать не следует.")
C8 = ("Тип по отрезкам считается по тем же порогам, что и тип за весь ролик, но оценка одного 20-секундного отрезка "
      "шумнее оценки всего ролика: смотрите, насколько устойчивы буквы, а не на отдельный отрезок. Если буква на оси "
      "часто меняется, вывод по этой оси неустойчив.")
C9 = ("Уверенность по оси — не вероятность того, что буква верна, а удалённость оценки от границы: 0 — на самой "
      "границе 0.5, 1 — у края шкалы (0 или 1).")
C10 = ("Эмоции, голос и мимика — сигналы отдельных моделей, обученных на англоязычных корпусах и фотографиях; это "
       "наблюдения о поведении на видео, а не диагноз. В оценку черт они не входят и сами по себе черт не доказывают.")
C11 = ("По первому впечатлению лучше всего считывается экстраверсия, хуже всего — эмоциональная стабильность, поэтому "
       "выводы о нейротизме самые осторожные.")
C12 = ("Описания уровней показывают, как обычно выглядит поведение, создающее такое впечатление, а не пересказывают "
       "эпизоды этого ролика.")
C13 = ("В {k} {segments_k} из {n} модель {model} не дала оценки (чаще всего не распознаны речь или лицо); {these} в "
       "основные оценки, характеристику и тип и на графиках {shown}.")
C14 = ("Это описание того, как человек воспринимается по одной записи, — рабочая гипотеза для беседы, а не "
       "психологическое заключение и не основание для решений о человеке.")
C15 = "Ни характеристика, ни тип MBTI не являются оценкой пригодности человека к работе или учёбе."
C16 = "Названия типов условные и приведены для удобства чтения; в разных источниках они различаются."
C17 = ("Текст написан локальной языковой моделью по уже посчитанным числам и проверен программой на совпадение букв и "
       "чисел; тип она не пересчитывает, это вспомогательная формулировка, а не отдельный вывод.")
C19 = "Ролик короче 30 с оценивается целиком, одним отрезком, поэтому типа по ходу ролика нет."
# an older job (3.0 or imported from 2.0) whose recorded model OCEAN-AI gave no scores at all: the view shows the
# scores of AMLAI 1.0 that the job carries (scores.main_system; a 3.1 job holds one model and never falls back)
C20 = ("Модель OCEAN-AI не дала оценок по этому ролику, поэтому характеристика и тип построены по оценкам модели "
       "AMLAI 1.0, сохранённым в этом задании.")
C21 = "Тип MBTI не рассчитан: в результате нет оценок Big Five."
C22 = (f"Раздел mbti для этого задания посчитан при показе версией {PRODUCT.split()[-1]} и в файл result.json "
       "не записан.")

TEXTS = {
    "C1": C1, "C2": C2, "C3": C3, "C4": C4, "C5": C5, "C6": C6, "C7": C7,
    "C8": C8, "C9": C9, "C10": C10, "C11": C11, "C12": C12, "C13": C13, "C14": C14, "C15": C15, "C16": C16,
    "C17": C17, "C19": C19, "C20": C20, "C21": C21, "C22": C22,
}
CODES = tuple(f"C{i}" for i in range(1, 23) if i != 18)      # C18 (the second strip of 3.0) is gone with 3.1

# The lists of caveats printed as one block, in their order (design 11). «Как читать результаты» at the foot of the
# page: the footer is built once, before any analysis, so it carries only what holds for both models; C2 (the label
# «собеседование» of AMLAI 1.0) stands under the score bars of a job that shows that label (web.parts._bar_html)
PAGE_FOOTER = ("C1", "C10", "C3")
# «Как читать результаты» of the PDF; C2 is printed only when the report carries the label «собеседование»
PDF_HOW_TO_READ = ("C1", "C2", "C10", "C11", "C14", "C15")
# «Как читать тип MBTI» on the tab «Тип MBTI» and in section 2 of the PDF
MBTI_READ = ("C3", "C4", "C5", "C6", "C7", "C9", "C16")


def text(code: str, **values) -> str:
    """The caveat `code` ('C1' … 'C22' without C18). Templates are filled with `values` (C13: k, segments_k, n, model,
    these, shown); the helper c13 computes these values itself."""
    t = TEXTS[code]
    return t.format(**values) if values else t


def c13(k: int, n: int, model: str = "oceanai") -> str:
    """«В 7 отрезках из 33 модель OCEAN-AI не дала оценки …; эти отрезки не вошли … показаны пропусками.»
    `model`: the internal key of the model the view shows ("oceanai" | "mm"), named by its title."""
    one = int(k) == 1
    return C13.format(k=k, segments_k=plural_ru(k, "отрезке", "отрезках", "отрезках"), n=n,
                      model=MODEL_TITLES.get(model, str(model)),
                      these="этот отрезок не вошёл" if one else "эти отрезки не вошли",
                      shown="показан пропуском" if one else "показаны пропусками")
